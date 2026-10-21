# Safe Lookahead：把 host 重叠深度当作可调度资源

更新：2026-10-22。定位：**单卡、每天约两小时、只做一个方向**，代码落在 vLLM、SGLang、FlashInfer、FlashAttention 等主流仓库。本文把此前的三个候选收敛为一个锚定方向，并给出一项可证伪的测量门槛、单卡实验、与相邻工作的差异和止损条件。

本文是选题与实验决策，不是已有结果。本项目仍无目标 GPU 的实测数据；文中所有公开 issue/PR 的数字都属于报告者，不是我们的测量。

> **2026-10-22 复审修订（回应 Codex review）**：本版收缩为一项可证伪的测量与机制实验。主要改动：明确 `step` 是引擎调度步；首个实验改为测实际领先深度、等待原因与 GPU 空闲；修正 CUDA Graph 的节点参数更新/并发语义；把 `S_max` 降为筛选模型；首阶段只做 vLLM + FlashInfer 的 plan buffer；删除"转 checker 仍可发 ATC/TPDS"的预设。逐条回应见附录。

## 摘要

一句话：研究动态 LLM 服务里 **host 最多能安全地领先 GPU 多少步（safe lookahead `k`）**，它受什么约束，以及如何在有限显存下把 `k` 顶到最大。

这里的 `step` 指引擎的一次 batch 调度/执行步（scheduler 为当前 batch 准备一次 forward，device 执行一次），不是同一请求的逐 token 生成步。**第一阶段的交付物不是控制器，而是证明目标路径上确实存在可安全利用的 `k>1`，且简单修复后仍有性能空间。**

核心问题不是"buffer 归谁"，而是"**重叠深度是多少、能不能控制**"。版本/所有权协议只是实现手段；标题和 claim 统一用 **Safe Lookahead / overlap depth**，不用 "versioned execution" 或 "state ownership"——这两个词在 2026 年已被相邻工作占用（Retire、Execution-State Capsules）。

三个候选的归宿：

| 原方向 | 在 Safe Lookahead 中的角色 | 是否独立立项 |
|---|---|---|
| ① 增量式、异步安全的 Attention Plan | plan buffer 的生命周期案例 | 否 |
| ② 混合有状态模型的图安全 Prefill | recurrent state slot 的图内版本案例 | 否；GDN 图捕获已有公开在途 PR（未合入），存在撞车风险 |
| ③ 混合模型的角色感知缓存与执行 | KV block / draft pool 的回收案例 | 否；上游 POC 与 2026 论文已覆盖 |

## 一、为什么是 Safe Lookahead

### 1.1 三个方向共同的问题

三者都是同一个矛盾在不同 artifact 上的表现：

- 引擎每一步都在改变 batch、KV 页、recurrent state 和投机解码的验证形状；
- 算子却希望调度计划、内存地址和 workspace 在若干步内保持稳定，最好能整段录进 CUDA Graph；
- 引擎用"加同步 / 双缓冲 / 关闭 async"来保平安，代价是 GPU 等待、显存浪费和功能组合被禁用。

把这三件事抽象一层，问题是同一个：**host 想跑在 GPU 前面，但能跑多远取决于可变状态还能保留几份。** 这个"多远"就是 Safe Lookahead。

### 1.2 为什么换名字

- "ownership / versioned execution" 已被 [Retire](https://arxiv.org/abs/2610.01160)（2026-10）用于请求级 authority 与异步回收；
- "KV 回收 / 虚拟化" 已被 [vToken](https://arxiv.org/abs/2608.13263)（2026-08）等占据；
- "图绑定状态" 已被 [Execution-State Capsules](https://arxiv.org/abs/2606.20537)（2026-06）用于 checkpoint/restore。

继续用这些词，审稿人第一反应就是"和它们有什么区别"。换成 **Safe Lookahead** 后，单位是 **step**（不是 request），目标是 **稳态重叠**（不是修订、驱逐或快照），差异是结构性的。

### 1.3 本文不主张什么

- 不主张新的 attention 算法，也不主张更快的 kernel inner loop；
- 不主张请求级版本化或 agent 修订（Retire 的领域）；
- 不主张新的 KV 驱逐/压缩策略（vToken、DASC 的领域）；
- 不主张图绑定 checkpoint/restore/fork（Capsules 的领域）。

**主张的是**：稳态执行中 `k` 的上界、资源/依赖两类约束的分解、单图多版本算子 ABI、以及把 `k` 顶到最大的控制器。

## 二、问题定义

### 2.1 执行模型

本文的 `step` 指引擎的一次 batch 调度步，不是同一请求的逐 token 步。同一请求的下一个 token 依赖上一步的采样结果（`k_dep=1`），但不同请求的 batch 组合、plan 构建、block 回收仍可能领先；是否真的能领先、能领先多少，必须实测，不能由流水线公式推定。

一个 decode step 中：

- **host**：调度、构建 attention metadata/plan、分配与回收 KV block、采样后处理；
- **device**：attention、GEMM/MoE、recurrent state 更新；
- **async scheduling** 允许 host 领先 device 若干步；
- **CUDA Graph** 在捕获时固定 kernel 节点的参数与地址；回放不走常规 Python/分配路径，但参数可在两次 replay 之间用 CUDA API 更新（见 3.4）。

可变执行状态分四类：

| artifact | 典型大小 | 变化频率 | 当前生命周期规则 |
|---|---|---|---|
| attention plan / metadata buffer | 小（KB–MB） | 每步 | wrapper 内复用；host 写、device 读 |
| KV block / page table | 大（GB） | 请求级 | block manager 分配/回收；图内读地址 |
| recurrent state slot | 中（MB） | 请求/chunk 级 | state pool 按 slot 复用 |
| graph workspace（MoE/cuDNN/scratch） | 中 | 捕获时固定 | 预分配；禁止图内 resize |

系统里同时存在**三个分配器**：Python 对象池、引擎的 block manager、CUDA/CUDA Graph 内存池。捕获期间的内存分配进入 graph 的私有池，replay 不走常规分配路径；因此 block manager 与 graph 池的生命周期必须显式对齐，"用 stream-ordered allocator 就好了"并不成立。

### 2.2 三类失败模式

- **WAR**：host 为第 N+k 步重写 buffer，而第 N 步的 kernel 仍在读它（vLLM #61239）。
- **UAF**：block/slot 被提前回收并复用，图内 kernel 仍持有旧地址（vLLM #45357、FlashInfer #4553/#4827）。
- **RAW**：新版本的 plan 被旧消费者读到，或旧版本被新消费者误用（#59493 的 staging buffer race）。

这三类问题不是"某个 buffer 的 bug"，而是 **`k` 超过该 artifact 能保留的份数**时的必然结果。Safe Lookahead 把这个条件显式化。

### 2.3 现有做法及其边界

| 做法 | 代表 | 边界 |
|---|---|---|
| 加同步 | `seq_lens.cpu()` | 正确但阻塞 host，抵消 async 收益 |
| pageable staging | vLLM #59493 | 只保护一个 staging buffer |
| 延迟 block free | vLLM #45357 | 只覆盖 PD consumer 场景 |
| 把 build 录进图 | vLLM #62749 | 只覆盖 uniform decode；mixed/ragged 仍走 host |
| 逐 workspace 修补 | FlashInfer #4553/#4827 | 每个新 workspace 都要再修一次 |

结论：都是点修，没有"能跑多前"的模型，也没有控制器。

### 2.4 研究问题

- **RQ1**：在目标负载上，引擎实际能维持多少 in-flight step（`k_actual`）？host 在等什么（采样 token、acceptance count、block free、event、Python/GIL）？GPU 空闲区间有多长、由谁造成？H/D 与 `S_max` 只作为上界参考。
- **RQ2**：如何用 per-artifact 生命周期策略，以最小显存维持最大安全 `k`？
- **RQ3**：单图多版本 kernel ABI 能否在不重新捕获的前提下服务 `k` 个 in-flight step？
- **RQ4**：自适应 `k` 能否在显存与 SLO 约束下跑赢静态 `k=1/2/4`？
- **RQ5**：协议能否同时消除已知 race 类，并在两个引擎/两个后端上泛化？

## 三、核心创新：L1 → L3

| 层次 | 主张 | 评审视角 |
|---|---|---|
| L1 正确性 | 给 plan/block/state 加版本，消除 race | "RCU/双缓冲的工程应用"；不够 |
| L2 筛选模型 | 重叠有上界 `(H+D)/max(H,D)`；`k_max` 受资源与依赖约束 | 标准两阶段流水上界，只够筛选负载，不够当贡献 |
| L3 机制+控制 | 单图多版本 kernel ABI + lifetime-aware allocator + 自适应 `k` | 算子与引擎联合；够 ASPLOS/EuroSys 主轨 |

**所有权是手段，`S_max` 是筛选工具，二者都不是贡献本身。** 真正可能形成贡献的是：找出生产引擎中阻止领先的具体依赖，给出可证明安全、且明显优于 pageable staging / 同步 / 静态 ring 等简单修复的机制，并量化"简单修复之后仍剩下多少 gap"。

### 3.1 筛选模型（不是贡献）

设每步 host 时间为 H，device 时间为 D。

```
无重叠：      T_no   = H + D
理想流水：    T_pipe = max(H, D)
加速上界：    S_max  = (H + D) / max(H, D)
```

| H/D | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|
| S_max | 1.10× | 1.20× | 1.50× | 2.00× |

这是标准两阶段流水上界，只用于筛选负载：如果目标负载上 H/D < 0.1，后面的机制工作都不值得做。实际流水还要扣掉依赖停顿和 `k` 不足，所以真实可达值低于 `S_max`。**但 `S_max` 不是论文贡献**；第一个实验要测的是 `k_actual`、等待原因和 GPU 空闲，而不是只测 H/D。

### 3.2 两类约束必须分开

```
资源约束：k_res = min_a floor(slots_a / writes_per_step_a)
显存约束：sum_a slots_a * size_a <= M
依赖约束：k_dep = 下一步需要多少步之前的 device 结果
最大安全深度：k_max = min(k_res, k_dep)
```

- **资源约束**来自 buffer 槽位和显存：plan buffer 小，可以多版本；KV block 大，只能按 event 延迟回收。
- **依赖约束**来自"下一步要等上一步的 device 结果"：spec decode 的 acceptance count、采样 token、prefix 命中判定都是例子。
- 现有引擎把两者混成"async on/off"；拆开之后才能分别优化：资源约束用版本池解决，依赖约束用 upper-bound/padding、device-side 规划或延迟采样解决。

### 3.3 artifact 分类与策略（lifetime-aware allocator）

| artifact | 策略 | 理由 |
|---|---|---|
| plan / metadata buffer | `k` 版本 ring buffer + device 版本计数 | 小，适合多版本；host 写、device 读 |
| KV block / page table | event-deferred free + per-block epoch | 大，不能整份多版本；只能延迟回收 |
| recurrent state slot | slot pool + epoch；preemption 时延迟复用 | 中等，按请求/chunk 生命周期 |
| graph workspace | 捕获前预分配 + 版本槽；禁止图内 resize | 地址被图冻结 |

**不要一刀切 double buffer。** 静态双缓冲把 `k` 锁死在 2、对大 artifact 显存翻倍，而且 spec decode 需要 `k>2`。论文必须画出"`k` vs 显存 vs TPOT"的 Pareto 曲线，证明静态双缓冲只是曲线上的一个点。

### 3.4 算子侧 ABI：单图多版本的设计空间

先修正一个过强的说法：CUDA Graph **不是**"kernel 参数完全冻结"。CUDA 提供 `cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate`，可以在两次 replay 之间更新 kernel 节点参数；也可以 clone 出多个 `graphExec`。真正的约束有两条：

- 同一个 `graphExec` 不能与自身并发执行；要并发服务多个 in-flight step，需要多个 exec 克隆或设备端间接寻址；
- `cudaGraphExecUpdate` 不能在 exec 正在执行时安全调用；host 必须等它完成，或改用独立 exec，否则会引入新的竞态。

因此至少有四种方案，必须实测比较，不能预设哪一种更好：

| 方案 | 机制 | 代价 / 风险 |
|---|---|---|
| 每次 replay 更新节点参数 | host 调 `SetParams` | host 调用与排序；可能抵消领先收益 |
| `k` 个 graphExec 克隆 | 每个 slot 固定指针 | 捕获与显存开销；实现简单 |
| 静态 ring + device 版本计数 | 一张图，kernel 读版本 | 需要定义 commit/read 顺序；调试难 |
| eager | 不走图 | 无图开销，但 launch 开销高 |

本方向要测的是四者在 `k`、TPOT、显存和正确性上的差异。若采用 device 版本计数，必须说清：版本计数器由哪个 node 递增、kernel 在哪个点读取、host 写 slot `v+1` 与 device 读 slot `v` 如何用 event 排序。第一版只改 FlashInfer 的 decode plan，验证"一张图服务 `k` 个 in-flight step"；再扩到 prefill/mixed。

### 3.5 控制器

给定显存预算 M 和 SLO，选择 `k`：

- 观测：H/D、队列深度、preemption 次数、in-flight 版本数；
- 策略：静态 `k`、贪心取 `k_max`、带反馈的自适应；
- 目标：在不超过 M 和 SLO 的前提下把 `k` 顶到 `k_max`；
- 对照：async off（H+D）、当前 async（`k≈1`）、静态 `k=2/4`、本协议自适应 `k`。

没有控制器，就只是"更大的 double buffer"。

## 四、学术与工业价值、谁在做、拥挤度

### 4.1 学术价值

- **一个筛选模型 + 一个依赖分类**：`S_max` 只用来判断"这个负载值不值得做"；真正的贡献是把阻止 `k>1` 的依赖拆成资源、采样依赖、acceptance、event、Python/GIL 等可测类别，并指出哪些可移除。
- **一个跨层机制**：host Python + C++ scheduler + device kernel + CUDA Graph 四层，配合单图多版本 ABI。这是算子与引擎联合优化，不是纯工具。
- **一个可复现的 checker + bug corpus**：单卡就能复现 WAR/UAF/RAW 三类违反。
- **venue**：只有在机制相对简单修复有稳定优势时，ATC/EuroSys/ASPLOS 才合理；OSDI/SOSP 偏难；TPDS 是稳妥的期刊路径。引用量不会爆，但会被后续引擎论文引用。

### 4.2 工业价值

- **可靠性**：#61239 这类问题在生产里表现为 IMA/hang，报错位置和根因相隔很远，排查成本极高。一个低开销 checker 比一套协议更容易被 vLLM/SGLang 采纳。
- **性能**：低延迟、小 batch、spec decode、hybrid、preemption 这些区间，H/D 可能到 0.2–0.5，重叠理论上有 10–30% 空间；**前提是实测 `k>1` 且简单修复后仍有残余 gap**，否则收益为 0。大 batch 8B 上只有几个点，要如实报告"这里不值得做"。
- **工程成本**：现在是每个 buffer 打一个补丁（defer free、pageable staging、don't resize workspace），协议化能减少重复修补。
- **采纳路径**：checker/debug 工具最容易进上游；协议重写难，但可以作为可选路径。

### 4.3 谁在做

| 子方向 | 代表工作 | 状态 |
|---|---|---|
| 请求级 versioned execution / authority | [Retire](https://arxiv.org/abs/2610.01160)（2026-10） | 概念最接近，但单位是 request，不是 step |
| KV 回收 / 虚拟化 | [vToken](https://arxiv.org/abs/2608.13263)（2026-08）、LMCache、PagedAttention | 拥挤；vToken 兼容 CUDA Graph |
| 图绑定状态 checkpoint/restore | [Execution-State Capsules](https://arxiv.org/abs/2606.20537)（2026-06） | 相邻；不是稳态重叠 |
| 异步复用与资源权威 | [Memory-Sovereign Inference](https://arxiv.org/abs/2608.23805)（2026-08） | 相邻；重点是 residency/exactness |
| 引擎点修 | vLLM #61239/#59493/#45357/#62749、FlashInfer #4553/#4827 | 工程侧非常拥挤，每周都在动 |
| **重叠深度建模与控制** | 未检索到直接同题论文 | **低**，但"没检索到"不等于空白 |

### 4.4 拥挤度判断

| 子方向 | 拥挤度 | 判断 |
|---|---|---|
| 请求级 versioned execution | 高 | "versioned execution" 这个词不要用 |
| KV 回收/虚拟化 | 高 | 不要提新的驱逐策略 |
| 图绑定状态 checkpoint/restore | 中高 | 不要做 checkpoint/fork |
| 引擎点修 | 很高 | 不要以"修 bug"作为贡献 |
| **Safe Lookahead / overlap depth** | **低** | 这是切入点 |

**窗口大约 6–12 个月。** 相邻工作出现得很快；一旦工业实验室发一篇"safe async execution"，空间会被压缩。现在动手，2027 年中之前要形成 MPU。

## 五、与相邻工作的差异

| 工作 | 单位 / 目标 | 与 Safe Lookahead 的差异 |
|---|---|---|
| [Retire](https://arxiv.org/abs/2610.01160) | request / agent 修订与 authority | 不做 host 前视步数，不做图内多版本 kernel ABI |
| [vToken](https://arxiv.org/abs/2608.13263) | token / KV 回收 | 不提出新的驱逐策略；只研究 block 回收对 `k` 的约束 |
| [Execution-State Capsules](https://arxiv.org/abs/2606.20537) | execution boundary / checkpoint | 不做 checkpoint/restore/fork；研究稳态重叠 |
| [Memory-Sovereign](https://arxiv.org/abs/2608.23805) | residency / exact reuse | 不做 host/device pipeline 的 `k` 上界与控制 |
| vLLM #62749 | decode step / metadata build | 单一形态；没有上界模型、没有控制器 |
| **本方向** | **step / 最大化安全 `k`** | **筛选模型 + 依赖分类 + 约束分解 + 单图多版本 ABI + 控制器** |

## 六、审稿人质疑与回答

| 质疑 | 回答 |
|---|---|
| 为什么不直接 double buffer？ | 它把 `k` 锁死在 2；KV block 无法整份双缓冲；spec decode 需要 `k>2`；用 Pareto 曲线证明它只是特例。 |
| CUDA stream 不是已经保证顺序了吗？ | stream 只排序 device work，不排序 host 对 pinned buffer 的写，也不管 Python 对象复用；CUDA Graph 在捕获时固定地址、replay 不走常规分配路径；engine 的 block manager 是第三个分配器。三个分配器，一个契约。 |
| 同一个 graphExec 能并发跑多个 step 吗？ | 不能；需要 `k` 个 exec 克隆，或一张图 + 设备端版本间接寻址。第 3.4 节列出四种方案并实测。 |
| 这和 Retire 有什么区别？ | Retire 是请求级 authority/失效，单位是 request；本方向单位是 step，研究 `k_actual` 与 `k` 的上界/控制，不涉及图内多版本 kernel ABI。 |
| 这和 vToken / Capsules 有什么区别？ | vToken 是 KV 驱逐与虚拟化，Capsules 是 checkpoint/restore；本方向不提出新的驱逐或快照机制，只研究稳态重叠深度。 |
| 上游不是已经在修了吗？ | #45357、#59493、#62749 都是点修，没有模型、没有控制器、不覆盖组合场景。论文量化的是"点修之后还剩多少 gap"。 |
| effect size 太小。 | 只打 H/D 高的区间：小 batch、短输出、spec decode、hybrid、preemption。大 batch 8B 上 H/D 可能 <0.1，直接报告"这里不值得做"，反而增加可信度。 |
| 这不就是 RCU/EBR？ | EBR 假设单一分配器、没有冻结指针；这里是 host Python + C++ scheduler + device kernel + graph replay 四层跨边界，且要控制 lookahead，不只是回收。 |
| checker 能证明正确吗？ | 它是动态检查器，不是证明；配不变量 + 压力测试 + 误报率，不声称 sound。 |
| 只做单卡够吗？ | 核心 claim 都是 per-device 的；多卡是 future work。单卡是受控实验，不是缺陷。 |

## 七、截至 2026-10-22 的证据

### 7.1 工程证据

| 日期 | 证据 | 内容 | 对本方向的作用 |
|---|---|---|---|
| 2026-10-20 | vLLM [#62749](https://github.com/vllm-project/vllm/pull/62749) | 把 decode 的 metadata build 录进 CUDA Graph；`prepare_attn` 103→23 µs；小模型 step −8%~−19% | decode 侧 host 成本可被图消除；说明 `k` 的上界在变化 |
| 2026-10-16 | vLLM [#61952](https://github.com/vllm-project/vllm/pull/61952) | GDN core 捕获进 breakable graph；8×A100；host 99→55 ms；TPOT P50 −9~10% | recurrent state 的图内版本案例 |
| 2026-10-15 | vLLM [#61779](https://github.com/vllm-project/vllm/pull/61779) | 混合 batch 的 GDN metadata 全部持久化 padded buffer + static grid | 固定地址状态槽的工程先例 |
| 2026-09-30 | vLLM [#59493](https://github.com/vllm-project/vllm/pull/59493) | FlashInfer plan staging buffer 改为 pageable，修复用竞态 | WAR 点修；说明缺少 `k` 的统一模型 |
| 2026-10-12 | vLLM [#61239](https://github.com/vllm-project/vllm/issues/61239) | async scheduling 下 plan 覆盖导致 IMA/hang；issue 提出 per-in-flight-step workspace | WAR 的主案例；单卡可复现 |
| 已合并 | vLLM [#45357](https://github.com/vllm-project/vllm/pull/45357) | async + PD consumer 下延迟 block free | UAF 点修；泛化空间仍在 |
| 2026-07-22 | vLLM [#49488](https://github.com/vllm-project/vllm/issues/49488) | 已关闭，被 MRv2 #29559 取代 | 旧文档引用需修正：不再是 2026 的活跃痛点 |
| 2026 | FlashInfer [#4553](https://github.com/flashinfer-ai/flashinfer/pull/4553) / [#4827](https://github.com/flashinfer-ai/flashinfer/pull/4827) | cuDNN GEMM workspace resize、MoE workspace evict 导致图内 UAF | workspace 生命周期的第二类案例 |
| 2026-08-21 | SGLang [#35851](https://github.com/sgl-project/sglang/issues/35851) | GDN prefill 每层约 590 µs idle，整步约 50% GPU 空闲；单 kernel 融合端到端 0% | recurrent state 案例的动机 |

### 7.2 学术证据

| 工作 | venue / 时间 | 与本方向的关系 |
|---|---|---|
| PagedAttention | SOSP 2023 | 分页 KV；block 生命周期问题的起点 |
| RadixAttention | NeurIPS 2024 | 前缀复用；block 复用更激进 |
| FlashAttention / FA2 | NeurIPS 2022 / ICLR 2024 | 无状态 attention 基线 |
| FlashInfer | MLSys 2025 | plan/run 分离与负载均衡；本方向的主要算子面 |
| LeanAttention | MLSys 2025 | decode 分块；不属于所有权问题 |
| vAttention | ASPLOS 2025 | 虚拟内存 KV；与 block 回收相邻 |
| Marconi | MLSys 2025 | hybrid prefix state 复用；案例 2 的近邻 |
| NanoFlow | OSDI 2025 | 同卡重叠；上界模型可借鉴 |
| Event Tensor | MLSys 2026 | 动态图融合；说明"融合"不是本方向 |
| [Execution-State Capsules](https://arxiv.org/abs/2606.20537) | arXiv 2026-06 | 图绑定状态快照；相邻但非稳态重叠 |
| [HYPIC](https://arxiv.org/abs/2607.01299) | arXiv 2026-07 | hybrid position-independent caching；案例 3 的近邻 |
| [DASC](https://arxiv.org/abs/2608.30386) | arXiv 2026-08 | GDN/KDA state 压缩；案例 3 的近邻 |
| [vToken](https://arxiv.org/abs/2608.13263) | arXiv 2026-08 | KV 回收虚拟化；案例 3 的近邻 |
| [Memory-Sovereign](https://arxiv.org/abs/2608.23805) | arXiv 2026-08 | 异步复用与资源权威；相邻 |
| [Retire](https://arxiv.org/abs/2610.01160) | arXiv 2026-10 | 请求级 versioned execution；最接近但单位不同 |
| HybridCache | NeurIPS 2026 | linear–softmax prefix caching；案例 3 的近邻 |
| LLM-42 / CoRun | 2026 | determinism / batch invariance；相邻但问题不同 |

### 7.3 证据等级

已正式发表：Marconi（MLSys 2025）、FlashInfer（MLSys 2025）、PagedAttention（SOSP 2023）、RadixAttention（NeurIPS 2024）等。预印本：HYPIC、DASC、vToken、Capsules、Memory-Sovereign、Retire。工程证据：上述 vLLM/SGLang/FlashInfer 的 PR/issue，反映截至 2026-10-22 的公开实现与报告者测量。本文假设：筛选模型、`k_max` 分解、单图多版本 ABI、控制器，均待 Gate 0 校准。

## 八、单卡实验设计

### 8.1 用卡

| 阶段 | 卡 | 用途 | 约束 |
|---|---|---|---|
| 第 1–2 个月 | **A100 PCIe 40GB** | Qwen3-8B BF16、async/graph、race 复现、H/D、lookahead | 无 FP8；GDN 走 vLLM Triton/FLA 路径 |
| 第 3–4 个月 | **A800 80GB** | Qwen3.5-9B hybrid、长上下文、preemption 压力 | 仍 SM80；结论可与 A100 对齐 |
| 交叉验证 | **H20 96GB** | FP8、SM90 FlashInfer GDN、跨架构复现 | 算力偏低，不用于"更快"结论 |
| 不租 | H100/H200/多卡 | — | 贵，核心 claim 不需要 |

### 8.2 负载矩阵

| 维度 | 取值 |
|---|---|
| 引擎 | vLLM 0.31 + MRv2；SGLang 0.5.16 |
| 后端 | FlashInfer；FlashAttention（无状态对照） |
| 模型 | Qwen3-8B（dense）；Qwen3.5-9B（hybrid） |
| 模式 | async on/off；graph eager/piecewise/full；spec on/off；preemption on/off |
| 负载 | B=1/8/32；短/长输出；chunked prefill；prefix 命中；4k 单请求；高 churn |
| 指标 | H、D、H/D、`S_max`、`k_max`、实际 `k`、GPU idle、TPOT P50/P95、吞吐、显存、violation 数 |

### 8.3 对照组

1. async off（H+D 基线）；
2. 当前 async（`k≈1`）；
3. 静态 `k=2/4` ring；
4. 简单点修：#59493 式 pageable staging、#45357 式 defer free；
5. 每次 replay 更新节点参数（`SetParams`）；
6. `k` 个 graphExec 克隆；
7. 本协议（版本池 + 控制器）；
8. 理论上界 `S_max`（只作参考，不是对照组）。

Gate 0 只比较 1–4 与 5/6，不做控制器；7 留到 Stage 1。

### 8.4 统计与记录

- 每个配置至少 3 轮、交错运行、报 median + IQR；尾延迟单独报；
- 用 `VLLM_GPU_SYNC_CHECK=error` 抓隐藏同步；
- 记录 commit hash、模型配置、warmup、图模式、显存、请求生成方法与 raw trace 的校验和；
- 不把单次 nsys 的 idle 比例当结论；不把报告者的 H200/8×A100 数字外推到 A100。

## 九、Gate 0：先证明 k>1 有利可图（4 周）

范围只做 **vLLM + FlashInfer + plan buffer**。不做 KV、recurrent state、SGLang，也不写控制器。

| 周 | 任务 | 产出 |
|---|---|---|
| 1 | 锁版本、建环境；写 per-step instrumentation：host timeline、in-flight `k`、等待原因、GPU 空闲 | 一张"谁在等谁"的时序图 |
| 2 | 复现 #61239；测 `k_actual` 在 B=1/8/32、chunked prefill、spec on/off 下的分布 | `k_actual` 分布与等待原因统计 |
| 3 | 五类对照：同步、pageable staging、静态 ring `k=2/4`、`SetParams`、`k` 个 exec 克隆 | 正确性、TPOT、显存、`k` 的对比 |
| 4 | 判定是否存在"简单修复后仍剩下"的 gap | Gate 0 通过/停止 |

**Gate 0 的建议阈值（待实测校准）**：

- 至少一种目标负载上 `k_actual > 1`；
- 可归因于 plan/metadata/event 的 GPU 空闲超过 step 的 5%；
- 简单修复后仍有 3–5% 的残余 gap；
- 机制相对最强简单修复有稳定优势。

四个条件缺一，就不进入 Stage 1。

代码入口：

- vLLM：`vllm/v1/attention/backends/flashinfer.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/gpu/attn_metadata_cudagraphs.py`；
- FlashInfer：`include/flashinfer/attention/scheduler.cuh`、`decode.py` 的 `plan()`；
- SGLang（Stage 3）：`python/sglang/srt/layers/attention/flashinfer_backend.py`、`runner_backend_utils/breakable_cuda_graph/`。

## 十、最小可发表单元与 venue

**Stage 1（论文核心）**：vLLM + FlashInfer + Qwen3-8B + A100 40GB；`k_actual` 与等待原因的 characterization；四种图内多版本方案对比；一个安全机制；1–2 类 bug 复现；端到端 TPOT/吞吐。

**Stage 2（泛化）**：加入 KV block 或 recurrent state；自适应 `k` 控制器；第二类模型。

**Stage 3（跨引擎/工具）**：SGLang 或 FlashAttention 复现；checker；bug corpus。

论文 MPU = Stage 1 + Stage 2，加一次跨引擎检查；Stage 3 是加分项，不是首阶段目标。

**Venue**：

- CCF-A 会议：ATC 2027、EuroSys 2028、ASPLOS 2028；OSDI/SOSP 需要团队与集群，不作为当前目标。
- MLSys 不是 CCF-A；MLSys 2027 的 deadline 是 2026-10-30，不现实。
- CCF-A 期刊：TPDS 是最现实的"顶刊"路径；TC/TOS/TOCS 可作备选。

**时间**：Gate 0 通过后再估算；按每天约 2 小时，9–12 个月形成 Stage 1 是乐观上限。

## 十一、止损条件

1. **`k_actual` 在目标负载上普遍只能到 1，或简单修复后残余 gap < 3%** → 性能故事不成立；checker/characterization 是否够 ATC/TPDS 需要重新评估，不能预设。
2. **Gate 0 的四个条件缺一** → 不进入 Stage 1；先补测量或换题，不要用更大范围掩盖核心假设未成立。
3. **单图多版本 ABI 做不动** → 退到 plan ring buffer + event-deferred KV free，仍是一篇机制论文，但天花板降到 MLSys/ATC。
4. **上游先发了 lookahead 控制器** → 需要重新定位；可能只剩测量/工具价值，不能假设仍有论文贡献。

## 十二、与旧文档的关系

2026-10-21 的两份三方向文档（选题总览、跨年复审）已在 2026-10-22 收敛后移除；仍然有效的文献先例压缩在本文第七节，作废的结论记录如下：

- "#49488 仍为活跃痛点"作废：该 issue 已于 2026-07-22 关闭，被 MRv2 #29559 取代。
- "首选题是增量 Attention Plan"作废：由本文的 Safe Lookahead 取代；① 的归因实验保留为案例 1。
- "② 作为强备选、③ 作为储备"作废：两者分别成为 recurrent state slot 与 KV block 回收案例，不再独立立项。

## 十三、待办

- [ ] 锁定 vLLM/FlashInfer/SGLang 的 commit 与镜像；
- [ ] 写 H/D 采集脚本与 6 种负载配置；
- [ ] 复现 #61239（4,097-token 单请求、chunked prefill、fp8 KV、async 默认开）；
- [ ] 读 #62749、#61952、#45357 的实现，标注可复用与不可复用部分；
- [ ] 设计 plan ring buffer 的版本计数与 commit kernel；
- [ ] 决定第一版 `k` 控制器是静态扫描还是反馈控制；
- [ ] 记录每次租卡的 GPU 小时与实验目的。

## 十四、证据等级约定

- **已发表**：会议/期刊正式论文；
- **预印本**：arXiv，仅作相邻工作与撞车风险；
- **工程证据**：官方仓库的 PR/issue，含报告者测量，需在目标卡独立复现；
- **本文假设**：筛选模型、`k_max` 分解、单图多版本 ABI、控制器、预期收益；均待 Gate 0 校准。

未检索到同题论文不等于证明新颖性；所有公开数字在引用时必须标注来源与硬件条件。

## 附录：对 2026-10-22 review 的逐条回应

| # | 意见 | 处理 |
|---|---|---|
| 1 | 先证明目标路径有可利用的 `k>1`；`step` 定义不清 | 2.1 明确 `step` 是引擎 batch 调度步；RQ1 改为 `k_actual`/等待原因/GPU 空闲；新增 Gate 0 |
| 2 | CUDA Graph 论证有误 | 3.4 改为四种设计空间，补充 `SetParams`、exec 克隆、同一 exec 不能并发；2.1 修正"绕过分配器" |
| 3 | 上界不足以支撑贡献 | 3.1 降为筛选模型；贡献改为依赖分类 + 安全机制 + 相对简单修复的残余 gap |
| 4 | 实施范围过大 | 首阶段只做 plan buffer；MPU 拆为 Stage 1/2/3；Gate 0 只有 4 周 |
| 5 | "抢跑"措辞易误解 | README 与正文改为"公开在途 PR（未合入）" |
| 6 | "转 checker 仍可发 ATC/TPDS"无保证 | 止损条件改为"重新评估"，不预设 venue |
