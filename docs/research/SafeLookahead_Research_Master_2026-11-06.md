# Safe Lookahead：研究总文档

> **最后更新：2026-11-06**  
> **状态：Gate 0 / Research Validation，尚未通过**  
> **本文件是 docs/research/ 的唯一研究主文档。** 旧版选题深化、论文初稿、章节蓝图、证据表和阶段分析已合并到本文；历史版本仍可从 Git commit history 查看。  
> **证据纪律：** 区分已完成实验、一次性 smoke、研究假设和未来计划；未验证的机制不得写成已实现贡献，微基准不得外推为端到端推理加速。

---

## 0. 当前结论

Safe Lookahead 的核心问题是：

**CUDA Graph 逐步接管 Host 常驻工作之后，大模型推理引擎剩余的 Host–Device 重叠深度 `k` 由什么决定？它受到资源生命周期还是执行依赖的限制？在有限显存下，能否安全地提高这一深度，并转化为端到端 serving 收益？**

当前已经具备：

- 相对明确的问题定义：`k_actual`、`k_res`、`k_dep`；
- 候选机制：Plan / Metadata 多版本生命周期管理，以及 CUDA Graph 与版本槽的绑定；
- H20 上的 CUDA Graph executable / 外部 buffer 隔离微基准；
- 一次 vLLM mixed workload smoke；
- 论文工作稿的章节和文本框架；
- Gate 0 实验脚本与证据记录。

当前还没有：

- 正式、重复的负载归因结果；
- Nsight Systems 校准后的 GPU 空闲时间线；
- clean multiprocessing 对照；
- 在真实 vLLM 路径中实现的多版本机制；
- 与强基线相比的端到端 goodput / TTFT / TPOT 收益。

因此，**当前不能声称 Safe Lookahead 已经证明有效，也不能声称核心机制已实现。下一步的唯一优先级是完成 Gate 0。**

## 1. Idea 与研究动机

### 1.1 一句话解释

CPU 在 GPU 执行当前推理步骤时，会尝试提前准备后续步骤。但后续步骤可能需要使用仍被 GPU 消费的 Plan、Metadata、Workspace 等资源。如果 CPU 太早覆盖或释放这些资源，就会产生竞态；如果为避免竞态而同步等待，又会损失 CPU–GPU overlap。

Safe Lookahead 研究的就是：

**CPU 最多可以安全地领先 GPU 几个调度 step？限制这个深度的原因是什么？能否用更精细的资源生命周期管理提高安全深度，而不是简单地同步或固定双缓冲？**

这里的 `step` 是引擎的一次 batch 调度 / 执行步，不是单个请求生成一个 token。

### 1.2 为什么在 CUDA Graph 之后研究

CUDA Graph 能减少重复的 Host launch、metadata build 和 Python 调度开销，但它并没有自动解决所有动态状态的生命周期问题。

图化之后仍可能存在不能完全入图、地址固定或随请求变化的对象，例如：

- Attention Plan / Metadata Buffer；
- KV Block / Page Table；
- Recurrent State Slot；
- Graph Workspace。

因此，研究重点从“Host 有多慢”调整为：

**图化之后剩余的 `k` 由哪些可变对象和执行依赖共同限制？**

这也包含一个必须面对的悖论：如果 CUDA Graph 已经把 Host 常驻工作隐藏得很好，那么即使能够把 `k` 从 2 提高到 4，也未必有性能收益。必须同时测量残余不可图化 Host 工作和真实 GPU idle，不能只看 Host 函数耗时。

### 1.3 研究范围与不主张

首阶段聚焦 **vLLM + FlashInfer + Plan / Metadata Buffer**。

不主张：

- 新的 Attention 算法或 Kernel inner-loop 优化；
- 新的 KV eviction / compression 算法；
- 请求级 versioned execution；
- 图绑定 checkpoint / restore / fork；
- 仅仅修复某一个 Buffer Race 就构成系统论文贡献。

原先三个候选方向的归宿：

| 原候选方向 | 在 Safe Lookahead 中的角色 | 是否独立立项 |
|---|---|---|
| 增量式、异步安全的 Attention Plan | Plan Buffer 生命周期案例 | 否 |
| 混合有状态模型的图安全 Prefill | Recurrent State Slot 扩展案例 | 否，公开在途工作存在撞车风险 |
| 混合模型的角色感知缓存与执行 | KV Block / Draft Pool 回收扩展案例 | 否，相关工作较拥挤 |

## 2. 问题模型

### 2.1 执行路径

典型推理执行路径可抽象为：

`request → tokenizer → scheduler / batch formation → model runner → prefill / decode → attention / KV cache → sampling → response`

本研究聚焦其中的 Host–Device 执行边界：

- **Host：** Scheduler、Batch 准备、Attention Metadata / Plan 构建、资源分配与回收、采样结果处理；
- **Device：** Attention、GEMM / MoE、Recurrent State 更新等 GPU 执行；
- **Async Scheduling：** 允许 Host 在 GPU 执行当前 step 时准备后续 step；
- **CUDA Graph：** 固定图执行期间的节点和地址约束，减少重复 Host 开销，但不自动保证图外 Host 写入与图内 Device 读取的生命周期正确。

### 2.2 三个核心量

- `k_actual`：某一时刻真实处于 in-flight 状态、已准备但 Device 尚未完成消费的调度步骤数。
- `k_res`：由资源槽位、对象生命周期和显存预算决定的最大安全深度。
- `k_dep`：由前序 Device 结果依赖决定的最大可提前深度。

候选约束关系：

`k_max ≤ min(k_res, k_dep)`

这是一种待实测校准的归因模型，不应仅凭 Host 调度函数被调用的次数推断实际 in-flight 深度。必须对齐 Host 日志、CUDA Event 和 GPU 时间线。

### 2.3 两类约束

**资源约束（`k_res`）**

- Plan / Metadata Buffer 仍被旧 step 使用；
- Workspace 地址或 Graph 引用仍然有效；
- 旧消费者尚未完成，槽位不能被覆盖；
- 显存预算限制可以同时保留多少个版本。

**依赖约束（`k_dep`）**

- 后续步骤需要前一步生成的 sampled token；
- Speculative Decoding 需要 acceptance count；
- Prefix 命中、调度决策或其他运行时结果尚未产生；
- 后续步骤在逻辑上无法提前确定。

多版本 Buffer 只能缓解 `k_res`，不能消除真正的 `k_dep`。如果 `k_dep` 主导，增加 Buffer 数量不会解决问题。

### 2.4 三类典型 Failure Mode

| 类型 | 发生方式 | 典型后果 |
|---|---|---|
| WAR | Host 为后续 step 重写 Buffer，但 GPU 仍在读取旧内容 | 数据污染、Illegal Memory Access、Hang |
| UAF | Block / Slot 被提前回收并复用，Graph / Kernel 仍持有旧地址 | 非法访问、错误结果 |
| RAW | 新版本 Plan 被旧消费者读取，或新旧版本绑定错乱 | Metadata 不一致、执行错误 |

### 2.5 资源生命周期的基本不变量

至少需要保证：

1. 新版本写入先于对应 Device 读取；
2. 旧版本所有 Device 消费完成，才允许槽位复用；
3. Graph / Kernel 引用的地址在消费期间保持有效；
4. 每个 step 读取的版本必须与其提交的版本一致。

系统涉及多个层次的分配和复用：Python 对象池、引擎 C++ / KV Block Manager、CUDA / CUDA Graph 内存池。不能假设单一 allocator 或 stream ordering 自动覆盖所有层次。

## 3. Research Questions 与候选创新

### 3.1 Research Questions

- **RQ1：** 目标负载下实际 `k_actual` 是多少？`k` 的损失分别有多少来自 `k_res`、`k_dep`、Python / GIL、队列上限或其他实现因素？
- **RQ2：** 如何用 per-artifact 生命周期策略，在显存预算内维持更大的安全 `k`？
- **RQ3：** 能否让一张 CUDA Graph 服务多个 in-flight step，而不为每个版本复制完整 Graph？
- **RQ4：** 在显存和 SLO 约束下，自适应 `k` 是否优于静态 `k=1/2/4`？
- **RQ5：** 机制能否消除目标 Race，并在真实推理负载中带来可复现的 Goodput 改善？

### 3.2 主 Claim（候选，尚未验证）

在 CUDA Graph 接管部分 Host 常驻工作之后，剩余 Host–Device overlap depth 受到少数无法入图的动态 artifact 生命周期和前序 Device 依赖共同限制。通过对 `k_res` / `k_dep` 进行可观测归因，并对 Plan / Metadata 使用生命周期感知的多版本管理和 Graph-compatible ABI，有机会在有限显存下提高安全 overlap depth，并降低真实推理负载中的 GPU 等待。

这只是候选论文主张。只有 Gate 0 和后续机制实验成立，才能升级为论文结论。

### 3.3 创新点分层

| 层次 | 内容 | 当前判断 |
|---|---|---|
| L1：正确性 | 对 Plan / Metadata 版本化，避免 WAR / UAF / RAW | 单独看容易被视为双缓冲 / RCU 的工程应用，不足以构成强系统贡献 |
| L2：归因模型 | 测量 `k_actual`，把约束拆成 `k_res` 与 `k_dep` | 有价值的 characterization，但简单上界本身不是创新 |
| L3：机制与控制 | 生命周期感知版本池 + 单图多版本 ABI + 在显存 / SLO 约束下控制 `k` | 最有潜力的系统贡献，仍待实现和验证 |

`S_max = (H + D) / max(H, D)` 只是 Host / Device 两阶段流水的理想上界，用于筛选值得研究的 workload，不是论文贡献。真正要测的是 `k_actual`、等待原因、GPU idle 和残余不可图化 Host 工作。

### 3.4 Artifact 与候选管理策略

| Artifact | 候选策略 | 原因与边界 |
|---|---|---|
| Plan / Metadata Buffer | `k` 版本 Ring Buffer + Device Version Counter | 对象较小，适合多版本；重点研究对象 |
| KV Block / Page Table | Event-deferred free + per-block epoch | 对象较大，不适合整份多版本；留到 Stage 2 |
| Recurrent State Slot | Slot Pool + Epoch，Preemption 时延迟复用 | 留到后续泛化 |
| Graph Workspace | Capture 前预分配 + 版本槽；禁止运行时 resize | Graph 引用地址有生命周期约束 |

不要把所有 Artifact 一律 Double Buffer。静态双缓冲将深度锁定为 2，且对大对象可能显著增加显存。应通过 `k`–显存–TPOT 的 Pareto 曲线验证多版本策略是否值得。

### 3.5 CUDA Graph ABI 设计空间

CUDA Graph 并非“Kernel 参数永远不能变化”。可在两次 Replay 之间通过节点参数更新 API 修改参数，但更新时机、Graph Exec 并发语义和外部 Buffer 生命周期都必须验证。

| 方案 | 机制 | 代价 / 风险 |
|---|---|---|
| 每次 Replay 更新节点参数 | `cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate` | Host 调用与排序开销，可能抵消 overlap |
| `k` 个 Graph Exec Clone | 每个 Exec 固定绑定不同 Buffer | 实现相对简单，但可能增加显存；克隆不自动隔离外部 Buffer |
| 单图 + 静态 Ring + Device Version Counter | Kernel 根据版本索引访问 Buffer | 需要定义版本提交 / 读取顺序，调试和 ABI 设计复杂 |
| Eager | 不走 Graph | 灵活但可能失去 Graph 降低 Launch Overhead 的收益 |

当前设计纪律：

- **第一版可行性验证：** 先测试 Graph Exec Clone + 外部 Buffer 显式重绑定，不能把 Clone 预先认定为最终创新。
- **最强简单对照：** SetParams / 节点参数更新。
- **候选最终机制：** 单图多版本 ABI（Device Version Counter 或版本间接寻址）。
- **关键风险：** 同一个 graphExec 不能与自身并发执行；多个 Exec 的 Graph 私有内存和外部 Buffer 是否隔离必须实测；mixed/ragged 可能使用 Piecewise Graph，需要考虑每段 Graph 的版本一致性和 Shape Bucketing。

### 3.6 控制器（后续阶段）

给定显存预算 `M` 与 TTFT / TPOT SLO，观测 Host / Device 时间、队列深度、Preemption、in-flight 版本数和资源可用状态，选择静态或反馈式 `k`。目标是在不超过显存与 SLO 约束的情况下提高有效重叠深度。

控制器不是 Gate 0 的任务，也不是当前已经实现的组件。

## 4. 相关工作与新颖性边界

### 4.1 工程侧证据（原选题文档的 2026-10-22 快照）

| 工程工作 | 当时记录的内容 | 对本研究的意义 |
|---|---|---|
| vLLM #61239 | Async Scheduling 下 FlashInfer Plan / Workspace 覆盖导致 IMA / Hang；讨论 per-in-flight-step workspace | 真实 WAR 问题切入点 |
| vLLM #59493 | FlashInfer Plan staging buffer 改为 pageable，处理复用竞态 | 点修方案与对照 |
| vLLM #45357 | Async + PD Consumer 下延迟 Block Free | UAF 生命周期案例 |
| vLLM #62749 | 将部分 Uniform Decode Metadata Build 放入 CUDA Graph | 说明 Host 常驻工作正在被图化；图化后残余状态仍值得研究 |
| vLLM #61779 / #61952 | GDN / Hybrid 路径的 Metadata 持久化、Graph 捕获相关工作 | Recurrent State / 固定地址状态的后续案例；首阶段不做 |
| vLLM #49674 | 延迟 Block Free 在 Async KV Consumer 下报告 Zero-progress Preemption Cascade | 点修本身可能引入新的系统问题 |
| FlashInfer #4553 / #4827 | Workspace Resize / Evict 与 Graph 内引用生命周期相关 | Workspace 生命周期的相邻案例 |
| SGLang #35851 | GDN Prefill 的 GPU Idle 报告 | 后续跨引擎动机，不进入 Gate 0 主范围 |

**注意：** 上表是原选题文档截至 2026-10-22 的公开仓库快照，不等于 2026-11-06 的最新 PR 状态。投稿前必须重新核验 Issue / PR 的状态、合并情况和实现细节；公开报告者的性能数字不能当作本项目测量。

### 4.2 学术相邻工作

| 工作 | 领域 / 与本研究的关系 |
|---|---|
| PagedAttention（SOSP 2023） | KV 分页管理，是 Block 生命周期研究背景 |
| RadixAttention（NeurIPS 2024） | Prefix 复用与 KV Block 复用 |
| FlashAttention / FA2（NeurIPS 2022 / ICLR 2024） | 无状态 Attention Kernel 基线 |
| FlashInfer（MLSys 2025） | Plan / Run 分离与动态 Attention 执行 |
| LeanAttention（MLSys 2025） | Decode Attention 分块，问题侧重点不同 |
| vAttention（ASPLOS 2025） | 虚拟内存 KV 管理，相邻但不是 Host–Device overlap depth |
| Marconi（MLSys 2025） | Hybrid Prefix State 复用 |
| NanoFlow（OSDI 2025） | 同卡资源重叠，可参考其系统建模 |
| Event Tensor（MLSys 2026） | 动态图融合，说明单纯融合不是本研究贡献 |
| Execution-State Capsules（2026 预印本） | 图绑定状态快照；不等于稳态重叠深度管理 |
| HYPIC / DASC / vToken | Hybrid State / KV Cache 管理的相邻工作 |
| Memory-Sovereign Inference | 异步复用与资源权威，概念相邻 |
| Retire（2026 预印本） | Request-level versioned execution；本文单位是 step，研究对象不同 |
| Spexis（EMNLP 2026） | Request-level lookahead scheduling；本文是 Host–Device in-flight step 深度，不是请求级未来 Batch 投机 |
| HybridCache（NeurIPS 2026） | Hybrid Prefix Cache，相邻但目标不同 |
| LLM-42 / CoRun | Determinism / Batch Invariance，相邻但不是本文核心问题 |

### 4.3 命名边界

保留工作名 **Safe Lookahead**，但正文统一使用 **overlap depth `k`**：

- Spexis 的 lookahead 在请求 / Batch 调度层；
- Lookahead Decoding 是解码算法；
- 本研究的 `k` 是 Host–Device 执行层的 in-flight step 数量。

如果标题容易与请求级调度混淆，可回退为 **Safe Overlap Depth**。不要用 “versioned execution” 或 “state ownership” 作为主标题，这些概念与相邻工作重叠较大。

### 4.4 审稿人最可能提出的质疑

- **“这不就是 Double Buffer / RCU？”** 单纯版本化不够；必须证明跨 Python、C++ Scheduler、Device Kernel 和 Graph Replay 的生命周期约束如何限制 `k`，并优于最强简单修复。
- **“CUDA Stream 已经保证顺序。”** Stream 只排序 Device Work，不自动排序 Host 对 pinned Buffer 的写入，也不管理 Python / C++ 对象复用。
- **“为什么不直接把 Build 录进 Graph？”** 这正是图化后的问题；必须证明目标 mixed/ragged 路径仍有不可图化的残余 Host 工作。
- **“一个 Graph Exec 能否并发？”** 同一 Exec 自身并发限制必须遵守；要比较多个 Exec 与单图多版本的代价。
- **“效应量是否太小？”** 只对有足够残余 Host 工作、资源限制主导且存在可消除 GPU Gap 的 workload 主张收益。
- **“checker 是否能证明正确？”** 动态 Checker 不是形式化证明；必须配合不变量、有限状态模型和压力测试，不声称 Sound。
- **“单卡是否足够？”** 核心机制是 per-device 的执行和资源生命周期；单卡可做受控验证，多卡不是首阶段必要条件。

## 5. 当前实验与证据状态

### 5.1 Experiment A：CUDA Graph Executable 与外部 Buffer 隔离

设备：NVIDIA H20。受控 CUDA 微基准结果：

- 含图内 allocation / free 节点的 Graph，第二次 Instantiate 返回 `cudaErrorNotSupported`。
- 对图外 Buffer，两个 Executable 默认保留相同的捕获指针。
- 共享指针并发 Replay：200 / 200 次出现数据污染。
- 为第二个 Executable 显式重绑定独立 Buffer：200 次并发测试中 0 次观察到污染。
- Nsight 节点时间线：并发组 20 / 20 对 Kernel 发生重叠；串行组 0 / 20。
- 两个 Executable 的实例化显存增量合计约 2 MiB；外部 Buffer 另行分配。

**能支持的结论：** Graph Executable Clone 不会自动隔离它引用的外部 Buffer；显式重绑定在该受控微基准中可以避免共享 Buffer 污染。

**不能支持的结论：** 不能据此声称 Safe Lookahead 已在 vLLM 中实现，也不能把微基准中的并发时间差写成端到端推理加速。

### 5.2 Experiment B：vLLM Mixed Workload Smoke

一次有效 Smoke 的配置：

- 模型：Qwen3-8B；
- GPU：H20 单卡；
- vLLM：0.29.0；
- FlashInfer：0.6.18；
- Mixed Prompt：8 条请求，每条输出 8 Tokens；
- Async Scheduling：开启。

初步 Decode 指标：

| 指标 | 初步结果 |
|---|---:|
| `D_step` 中位数 | 6.211 ms |
| `H_res` 中位数 | 1.405 ms |
| `H_res / D_step` 中位数 | 22.60% |
| `k_actual` 中位数 / 最大值 | 2 / 2 |
| GPU Step 间空档总计 | 约 3.26 ms |
| GPU Step 间空档占测量跨度 | 约 0.64% |

还观察到 EngineCore 在 `AsyncOutputFuture.result()` 上存在等待，源码路径指向 sampled-token 异步 D2H copy event。

**关键观察：Host 仍有工作或 Future 在等待，不代表 GPU 同时处于空闲状态。** 本次 Smoke 的 GPU Step 间空档比例很低，现有 Async 路径可能已经隐藏了大部分 Host 工作。

这些数据只有一次有效样本，使用同进程采集路径，尚无 Nsight Systems 校准及未插桩的多进程对照。它们只能帮助提出下一轮假设，不能判定 Gate 0 通过或失败，也不能作为性能收益结论。

### 5.3 当前证据等级

| 证据 | 来源 | 可以支持 | 不能支持 |
|---|---|---|---|
| E1 | H20 Graph Clone 微基准与原始记录 | 外部 Buffer 指针共享会污染；显式重绑定在受控测试中有效 | 真实 vLLM 端到端加速 |
| E2 | 一次 vLLM Smoke | 一次负载的 Host / Device 代理指标与初步深度 | 正式的 `k_res` / `k_dep` 归因 |
| E3 | Safe Lookahead 研究协议 | 问题定义、候选机制、门槛和止损条件 | 已实现或已验证的结果 |
| E4 | vLLM Architecture / Async Scheduling 文档 | 公开的异步调度机制 | 锁定版本上的具体运行行为 |
| E5 | NVIDIA CUDA Graph API 文档 | Graph Exec、Launch、节点参数 API 语义 | 具体引擎实现的正确性或性能 |

## 6. Gate 0：首阶段唯一优先级

### 6.1 范围

Gate 0 只做：

- vLLM；
- FlashInfer；
- Plan / Metadata Buffer；
- mixed/ragged 为主要候选负载；
- preemption churn 作为第二负载特征；
- spec decode 只作 `k_dep` 边界案例。

Gate 0 不做：

- KV Block Manager 改造；
- Recurrent State 生命周期改造；
- SGLang 跨引擎适配；
- 完整自适应控制器；
- 多 GPU / 多节点实验。

Preemption 只作为负载特征启用，不修改 KV Block Manager。单卡 PD Consumer 旁路模拟最多时间盒两天，只用于验证动机，不参与 Gate 0 主判定。

### 6.2 版本与代码入口

已有 Smoke 使用 vLLM 0.29.0 + FlashInfer 0.6.18。正式复现公开 #61239 时应锁定与公开复现条件匹配的 vLLM / FlashInfer 版本，并单独记录，不要把不同版本数据混在同一组比较。

关键代码入口：

**vLLM**
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu/attn_metadata_cudagraphs.py`

**FlashInfer**
- `include/flashinfer/attention/scheduler.cuh`
- `decode.py` 中的 `plan()`

**SGLang（后续 Stage 3，不是 Gate 0）**
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
- `runner_backend_utils/breakable_cuda_graph/`

公开 #61239 的复现条件应按对应 Issue / 版本记录核对；原选题待办中记录的候选配置为 4,097-token 单请求、Chunked Prefill、FP8 KV、Async 默认开启。不得把候选参数当作已成功复现。

### 6.3 最小负载矩阵

| 维度 | 建议配置 |
|---|---|
| Workload | mixed/ragged；uniform decode 对照；async off 对照 |
| Batch | B=1 / 8 / 32，按显存与实际路径调整 |
| 场景 | Chunked Prefill + Decode；Prefix Hit；Preemption Churn |
| 边界案例 | Spec Decode on/off，用于观察 `k_dep` |
| 模型 | Qwen3-8B 作为起点 |
| 后端 | FlashInfer 主路径；FlashAttention 只作无状态对照（非 Gate 0 必需） |

Mixed/ragged 的 Plan Artifact 不一定是 FlashInfer Decode Plan，也可能是 vLLM Attention Metadata Builder。必须先确认真实调用路径和对象边界，再决定修改对象。

### 6.4 Gate 0 实验顺序

| 顺序 | 任务 | 交付 |
|---|---|---|
| 1 | 同图 Instantiate 两次，记录 Workspace / 外部 Buffer 地址；并发 Replay 检查污染 | Exec Clone 是否能服务多个 in-flight step |
| 2 | Per-step 记录不可图化 Host 工作及其构成 | 残余 Host 工作基线 |
| 3 | 锁定版本与环境，加入 Host Timeline、in-flight `k`、等待原因和 GPU Idle Instrumentation | 可对齐的“谁在等谁”时间线 |
| 4 | 复现 #61239；测 mixed/ragged、preemption、B=1/8/32 的 `k_actual` | 深度分布和 Race 复现记录 |
| 5 | 将每次等待归因到 `k_res`、`k_dep`、Python/GIL、队列上限或其他原因 | 归因统计 |
| 6 | 比较 async off / 同步路径与 Exec Clone 或 SetParams | Correctness、TPOT、显存、`k` |
| 7 | Nsight 校准、clean multiprocessing 对照、每配置至少三轮交错运行 | Median、IQR、尾延迟和可复现报告 |
| 8 | 对照预注册阈值和 Surprise 假设，作 Gate 0 决策 | GO / PIVOT / STOP |

### 6.5 对照与指标

Gate 0 最小对照：

1. Async Off / 同步基线；
2. `SetParams` 节点参数更新；
3. `k` 个 Graph Exec Clone + 外部 Buffer 显式重绑定。

Stage 1 再补充：

- 当前 Async Baseline；
- 静态 `k=2/4` Ring；
- Pageable Staging；
- Event-deferred Free；
- 最终候选多版本机制；
- 理论 `S_max` 只作为参考上界。

必须报告：

- `k_actual` 分布；
- `k_res` / `k_dep` 归因；
- 每步残余不可图化 Host 工作；
- GPU Idle / Step 间空档；
- Correctness Violation 次数；
- 显存占用；
- TTFT、TPOT P50/P95；
- 既定 TTFT / TPOT SLO 下的 Goodput；
- Throughput 与运行配置。

每个配置至少三轮、交错运行，报告 Median + IQR；尾延迟单独报告。使用 `VLLM_GPU_SYNC_CHECK=error` 排查隐藏同步。记录 commit hash、模型 revision、Warmup、Graph Mode、请求生成方式、显存和 Trace 校验和。

### 6.6 Gate 0 五个数值门槛（预注册，待实测校准）

以下五个条件缺一，不进入 Stage 1：

1. 至少一种目标负载上 `k_actual > 1`；
2. 可归因于 Plan / Metadata / Event 的 GPU 空闲超过 Step 的 5%；
3. 简单修复后仍有约 3–5% 的残余 Gap；
4. 候选机制相对最强简单修复有稳定优势；
5. 每步不可图化的残余 Host 工作至少占 Step Time 的 5%。

此外必须通过归因门槛：

- `k_res` 主导：机制方向成立，进入 Stage 1；
- `k_dep` 主导且可移除：需要重新设计为 Device-side Planning，重新评估范围和工作量；
- `k_dep` 恒为 1：当前 Lifetime-aware 版本池机制不成立，停止性能机制主张，转 Plan B 或重新定义问题。

### 6.7 预注册 Surprise 假设

- **S1：** 真正限制 `k` 的不是 Plan Buffer，而是 Graph Private Memory Pool 或 Block Manager 生命周期。
- **S2：** 简单点修之后的残余 Gap 来自 Python Object Pool、Block Manager、Graph Pool 三个分配器之间的交互，而非单纯 Host 延迟。
- **S3：** Async Scheduling 的 Token Speculation 已使 `k_dep` 大于 1，因此 `k_res` 成为 Binding Constraint，版本池有空间。

原研究协议要求至少观察到一个有解释力的 Surprise，才有充分的 Characterization 价值；如果一个都没有，应重新评估主张，不应事后编造惊喜。

## 7. GPU 与实验资源策略（2026-11-06 更新）

本研究核心是单 GPU 内的 Host–Device overlap、CUDA Graph 和对象生命周期，不需要多卡集群。

**当前优先策略：先延续 H20。** 已有 H20 实验和 Smoke，保持硬件一致有利于比较，减少切换变量。

| GPU | 用途 | 判断 |
|---|---|---|
| H20 96GB | 延续现有环境，正式 Gate 0 与交叉验证 | 优先使用，先保持数据连续性 |
| A100 80GB | 若 AutoDL 价格 / 可用性更好，作为主要实验替代 | 合适；与 H20 数据分开报告 |
| A100 40GB | 最小可行开发与 Qwen3-8B 实验 | 可用，但长上下文 / 高并发空间较小 |
| H100 80GB | 更强性能交叉验证 | 非必需，不要为核心机制强行租用 |
| RTX 4090 24GB | CUDA Graph、Buffer 生命周期和小型微基准 | 适合开发，不建议作为唯一端到端平台 |

资源纪律：

- 不需要多卡；
- 不一次性租满 20 天；
- 先租 4–6 GPU 小时跑通正式 Gate 0 最小矩阵；
- 确认环境、脚本、Nsight 和数据采集链路可靠后，再按实验阶段续租；
- 不同 GPU 的性能数字不能混在同一组比较。

每轮实验记录 GPU 型号、驱动、CUDA、PyTorch、vLLM、FlashInfer、模型 revision、Graph Mode、请求形状、并发度和 GPU 空载状态。

## 8. 论文工作稿与完成度

### 8.1 已有工作稿结构

此前论文 v0.1 已包含：

1. Abstract；
2. Introduction；
3. Problem / Execution Model；
4. Method Proposal；
5. Preliminary Experiments；
6. Evaluation Protocol；
7. Discussion / Limitations；
8. Conclusion；
9. References。

章节蓝图的核心要求：

| 章节 | 当前依据 | 升级为投稿稿前必须补齐 |
|---|---|---|
| 引言 | 重叠深度与生命周期 / 依赖区分 | Related Work 和正式瓶颈证据 |
| 问题模型 | step、`k_actual`、`k_res`、`k_dep`、正确性顺序 | 与真实时间线对齐的操作性定义 |
| 方法 | 归因、版本槽、Graph 执行绑定 | 代码实现、复杂度和不变量论证 |
| 初步实验 | Clone 微基准和一次 Smoke | 不外推性能 |
| 正式评估 | 预注册对照、指标和判定 | 正式重复、Nsight、clean 对照、Goodput |
| 讨论与结论 | 证据边界和止损条件 | Gate 0 决策后重写主张 |

### 8.2 当前完成度（主观估计）

| 维度 | 估计 | 含义 |
|---|---:|---|
| 论文文字框架 | 约 70% | 主要章节已经成形，但 Related Work、方法细节、结果章节仍需完善 |
| 实验支撑的研究成熟度 | 约 25–30% | 有受控微基准和一次 Smoke，但缺正式归因、机制实现和性能验证 |

这不是客观评分或录用概率。当前不是从零写论文，而是已有工作稿，但关键论文主张尚未得到实验支撑。

### 8.3 可直接用于论文的核心摘要（工作稿）

异步调度允许大模型推理引擎在设备执行当前批次时准备后续批次。CUDA Graph 减少了部分主机端开销，但动态图化路径仍涉及随批次变化的元数据、执行计划和工作空间。主机能够领先设备多少个调度步，既受这些对象的生命周期约束，也受前一步设备输出的依赖约束。本文将这一数量定义为安全重叠深度，并提出按资源约束与依赖约束分别归因的测量框架。在此基础上，我们研究面向 Attention Plan 的多版本管理，以及图执行与版本槽之间的绑定方式。当前 H20 实验表明，多个 CUDA Graph Executable 不会自动隔离其引用的外部缓冲区；显式重绑定可在受控微基准中避免共享缓冲区污染。一次 vLLM 负载探测观察到深度为 2、GPU 步间空档较小，但尚未经正式重复、Nsight 时间线和多进程对照验证。这些初步观察提示：扩大版本槽数量之前，必须先确认目标负载中的空档是否由资源生命周期造成，以及额外重叠是否具有端到端价值。本文据此给出可证伪的实验协议，而非预设性能结论。

**写作纪律：** 正式实验完成前，摘要和结论只能报告问题、研究假设、初步观察和实验协议，不得写入未验证的性能提升。

## 9. AI Native Research：科研工作流重构

目标不是简单让 AI 代写论文或代码，而是让 AI 并行承担检索、源码导航、实验自动化、数据分析和反向审查，把人的时间集中到研究判断与证据质量上。

| Agent / 工作流 | AI 负责 | 人必须把关 |
|---|---|---|
| Literature Agent | 论文检索、Related Work、相似工作检查 | 判断研究空白是否真实 |
| Code Agent | vLLM / FlashInfer 调用链、入口、关键对象导航 | 确认真实执行路径 |
| Experiment Agent | 实验脚本、自动运行、配置与数据保存 | 控制变量与实验协议 |
| Profiling Agent | Nsight 时间线整理、等待区间候选归因 | 判断因果关系 |
| Red-Team Agent | 反驳假设、寻找反例、审查 Baseline | 决定是否推翻主张 |
| Writing Agent | 章节组织、图表、英文表达、引用检查 | 确保每个主张都有证据 |

推荐统一 Workspace：

`research/`：研究问题、文献、假设  
`codebase_map/`：入口、调用链、关键数据结构  
`experiments/`：脚本、配置、运行记录  
`results/`：原始数据、Nsight、统计图表  
`claims/`：论文主张与证据映射、反例  
`paper/`：LaTeX、正文、图表和版本

其中 `claims/` 最重要：每个论文主张都要链接对应代码版本、实验配置、原始数据和分析结果。AI 可以解释 Trace、提出假设、生成分析脚本，但不能替代原始 Trace、可复现实验和人工因果判断。

建议采用短反馈闭环：

`真实问题 → AI 并行分析 → 可证伪假设 → 最小实验 → GPU 实际测量 → AI 分析 → Red-Team 反驳 → 人做判断 → 修改机制 / 继续实验`

## 10. 阶段路线、20 天冲刺与止损

### 10.1 Gate 0（当前阶段）

完成第 6 节的复现、时间线、归因、对照和判定。原研究协议按每天约 2 小时规划为 8–10 周；如果投入更高强度，可以压缩日历时间，但不能删掉正确性、重复测量、Nsight 校准和 clean 对照。

### 10.2 Stage 1：最小论文核心

前提：Gate 0 通过，确认 `k_res` 主导且有足够残余性能空间。

范围：

- vLLM + FlashInfer；
- Qwen3-8B；
- 一个主要 Artifact：Plan / Metadata；
- 一个主要负载：mixed/ragged 或经 Gate 0 证明更有价值的 preemption 场景；
- Plan Ring + 单图多版本 ABI；
- 一类 Race 的真实复现；
- SLO Goodput、TTFT / TPOT、GPU Idle、显存与 Correctness；
- `k=2` Plan Ring 的 TLA+ / PlusCal 有限状态模型，检查 Commit / Read / Reuse 顺序与不变量。

第一版 Clone 只作可行性对照；如果单图多版本做不动，必须降级论文主张并如实报告。

### 10.3 Stage 2：机制泛化

- 增加 KV Block 或 Recurrent State；
- 实现自适应 `k` 控制器；
- 增加第二类模型 / 负载；
- 若第一版使用 Clone，继续验证单图多版本 ABI。

### 10.4 Stage 3：跨引擎与工具

- SGLang 或 FlashAttention 路径复现；
- 动态 Checker；
- WAR / UAF / RAW Bug Corpus；
- 跨引擎泛化与工具化。

论文最小可发表单元需要由真实结果决定；完整强版本通常需要 Stage 1 + Stage 2，并有跨引擎检查。Stage 3 是加分项，不应挤占 Gate 0。

### 10.5 20 天论文初稿目标

20 天是当前的高强度交付目标，不等于研究协议的证据标准。可以争取完成完整论文初稿，但必须满足：

- 先通过 Gate 0 的关键判断；
- 核心机制有真实代码与正确性结果；
- 性能数据来自重复且控制变量明确的实验；
- 失败结果和局限如实写入。

此前粗略估计：若 Gate 0 快速通过、目标对象明确且机制实现顺利，集中投入约 10–15 天可能完成有实验支撑的初稿；这是条件估计，不是保证。若 Gate 0 证伪核心假设，必须调整主张，不能为了按期交稿制造收益。

### 10.6 Plan B：Checker / 诊断工具

若残余 Gap 小于 3%、`k_dep` 恒为 1，或不可图化 Host 工作低于 Step Time 的 5%，主性能机制论文可能不成立。可评估独立的低开销动态 Checker：

- 复现 WAR / UAF / RAW；
- 构建 Bug Corpus；
- 归因 Async Engine Hang；
- 报告误报率、开销和覆盖范围。

Checker 是动态辅助，不是形式化证明，也不预设一定能发表。Instrumentation 从 Gate 0 第一周起就应考虑复用。

### 10.7 Venue 选择原则

原选题的初步判断：

- CCF-A 系统会议：ATC、EuroSys、ASPLOS；
- OSDI / SOSP 对团队、系统规模和实验要求更高，不作为当前单人项目的首要假设；
- CCF-A 期刊可评估 TPDS 等；
- MLSys 不属于 CCF-A，投稿选择需按最新目录和 CFP 单独核对。

按实验效果粗略判断：

| 实验结果 | 可能的定位 |
|---|---|
| 残余 Gap 15%+，机制稳定领先最强简单修复 | 可考虑系统会议主轨 |
| Gap 8–15%，或归因结果有反直觉发现 | 需要强机制与充分评估，谨慎选会 |
| Gap 3–8% | 更适合评估期刊或 ML Systems 方向 |
| Gap <3%、`k_dep` 恒为 1 或 Host 工作 <5% | 主性能机制主张不成立，转 Plan B / 重新定义问题 |

以上是研究定位启发式，不是录用概率；正式投稿前必须重新核对会议级别、截稿日期和当年 CFP。

### 10.8 止损条件

1. 目标负载 `k_actual` 普遍只能到 1，或简单修复后残余 Gap 小于 3%：性能故事不成立；
2. Gate 0 五个门槛缺一，或归因不支持 `k_res`：不进入 Stage 1；
3. `k_dep` 主导且不可移除，或恒为 1：版本池机制不成立；
4. 单图多版本 ABI 做不动：退到 Plan Ring + Event-based Lifecycle Management，论文主张相应降级；
5. 上游出现实质相同的 overlap-depth controller / ABI：重新做 Novelty Review；
6. 不可图化 Host 工作低于 Step Time 的 5%：即使 `k` 能提高，也可能没有足够收益空间；
7. 不允许用更大 workload、更多图表或选择性报告掩盖核心假设未成立。

## 11. 下一步执行清单

当前最优先：

- [ ] 锁定正式实验的 GPU、驱动、CUDA、PyTorch、vLLM、FlashInfer 和模型 revision；
- [ ] 复核 vLLM #61239 的当前状态与复现版本；
- [ ] 完成 Graph Exec Clone 地址 / 并发污染最小实验；
- [ ] 建立 per-step Host / Device 时间线和残余 Host 工作基线；
- [ ] 完成 mixed + async on、mixed + async off、uniform + async on 的最小矩阵；
- [ ] 每个配置至少三轮重复；
- [ ] Nsight Systems 校准；
- [ ] clean multiprocessing 对照；
- [ ] 量化 `k_res` / `k_dep` 等等待来源；
- [ ] 对照五个 Gate 0 阈值，作 GO / PIVOT / STOP 决策。

在 Gate 0 通过之前，不开始完整自适应控制器，也不扩展到 KV / Recurrent State / SGLang。

## 12. 项目证据与参考链接

### 12.1 本项目材料

- [GitHub 仓库](https://github.com/WHUTcjh-2024/llm-inference-systems-research)
- [Gate 0 实验说明](../../experiments/gate0/README.md)
- [2026-11-01 无 GPU 交接记录](../../experiments/gate0/NO_GPU_HANDOFF_2026-11-01.md)
- [Week 1 实验报告](../../experiments/gate0/results/week1_report_2026-10-23.md)

### 12.2 工程与 API 参考

- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [NVIDIA CUDA Runtime API — Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)
- [vLLM #61239](https://github.com/vllm-project/vllm/issues/61239)
- [vLLM #59493](https://github.com/vllm-project/vllm/pull/59493)
- [vLLM #45357](https://github.com/vllm-project/vllm/pull/45357)
- [vLLM #62749](https://github.com/vllm-project/vllm/pull/62749)
- [vLLM #49674](https://github.com/vllm-project/vllm/issues/49674)
- [FlashInfer #4553](https://github.com/flashinfer-ai/flashinfer/pull/4553)
- [FlashInfer #4827](https://github.com/flashinfer-ai/flashinfer/pull/4827)
- [SGLang #35851](https://github.com/sgl-project/sglang/issues/35851)

### 12.3 学术相邻工作

- [Execution-State Capsules](https://arxiv.org/abs/2606.20537)
- [HYPIC](https://arxiv.org/abs/2607.01299)
- [DASC](https://arxiv.org/abs/2608.30386)
- [vToken](https://arxiv.org/abs/2608.13263)
- [Memory-Sovereign Inference](https://arxiv.org/abs/2608.23805)
- [Retire](https://arxiv.org/abs/2610.01160)
- [Spexis](https://arxiv.org/abs/2609.34370)
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057)

---

## 文档维护规则

1. 本文是研究定义、设计、证据、实验协议和论文进度的唯一主文档。
2. 新实验先写入 `experiments/` 和 `results/`，审核后再更新本文的“当前实验与证据状态”。
3. 新的研究主张必须附带证据来源、版本、硬件和实验配置。
4. 不把公开 Issue / PR 报告者的数字写成本项目结果。
5. 不删除原始实验数据、失败记录或可审计的 Trace；清理仅限重复文档、空文件和明确无用的临时产物。
6. 每次更新本文时，更新顶部日期，并在 Git commit message 中说明变更。
