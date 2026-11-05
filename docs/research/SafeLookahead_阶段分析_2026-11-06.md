# Safe Lookahead 项目阶段分析

> 整理日期：2026-11-06  
> 项目仓库：[WHUTcjh-2024/llm-inference-systems-research](https://github.com/WHUTcjh-2024/llm-inference-systems-research)  
> 研究主线：Safe Lookahead——动态图化推理中的安全重叠深度

## 1. Idea：研究什么问题？

Safe Lookahead 研究大模型推理引擎的异步调度：当 GPU 执行当前 step 时，Host 能否安全地提前准备更多后续 step？限制这种提前量的因素究竟是什么？

当前研究将安全重叠深度的限制拆成两类：

- **k_res（资源约束）**：Attention plan、metadata、workspace 等可变对象仍被旧 step 使用，槽位不能安全复用。
- **k_dep（依赖约束）**：后续 step 必须等待前一步的 sampled token 等设备结果，因而无法继续提前调度。

候选机制是在确实存在 k_res 瓶颈时，为关键 plan/metadata 建立有限的多版本缓冲区，并通过 CUDA Event / 完成事件管理版本槽位的写入、读取和回收。

**核心前提：增加版本槽只能缓解资源生命周期限制，不能消除真正的设备结果依赖。因此必须先测量和归因，再决定是否实现多版本机制。**

## 2. 当前候选创新点

### 2.1 安全重叠深度的归因模型

不再只讨论是否开启 async scheduling，而是测量实际 in-flight step 数量，并区分资源生命周期限制 k_res 与执行依赖限制 k_dep。这是当前最重要的问题定义和测量主线。

### 2.2 Plan / Metadata 多版本生命周期管理

若正式实验确认 k_res 是主要瓶颈，则为不同执行步骤保留独立版本：

1. Host 只写入当前可用的版本槽；
2. 每个执行 step 绑定自己的版本；
3. Device 消费完成后，相关 Event 才允许槽位回收；
4. 旧消费者未完成前，禁止覆盖或复用槽位。

需要验证两个核心不变量：

- Device 读取的版本必须与该 step 提交的版本一致；
- 所有旧消费者完成前，槽位不能被重新写入或回收。

### 2.3 CUDA Graph 与版本槽的绑定

已讨论过多个 executable 显式重绑定外部 buffer、节点参数更新、静态版本池、Single Graph Multi-Version ABI 等候选路径。

这些仍是待比较的实现方案，不应提前写成已经完成的创新。最终机制必须由真实 vLLM / FlashInfer 路径中的对象生命周期和实验结果决定。

## 3. 已有实验与证据

### 3.1 Experiment A：CUDA Graph executable 与外部 buffer 隔离

设备：NVIDIA H20。

已有结果：

- 含图内 allocation/free 节点的 Graph，第二次 instantiate 返回 cudaErrorNotSupported。
- 对图外 buffer，两个 executable 默认保留相同的捕获指针。
- 共享指针并发回放：200/200 次出现数据污染。
- 显式为第二个 executable 重绑定独立 buffer：200 次并发测试中 0 次观察到污染。
- Nsight 节点时间线确认并发组 20/20 对 kernel 重叠，串行组 0/20。
- 两个 executable 的实例化显存增量合计约 2 MiB；外部 buffer 另行分配。

**能支持：** 克隆 Graph executable 本身不会自动隔离其引用的外部 buffer；显式重绑定在该受控微基准中可以避免共享 buffer 污染。

**不能支持：** 不能据此声称 Safe Lookahead 已在 vLLM 中实现，也不能把微基准中的并发时间差写成端到端推理加速。

### 3.2 Experiment B：vLLM mixed workload smoke

一次有效 smoke 的配置：

- 模型：Qwen3-8B；
- vLLM：0.29.0；
- FlashInfer：0.6.18；
- H20 单卡；
- mixed prompt，8 条请求，每条输出 8 tokens；
- async scheduling 开启。

初步 decode 指标：

| 指标 | 结果 |
|---|---:|
| D_step 中位数 | 6.211 ms |
| H_res 中位数 | 1.405 ms |
| H_res / D_step 中位数 | 22.60% |
| k_actual 中位数 / 最大值 | 2 / 2 |
| GPU 步间空档总计 | 约 3.26 ms |
| GPU 步间空档占测量跨度 | 约 0.64% |

Smoke 还观察到 EngineCore 在 AsyncOutputFuture.result() 上存在明显等待，源码路径指向 sampled-token 异步 D2H copy event。

**关键观察：Host 仍有工作、Future 存在等待，并不等于 GPU 同时处于空闲状态。** 当前 smoke 中 GPU 步间空档比例很低，说明现有 async 路径可能已经隐藏了大部分 Host 工作。

因此，现有证据暂时没有证明增加 k 能带来端到端收益。k_dep、队列上限和实现细节可能比 k_res 更重要，但目前还不能作正式归因。

### 3.3 尚缺的关键证据

- uniform、mixed/ragged、async off 的正式重复实验；
- Experiment B 的 Nsight Systems 时间线校准；
- clean multiprocessing 对照，以估计插桩和同进程运行的扰动；
- 对实际运行路径、CUDA Graph 模式和 FlashInfer backend 分流的确认；
- k_res 与 k_dep 的决策级定量归因；
- 多版本机制的真实实现与正确性测试；
- 与强基线对照的端到端 goodput、TTFT、TPOT、P95 和显存开销；
- Ablation 与跨配置复现。

**Gate 0 当前仍为 PENDING，不能提前判定 GO、PIVOT 或 NO-GO。**

## 4. 当前论文初稿进度

仓库已经存在：

- docs/research/SafeLookahead_论文初稿_v0.1.md
- docs/research/SafeLookahead_初稿章节蓝图_v0.1.md
- docs/research/SafeLookahead_初稿证据表_v0.1.md

论文 v0.1 已经包含摘要、引言、问题模型、方法设想、初步实验、评估协议、讨论、结论和参考文献。

以下比例是根据当前仓库材料作出的主观阶段估计，不是客观评分：

| 维度 | 估计 | 说明 |
|---|---:|---|
| 论文文字框架 | 约 70% | 主要章节已经成形，但 Related Work、方法细节和结果章节仍需完善 |
| 实验支撑的研究成熟度 | 约 25–30% | 有受控微基准和一次 smoke，但缺正式归因、机制实现和性能验证 |

因此，当前不是从零开始写论文，而是**已经有工作稿，但关键论文主张还没有被实验支撑**。

## 5. GPU 与实验资源

当前已有 H20 实验基础。对 Gate 0 和首个机制原型：

- **优先延续 H20**：已有环境和初步结果，减少换卡引入的变量；H20 显存足以运行当前 Qwen3-8B 实验。
- **A100 80GB 是可用替代**：若 AutoDL 价格更合适可以使用，但应作为独立硬件配置记录，不能与 H20 数据直接混合比较。
- **不需要多卡，也不需要强行租 H100**：当前问题主要是单 GPU 内的 Host–Device overlap、CUDA Graph 和对象生命周期。
- **先短租验证**：先安排约 4–6 GPU 小时完成正式 Gate 0 最小矩阵，确认环境、脚本和采集链路可靠后，再决定是否继续租卡。

正式实验应锁定并记录 GPU 型号、驱动、CUDA、PyTorch、vLLM、FlashInfer、模型 revision、Graph 模式、请求形状、并发度和 GPU 空载状态。

## 6. AI Native Research：科研工作方式重构

目标不是简单让 AI 代写论文或代码，而是让 AI 并行承担检索、源码导航、实验自动化、数据分析和反向审查，把人的时间集中到研究判断和证据质量上。

| 工作流 / Agent | 主要任务 | 人需要把关的部分 |
|---|---|---|
| Literature Agent | 检索论文、整理 Related Work、检查相似工作 | 判断研究空白是否真实 |
| Code Agent | 梳理 vLLM / FlashInfer 调用链和关键对象 | 确认真实执行路径 |
| Experiment Agent | 编写实验脚本、自动运行、保存配置和原始数据 | 设计控制变量和实验协议 |
| Profiling Agent | 整理 Nsight 时间线、定位等待区间 | 判断性能归因是否成立 |
| Red-Team Agent | 反驳假设、寻找反例、审查 baseline | 判断反例是否推翻主张 |
| Writing Agent | 组织章节、图表、英文表达和引用 | 确保每个主张都有证据 |

建议统一 Research Workspace：

- research/：研究问题、论文检索、假设；
- codebase_map/：入口、调用链、关键数据结构；
- experiments/：脚本、配置、运行记录；
- results/：原始数据、Nsight、统计图表；
- claims/：论文主张与证据映射、反例；
- paper/：LaTeX、正文、图表和版本。

其中 claims/ 最重要：每个论文主张都要链接到对应代码版本、实验配置、原始数据和反例。AI 可以帮助解释数据，但不能替代原始 Trace、可复现实验和人工因果判断。

## 7. 下一阶段行动顺序

### 第一阶段：完成 Gate 0

优先完成：

1. mixed + async on；
2. mixed + async off；
3. uniform + async on；
4. 每个配置至少三轮重复；
5. Nsight Systems 校准；
6. clean multiprocessing 对照；
7. 核对真实 attention backend、Graph 模式和运行时分流。

先确认目标对象是否真的造成 GPU 空档，再做机制实现。

### 第二阶段：根据 Gate 0 决策

- **若 k_res 主导：** 实现最小 Multi-Version Buffer + Event-based Lifecycle Management，先做 correctness，再做性能。
- **若 k_dep 主导：** 不要继续堆版本槽；重新评估依赖链是否存在可移除的等待，或转向诊断/测量贡献。
- **若 GPU 空档已经很小：** 重新评估性能机制论文的价值，不要为了保住原 Idea 而制造收益结论。

### 第三阶段：机制与性能评估

至少比较当前 async baseline、async off / 同步对照、简单双缓冲和候选多版本机制。

报告正确性、端到端 goodput、TTFT、TPOT、P95、GPU 空档和显存开销。所有性能结论都必须来自重复且控制变量明确的实验。

### 第四阶段：论文整合

将已验证结果写入论文，补齐 Related Work、方法实现细节、正确性论证、性能评估、Ablation、局限性和复现说明。

## 8. 时间预估与主要风险

如果 Gate 0 通过、目标对象明确、原型实现顺利，完成一份有实验支撑的完整初稿，粗略估计还需要约 **10–15 天集中工作**。这不是保证；若 Gate 0 发现 k_res 并非真实瓶颈，研究主张可能需要调整。

当前主要风险：

1. 现有异步路径可能已经把 GPU 空档压得很低；
2. smoke 中的等待不一定对应可优化的 GPU idle；
3. mixed workload 未必实际走预想的 FlashInfer native plan 路径；
4. Graph executable 克隆微基准不能代表真实 serving 收益；
5. 多版本机制可能增加显存和管理开销，却没有足够端到端收益。

## 9. 当前阶段结论

Safe Lookahead 已形成相对清晰的问题定义、候选机制、实验框架和论文工作稿。现有 H20 实验提供了有价值的 CUDA Graph buffer 隔离证据，但还没有证明资源生命周期是 vLLM 真实负载中的主要性能瓶颈。

**下一步不是继续扩写 Idea，而是完成 Gate 0：用正式负载、Nsight 时间线和 clean 对照回答 k_res 是否真的限制了重叠深度。**

只有这个问题得到肯定且可复现的回答，Multi-Version Buffer 才值得成为论文的核心性能机制。否则，应根据证据调整研究主张，而不是预设结论。

## 项目材料

- [GitHub 仓库](https://github.com/WHUTcjh-2024/llm-inference-systems-research)
- [论文初稿 v0.1](https://github.com/WHUTcjh-2024/llm-inference-systems-research/blob/main/docs/research/SafeLookahead_%E8%AE%BA%E6%96%87%E5%88%9D%E7%A8%BF_v0.1.md)
- [初稿章节蓝图](https://github.com/WHUTcjh-2024/llm-inference-systems-research/blob/main/docs/research/SafeLookahead_%E5%88%9D%E7%A8%BF%E7%AB%A0%E8%8A%82%E8%93%9D%E5%9B%BE_v0.1.md)
- [初稿证据表](https://github.com/WHUTcjh-2024/llm-inference-systems-research/blob/main/docs/research/SafeLookahead_%E5%88%9D%E7%A8%BF%E8%AF%81%E6%8D%AE%E8%A1%A8_v0.1.md)
- [Gate 0 实验说明](https://github.com/WHUTcjh-2024/llm-inference-systems-research/blob/main/experiments/gate0/README.md)
- [2026-11-01 无 GPU 交接记录](https://github.com/WHUTcjh-2024/llm-inference-systems-research/blob/main/experiments/gate0/NO_GPU_HANDOFF_2026-11-01.md)
