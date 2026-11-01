# Safe Lookahead 初稿证据表

| 证据 | 来源 | 可支持的表述 | 边界 |
|---|---|---|---|
| E1 | `experiments/gate0/results/2026-10-26_h20/week1_report.md`，Experiment A 及原始 CSV/Nsight 汇总 | graph executable 克隆实验的污染、重绑定、并发和显存观察 | 仅受控微基准，不是 vLLM 性能 |
| E2 | 同一报告，Experiment B smoke 及原始 CSV | 一次负载的 host/device 代理测量、深度与空档 | 缺重复、Nsight 校准和 clean 多进程对照 |
| E3 | `docs/research/2026-10-22_SafeLookahead_重叠深度_选题深化.md` | 问题定义、候选机制、预注册门槛、止损条件 | 研究计划，不是已完成结果 |
| E4 | https://docs.vllm.ai/en/latest/design/arch_overview/ | vLLM 异步调度的公开机制描述 | 文档可能对应晚于实验的版本；实验行为以锁定版本与实测为准 |
| E5 | https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html | graph executable launch 排序、并发与节点参数 API 语义 | 不证明具体引擎实现的正确性或收益 |

当前缺口：正式负载矩阵、设备时间线校准、强基线、协议实现及正确性验证、端到端 goodput、跨配置复现。缺口补齐前，摘要和结论只报告问题、方法设想及初步观察。
