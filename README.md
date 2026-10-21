# 大模型推理算子与引擎研究

本私有仓库保存当前选题、文献复审和后续经过审核的实验协议与结果。当前首选题是**增量式、异步安全的 Attention Plan**，目标是在主流推理引擎和 GPU 算子的交界处减少动态执行开销。

## 从这里开始

1. [当前选题总览](docs/research/方向总览.md)：三条候选方向、优先级、验证对象和第一轮实验。
2. [跨年文献与机制复审](docs/research/2026-10-21_主流推理引擎与算子_A会选题_跨年复审.md)：2023–2026 年先例、首选题与备选题的完整分析。

## 当前状态

| 项目 | 状态 |
|---|---|
| 首选：增量式、异步安全的 Attention Plan | 研究假设；尚无本项目目标 GPU 的瓶颈或加速实测 |
| 备选：混合有状态模型的图安全 Prefill | 公开问题与研究设计；尚无本项目复现 |
| 储备：混合模型的角色感知缓存与执行 | 早期候选；已有近邻实现，尚未立项 |

第一轮以 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 与 [vLLM](https://github.com/vllm-project/vllm) 为主要代码路径，再评估 [SGLang](https://github.com/sgl-project/sglang)；[FlashAttention](https://github.com/Dao-AILab/flash-attention)是强基线。[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)与 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)的使用边界见选题总览。

## 数据原则

只提交源码、配置、实验协议及审阅后的精简结果；不提交 checkpoint、缓存、原始 trace、完整日志或密钥。论文叙述须区分公开报告、本项目 GPU 正确性与本项目测量的端到端性能。
