# 大模型推理算子与引擎研究

本私有仓库保存当前选题、文献复审和后续经过审核的实验协议与结果。当前锚定方向是**可变执行状态的所有权与安全重叠**：把 host–device 重叠深度当作可调度资源，在 vLLM/SGLang/FlashInfer 上研究 attention plan、KV block、recurrent state 与 CUDA Graph workspace 的生命周期协议、单图多版本算子 ABI 和自适应 lookahead。

## 从这里开始

- [选题深化：所有权与安全重叠](docs/research/2026-10-22_所有权与安全重叠_选题深化.md)：锚定方向、上界模型、核心创新、单卡实验与止损。

2026-10-21 的两份三方向文档已在收敛后移除；仍然有效的文献先例压缩在新文档第五节。

## 当前状态

| 项目 | 状态 |
|---|---|
| 锚定：可变执行状态的所有权与安全重叠 | 研究假设；尚无本项目目标 GPU 的 H/D、lookahead 或加速实测 |
| 案例 1：增量式 Attention Plan | 收敛为 plan buffer 所有权；不再独立立项 |
| 案例 2：混合模型图安全 Prefill | 收敛为 recurrent state slot；GDN 图捕获已被上游 PR 抢跑 |
| 案例 3：角色感知缓存与执行 | 收敛为 KV block 回收；不再独立立项 |

第一轮以 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 与 [vLLM](https://github.com/vllm-project/vllm) 为主要代码路径，再评估 [SGLang](https://github.com/sgl-project/sglang)；[FlashAttention](https://github.com/Dao-AILab/flash-attention)是强基线。[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)与 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)的使用边界见选题总览。

## 数据原则

只提交源码、配置、实验协议及审阅后的精简结果；不提交 checkpoint、缓存、原始 trace、完整日志或密钥。论文叙述须区分公开报告、本项目 GPU 正确性与本项目测量的端到端性能。
