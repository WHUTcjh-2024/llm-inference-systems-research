# 大模型推理算子与引擎研究

本私有仓库保存当前选题、文献复审和后续经过审核的实验协议与结果。当前锚定方向是 **Safe Lookahead：把 host 重叠深度当作可调度资源**。核心问题是 host 最多能安全地领先 GPU 多少步、受什么约束、如何用有限显存把 `k` 顶到最大；版本/所有权协议只是实现手段。代码落在 vLLM、SGLang、FlashInfer、FlashAttention。

当前阶段是 **Gate 0**：只做 vLLM + FlashInfer 的 plan buffer，先证明目标负载上 `k>1`，且简单修复后仍有残余 gap；不写控制器，不做 KV、recurrent state 或 SGLang。

## 从这里开始

- [选题深化：Safe Lookahead](docs/research/2026-10-22_SafeLookahead_重叠深度_选题深化.md)：Gate 0 测量设计、CUDA Graph 设计空间、核心创新、单卡实验与止损。

2026-10-21 的两份三方向文档已在收敛后移除；仍然有效的文献先例压缩在新文档第七节。

## 当前状态

| 项目 | 状态 |
|---|---|
| 锚定：Safe Lookahead（重叠深度） | Gate 0：测量 `k_actual`、等待原因与残余 gap；尚无本项目 GPU 数据 |
| 案例 1：增量式 Attention Plan | 收敛为 plan buffer 生命周期案例；不再独立立项 |
| 案例 2：混合模型图安全 Prefill | 收敛为 recurrent state slot 的图内版本案例；#61779/#61952 是公开在途 PR（未合入），#62749 为 draft，存在撞车风险 |
| 案例 3：角色感知缓存与执行 | 收敛为 KV block 回收案例；不再独立立项 |

第一轮以 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 与 [vLLM](https://github.com/vllm-project/vllm) 为主要代码路径，Stage 3 再评估 [SGLang](https://github.com/sgl-project/sglang)；[FlashAttention](https://github.com/Dao-AILab/flash-attention) 是无状态强基线。DeepGEMM 与 DeepInfer 只作设计参考，不作为单卡主验证对象。

## 数据原则

只提交源码、配置、实验协议及审阅后的精简结果；不提交 checkpoint、缓存、原始 trace、完整日志或密钥。论文叙述须区分公开报告、本项目 GPU 正确性与本项目测量的端到端性能。
