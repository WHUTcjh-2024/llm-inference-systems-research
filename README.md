# 大模型推理算子与引擎研究

本仓库围绕 AI Infrastructure / LLM Inference Systems 的真实系统问题开展研究。当前唯一锚定方向是 **Safe Lookahead：图化推理中的安全 Host–Device Overlap Depth**。

## 唯一研究主文档

- [Safe Lookahead Research Master（2026-11-06）](docs/research/SafeLookahead_Research_Master_2026-11-06.md)

该文档统一维护 Idea、问题模型、候选创新、相关工作、已有实验与证据边界、Gate 0 协议、GPU 策略、论文进度、AI Native Research 工作流和后续路线。后续研究设计与阶段状态以该文档为准，不再维护重复的选题深化、阶段总结、论文蓝图和证据表副本。

## 当前阶段

**Gate 0 / Research Validation：尚未通过。**

目前已有 H20 上的 CUDA Graph executable / 外部 buffer 隔离微基准，以及一次 vLLM mixed workload smoke。正式重复实验、Nsight Systems 校准、clean multiprocessing 对照和 `k_res` / `k_dep` 定量归因仍未完成；当前没有可声称的端到端性能收益。

下一步优先完成：

1. 锁定 GPU、驱动、CUDA、PyTorch、vLLM、FlashInfer 和模型版本；
2. 完成 Graph Exec Clone 地址 / 并发污染实验；
3. 建立 per-step Host / Device 时间线与残余 Host 工作基线；
4. 完成 mixed + async on、mixed + async off、uniform + async on 的重复实验；
5. 使用 Nsight Systems 校准，并完成 clean multiprocessing 对照；
6. 按预注册阈值决定 GO / PIVOT / STOP。

## 实验入口

- [Gate 0 实验说明](experiments/gate0/README.md)
- [2026-11-01 无 GPU 交接记录](experiments/gate0/NO_GPU_HANDOFF_2026-11-01.md)

## 代码路径

第一阶段以 [vLLM](https://github.com/vllm-project/vllm) + [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 为主。SGLang 和其他后端属于后续泛化，不纳入 Gate 0。

## 数据与证据原则

- 区分本项目实测、公开 Issue / PR 报告、研究假设和未来计划。
- 所有性能结论必须对应代码版本、GPU、模型、负载配置、原始数据和可复现实验。
- 不把单次 smoke 或 CUDA 微基准外推为端到端推理收益。
- 不提交模型 checkpoint、缓存、密钥和不必要的大型二进制 Trace；原始实验数据与失败记录按实验目录规则保留。
