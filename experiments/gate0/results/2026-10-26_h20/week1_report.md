# Week 1 决策报告（草稿，待正式矩阵）

- 日期：2026-10-26
- 状态：**未完成**。Experiment A 已完成；Experiment B 只有 1 次有效 smoke。
  uniform / mixed / mixed+async off 的正式重复、Nsight 校验和
  multiprocessing 对照尚未运行，因此本报告不给出 GO/PIVOT/RETEST/NO-GO。
- 设备：1 × NVIDIA H20，compute capability 9.0，97,871 MiB
- 驱动 / CUDA：580.65.06 / toolkit 13.0.88
- PyTorch：2.13.0+cu130
- vLLM：0.29.0（PyPI wheel，sha256
  `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`）
- FlashInfer：0.6.18（SM90 sampling 首次使用时 JIT 编译）
- Transformers：5.20.2
- Python / OS：3.12.3 / Ubuntu 22.04.5，kernel 5.15.0-130
- 模型：`Qwen/Qwen3-8B`，HF revision
  `b968826d9c46dd6066d109eabc6255188de91218`，15.26 GiB
- 是否锁频：否
- 实验基线：`345ba8b` + 本次 harness 修改；结果提交 `c7b97d9`

## A. exec 克隆语义

原始数据：`exp_a/`。

| 项目 | 结果 |
|---|---|
| 图含 allocation/free 节点时第二次 instantiate | 返回 `cudaErrorNotSupported`；CUDA 规定含 alloc/free 节点的图同时只能有一个 executable |
| 两个 exec 的 workspace 地址是否相同 | 保留同一捕获指针：200/200 相同；显式重绑定第二 buffer：0/200 相同 |
| 并发 replay 的 race 率 | 共享指针 200/200 出现污染；独立 buffer 并发 0/200；独立 buffer 串行 0/200 |
| e1 / e2 显存增量 | 两个 exec 在 instantiate 阶段共 2.00 MiB（约 1 MiB/exec）；外部 buffer 另由调用方分配 |
| first replay 是否有 lazy allocation | 无。e1 首次 replay 0.00 MiB，e2 首次 replay 0.00 MiB（4 MiB workspace 分辨率下） |
| serial / concurrent 的 wall time | 独立 buffer：串行中位数 36.641 ms，并发 18.322 ms；单 kernel 中位数约 18.31 ms |
| Nsight 节点时间线是否证实重叠 | 是。并发组 20/20 对 kernel 完全重叠；串行组 0/20 |
| 结论：克隆能否服务 `k` 个 in-flight step | 不能自动隔离。图内分配路径被 CUDA 拒绝；图外路径必须为每个 exec 显式重绑定独立 buffer。克隆本身只增加约 1 MiB/exec，但“克隆即可并发”不成立 |

这一组是刻意拉长的单 kernel 微基准，wall time 的 2 倍差异不是 vLLM 端到端收益。

## B. 残余 host 工作

### 已有有效数据：smoke（不是正式 Gate 0 样本）

配置：mixed prompt（64/1024 交替）、8 条请求、每条输出 8 token、
`max_num_seqs=8`、`max_model_len=4096`、`gpu_memory_utilization=0.70`、
chunked prefill 开、prefix cache 关、V2 model runner、FlashInfer、
`async_scheduling=on`、engine 与采集器同进程。

原始数据：`exp_b/smoke/`。按 `scheduled_tokens <= 8` 取 decode step：

| 指标 | 数值 |
|---|---:|
| decode `D_step` median / p95 | 6.211 ms / 6.216 ms |
| decode `H_res` median | 1.405 ms |
| decode `H_res / D_step` median / p95 | 22.60% / 29.40% |
| `k_actual` median / max | 2 / 2 |
| GPU 步间空档 | 总计 3.26 ms，占测量跨度 0.64%；中位数 0.4 µs |
| `AsyncOutputFuture.result()` 等待占 EngineCore 迭代 | 全区间 93.4%；decode 稳态约 79% |
| free KV blocks / waiting | 22,053 / 0 |
| chunked-prefill 首步 | `D_step` 462.9 ms，`H_res` 21.6 ms，`H_res/D_step` 4.7% |

`H_res` 的构成（decode）：schedule ≈0.11 ms、execute（prepare/attn/sample/
graph replay 等）≈0.62–0.64 ms、engine_update ≈0.05 ms，其余为未归类的
Python/引擎胶水；单次 smoke 的水平只作方向性参考。

### 正式矩阵（pending）

| 负载 | async | H_step (µs) | H_res (µs) | D (µs) | H_res/step_time | k_actual (median) |
|---|---|---|---|---|---|---|
| uniform | on | pending | pending | pending | pending | pending |
| mixed | on | pending | pending | pending | pending | pending |
| mixed | off | pending | pending | pending | pending | pending |
| preemption | on | pending（可选） | pending | pending | pending | pending |

正式运行命令已写入 `exp_b_host_work/run_formal.sh`：每个配置 3 次，
随后跑 clean multiprocessing 对照与 Nsight。聚合输出为
`results/formal_aggregate.csv`。

### 无效数据（保留，不参与结论）

`exp_b/invalid_unitbug/` 是第一次 smoke。CUDA event 的 `elapsed_time`
返回毫秒，采集器误按 `ms × 1e6` 存成微秒，导致 `D_step`、`H_res/D_step`
和 `k_actual` 全部错误。修正为 `ms × 1e3` 后才产生 `exp_b/smoke/`。
该目录仅作失败记录。

## Nsight validation

- Experiment A：节点级时间线已完成（`exp_a/clone_external_nodes_summary.txt`）。
  并发组 20/20 对 kernel 重叠，串行组 0/20，与 CUDA event/wall time 一致。
- Experiment B：**尚未采集**。因此当前不能声称 Python 插桩的 `D_step`
  与真实 device 时间线一致，也不能排除关闭 vLLM multiprocessing 改变了
  被研究的重叠现象。正式运行会同时采集 `gate0:measure` 区间并输出
  sync API、graph launch 时 GPU 是否已忙、NVTX 相位和最大 GPU 空档
  附近的主机活动。

## Attribution（初步，非决策级）

| 约束 | 当前证据 | 判断 |
|---|---|---|
| `k_res`（槽位/显存/workspace） | free KV blocks 22,053、waiting 0；未观察到 KV 或 workspace 压力 | **暂无证据表明 k_res 限制 k** |
| `k_dep`（必须等前一步 device 结果） | decode 每步在 `AsyncOutputFuture.result()` 中等待约 5.2 ms；源码中该 future 等待 sampled-token 的异步 D2H 完成 | **有直接证据：k_dep 是主要等待来源** |
| implementation artifact | `max_concurrent_batches`：async on 时 V2 runner 为 `pp_size + 1 = 2`，async off 时为 1；实测 k=2 / 预期 off=1 | **k 的 2 步上限是引擎队列/实现参数，不是显存** |
| instrumentation artifact | 插桩路径关闭了 V1 multiprocessing；clean 对照未跑 | **未量化，不能排除** |

因此目前只能写：**current evidence insufficient for a decision-grade
k_res/k_dep attribution**。smoke 支持“k_dep 主导、k_res 未观察到”，但
样本只有一次、负载只有 8 条请求，不能替代正式矩阵。

源码复核（vLLM 0.29.0 wheel，行号可能随版本变化）：

- `vllm/config/vllm.py:559` `max_concurrent_batches`：async scheduling +
  V2 runner 时返回 `pp_size + 1`，单卡即 2；async off 时为 `pp_size = 1`。
- `vllm/v1/engine/core.py:638` `step_with_batch_queue`：队列未满时先
  schedule/execute 并直接返回；队列满时才 `future.result()` 阻塞。
- `vllm/v1/executor/uniproc_executor.py:32` `AsyncOutputFuture.result`
  调用 `AsyncOutput.get_output()`。
- `vllm/v1/worker/gpu/async_utils.py:167` `AsyncOutput.get_output()` 先
  `copy_event.synchronize()`，等待 sampled-token 的异步 D2H 完成。

这四点是“k 的 2 步上限 + 每步等待采样 token”这一初步归因的来源；它们解释
机制，不替代正式测量。

需要特别记录的反直觉点：decode 的 `H_res/D_step` 约 22.6%，高于 5%
门槛；但 GPU 步间空档只有 0.64%。这说明 k=2 已经基本把 host 工作藏住，
“残余 host 工作占比大”与“增加 k 能带来收益”不是同一件事。正式矩阵要
同时报告 `H_res/D_step` 与 GPU 空档，避免只看前者。

## 决策

- [ ] 进入 Stage 1（`k_res` 主导，`H_res/step_time ≥5%`）
- [ ] 改写 L3 为 device-side planning（`k_dep` 主导但可移除）
- [ ] 转 Plan B（`k_dep` 恒为 1，或 `H_res/step_time <5%` 且 A100 复测仍低）
- [x] **PENDING：证据不足以做预注册决策**

理由：正式 mixed/uniform/async-off 重复、Nsight 校验和 multiprocessing
对照均未运行。当前只有 Experiment A 的完整结论与一次 smoke；用 smoke
触发任何止损或 GO 都不符合仓库的 Gate 0 协议。

## 下一步（最多三条）

1. 在 GPU 模式运行 `run_formal.sh`（3 次重复 + Nsight + clean 对照），
   得到每个 workload 的 `D_step`、`H_res`、`k_actual`、GPU 空档与等待占比。
2. 用 Nsight 时间线校验 `D_step`、host-ahead 与 sync 点，并量化关闭
   multiprocessing 对吞吐/重叠的影响；若冲突，以时间线为准。
3. 在正式数据上完成 `k_res` vs `k_dep` 归因，再按预注册阈值给出
   GO / PIVOT / RETEST / NO-GO。
