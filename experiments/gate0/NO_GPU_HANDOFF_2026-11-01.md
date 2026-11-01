# Gate 0 无卡交接：源码路径、等待归因和上卡流程

日期：2026-11-01。范围：vLLM `v0.29.0` + FlashInfer；这是源码核对与实验准备，**没有新增 GPU 测量**。此前 H20 的 Experiment A 完成、Experiment B 仅一次有效 smoke；决策仍为 PENDING。

## 1. 源码核对：要保护哪个对象

以下行号来自 [vLLM `v0.29.0` 标签](https://github.com/vllm-project/vllm/tree/v0.29.0)，不是主分支。正式实验使用的 PyPI wheel 是 0.29.0；仅凭版本号不能保证 wheel 与标签逐字节一致，最终仍以运行时模块和配置日志为准。

| 路径 | 已核对的分支条件 | 生命周期风险与需要记录的事实 |
|---|---|---|
| [`gpu_model_runner.py:2318`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/worker/gpu_model_runner.py#L2318) | `_build_attention_metadata()` 构建各层 metadata | 要确认目标负载实际调用哪一层 builder、是否在图内或图外，以及主机耗时。不能把整段 `prepare_attn` 都算作不可图化工作。 |
| [`flashinfer.py:1283`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/flashinfer.py#L1283) | `FlashInferMetadataBuilder.build()` 先按批次拆分 decode/prefill，再选择 native FlashInfer 或 TRTLLM 等路径 | “mixed/ragged”是负载形状，不保证进入同一个 plan API。必须在运行时记录 `num_prefills`、`num_decodes` 和所选后端。 |
| [`flashinfer.py:923`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/flashinfer.py#L923) 与 [`:1257`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/flashinfer.py#L1257) | `paged_kv_indices` 是 builder 持有的持久 GPU tensor，native 路径原地重写 | 这是与 FlashInfer plan 的 pinned host workspace 不同的对象；要分别记录写入、H2D、设备读取与复用时间。 |
| [`flashinfer.py:1441`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/flashinfer.py#L1441) | 仅 native prefill、native decode 或 cascade 需要 `paged_kv_indices`；TRTLLM/XQA 使用 GPU block tables | 后端分流可能直接排除版本池的目标对象。不能仅凭配置写着 `FLASHINFER` 就断言走 native plan。 |
| [`flashinfer.py:1603`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/flashinfer.py#L1603) | native prefill 调 `prefill_wrapper.plan()` | mixed batch 中的 prefill 是首要核对路径；需区分 wrapper 自身的 pinned staging buffer 与 builder 的持久 GPU buffers。 |
| [`flashinfer.py:1677`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/backends/flashinfer.py#L1677) | `use_cudagraph` 要求 `pure_decode`，且 decode token 数在图捕获上限内；随后调用 `fast_plan_decode()` | 纯 decode 的快速 plan 路径不能代表 mixed prefill。首阶段设计必须先指定一条实际受限的路径，而非笼统说“plan buffer”。 |

直接相关的上游 [#61239](https://github.com/vllm-project/vllm/issues/61239) 指出，异步 metadata 构建可能在上一步 H2D 消费前重写 FlashInfer 的 pinned plan workspace。它是**公开故障报告与机制线索**，不是本项目已复现的 bug。上游 [#61621](https://github.com/vllm-project/vllm/issues/61621) 还提出同步 `.cpu()` 可能持有驱动上下文锁；因此“恢复同步”有不同成本，不能把所有同步修复视为同一 baseline。当前 vLLM 文档还描述了 mixed batch 的 full CUDA Graph 捕获能力；这不证明 0.29.0 上本负载已完全图化，但足以要求运行时核对图模式，不能预设 mixed 一律走 eager。

## 2. 等待链和可检验归因

```mermaid
sequenceDiagram
    participant S as Scheduler / EngineCore
    participant M as Metadata builder
    participant G as GPU / graph
    participant O as Async output
    S->>M: schedule step N; build metadata
    M->>G: enqueue copies and execute step N
    S->>M: try schedule step N+1
    Note over M,G: 若重用中的 buffer 无槽位，可能出现 k_res 等待或竞态
    Note over S,O: 若下一步需要 sampled token，出现 k_dep 等待
    S->>O: batch queue full; future.result()
    O->>G: wait copy_event completion
    G-->>O: sampled output ready
    O-->>S: update scheduler
```

vLLM 0.29.0 的 [`max_concurrent_batches`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/vllm.py#L559) 在单卡、V2 runner、async on 时返回 `pp_size + 1 = 2`。[`step_with_batch_queue()`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/engine/core.py#L638) 在队列未满时先提交下一步，队列满或无新工作时取队首 `future.result()`。[`AsyncOutput.get_output()`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/worker/gpu/async_utils.py#L167) 等待 sampled output 的 `copy_event`。这条路径解释了 smoke 中的等待及深度 2 的实现上限；**future 等待时间并不自动等于 GPU 空闲时间，更不能直接归因为 `k_dep` 导致的可优化损失**。

| 类别 | 只有看到什么证据才能判定 | 反证或混杂因素 |
|---|---|---|
| `k_res` | 目标对象确无可写槽位；host 或 GPU 因它等待；对应时间线出现可归因空档 | buffer 虽被复用但没有等待，或当前负载走 TRTLLM/XQA；仅有潜在 race 不等于性能空档 |
| `k_dep` | 下一步确实需要前一步的采样、acceptance 或其他设备输出，且等待挡住了下一次可执行工作 | `future.result()` 可与仍在运行的 GPU 工作并存；其时长不是空档 |
| 队列上限 | `batch_queue_size=2` 先于资源/依赖上限触发；提高上限的对照可区分实现限制 | 强行提高上限可能破坏正确性，不能直接作为性能机制 |
| 插桩/进程 | 同一请求集的 clean、in-process、instrumented 输出 hash 与吞吐/TPOT 对照 | 多进程差异、采集开销、首次 JIT、图预热、时钟波动 |

本项目 `analyze_b.py` 的 `D_step` 是 `execute_model` 周围的 CUDA event window；`k_actual` 由该窗口结束时间估算。它们必须与 Nsight 时间线核对后才可作为论文指标。`H_res` 目前是 EngineCore iteration wall time 减 `AsyncOutputFuture.result()` 阻塞时间，不等同于严格测出的“不可图化 host 工作”。

## 3. 上卡时的最小顺序

运行前固定 vLLM wheel、FlashInfer、模型 revision、驱动、图模式、后端选择和 GPU 空载状态，并保留 `[config]` 日志及输出 hash。`run_formal.sh` 现在为每次会话创建独立目录，轮换各配置的起始顺序；Nsight 数据写入 `nsys_runs/` 子目录，不参与正式汇总。主机是 Windows 时，请在远端 Linux 环境运行以下 Bash 命令。

```bash
cd experiments/gate0/exp_b_host_work
MODEL=/path/to/Qwen3-8B PYTHON=/path/to/python \
  REPEATS=3 WITH_NSYS=1 ./run_formal.sh
```

最短决策路径：

1. 先看 `mixed_async-on` 的 patch 状态、`[config]` 中的 `cudagraph_mode`、`[route]` 中的 `prefill`/`decode` 类型与数量，以及输出 hash。`FIPrefill`/`FIDecode` 表示 FlashInfer native 路径，`TRTLLMPrefill` 或 `FlashInferTrtllmAPIDecode` 表示对应 TRTLLM 路径。`decode_wrapper_cuda_graph` 只反映 native decode wrapper 的标志；图实际是否 replay 仍须看时间线。路径不符时先停止，调整负载或研究对象。
2. 查看同会话 `formal_aggregate.csv`、各轮 summary、clean 控制和 `nsys_runs/` 下的 Nsight SQLite。先核对 CUDA event 的 `D_step`、空档及图 launch 时刻，再对等待做归因。`formal_aggregate.csv` 是探索性汇总，不能单独作 GO/NO-GO。
3. 仅在多轮时间线显示**可归因于目标对象生命周期的 GPU 空档**且强简单修复仍留有 gap 时，进入版本池实验；否则按研究协议 PIVOT、RETEST 或 NO-GO。正式阈值及完整矩阵仍以研究协议为准。

## 4. 无卡完成的检查与余项

- 已按 vLLM `v0.29.0` 标签核对上述分支和等待链；尚未核对 wheel 内源码逐字节内容，也未核对 FlashInfer 0.6.18 的内部 plan buffer 实现。
- 已将正式运行脚本改成会话隔离、Nsight 样本隔离和轮次顺序轮换，并添加运行时分流记录；未在 Linux/GPU 环境执行，因此 Nsight 选项、vLLM 运行时分流和性能数值仍待实测。
- `run_formal.sh` 仍是完整正式矩阵，不承诺在单次短租卡时段内全部跑完。若时间紧，先跑 `mixed` on/off + clean 对照 + 一次 Nsight，再决定是否继续正式重复；短集只能用于排错，不能作 Gate 0 结论。
