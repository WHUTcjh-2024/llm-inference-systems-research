# Gate 0 首周实验记录（进行中）

- 日期：2026-10-23（北京时间）
- 设备：AutoDL H20，compute capability 9.0，显存 97,871 MiB
- 驱动 / CUDA：595.71.05 / toolkit 13.0.88
- 实验前仓库基线：`456ada27801d0a29401bea8324e982e082536734`
GPU 隔离：实验前 `nvidia-smi` 无 compute app

## A. Graph exec 克隆语义

### 图内分配节点

同一张 graph 捕获 `cudaMallocAsync → kernel → cudaFreeAsync` 后，第一次
`cudaGraphInstantiate` 成功，第二次返回 `cudaErrorNotSupported`。CUDA Runtime
API [明确规定](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)，含 allocation/free 节点的 graph 同时只能存在一个 executable
实例。因此这条具体的 exec 克隆路径在实例化阶段就被排除，无法继续测其
workspace 地址或并发污染。

原始诊断已保存在本地忽略目录
`exp_a_clone_semantics/results/exp_a_20261023_000019.meta.txt`。

### 图外 buffer 对照

对不含 allocation/free 节点的 graph，两次实例化均成功。使用 4 MiB
外部 workspace，`spin=5,000,000`，每组重复 100 次：

| 对照 | 地址相同 | 出现污染 | 回放 wall time 中位数 |
|---|---:|---:|---:|
| 两个 exec 保留同一捕获指针，并发 | 100/100 | 100/100 | 18.326 ms |
| 第二个 exec 显式绑定独立 buffer，串行 | 0/100 | 0/100 | 36.639 ms |
| 第二个 exec 显式绑定独立 buffer，并发 | 0/100 | 0/100 | 18.326 ms |

Nsight Systems 的节点级时间线在 2 次复查中确认：独立 buffer 并发组的两条
stream 上，约 18.31 ms 的 kernel 执行区间重叠；串行组无重叠。这里的
约 2 倍 wall time 差异只属于刻意拉长的单 kernel 微基准，不是 vLLM
端到端收益。

原始 CSV：`exp_a_clone_semantics/results/exp_a_external_20261023_000019.csv`

Nsight 证据：`exp_a_clone_semantics/results/clone_external_nodes.nsys-rep`
及对应 SQLite 导出，均只保存在本地忽略目录。

**A 结论**：exec 克隆不会自动隔离捕获的图外 workspace。图内分配路径受
CUDA API 限制；图外路径必须为每个 in-flight 实例提供独立 buffer，并显式
重绑定 kernel 参数。真实 FlashInfer plan/workspace 的内存成本和性能仍待测。

## B. vLLM + FlashInfer 的 host 工作与 GPU 时间线

运行环境：vLLM 0.29.0 wheel、PyTorch 2.13.0、FlashInfer 0.6.18、
Transformers 5.20.2；Qwen3-8B 从 ModelScope 下载到实例临时根盘。
模型下载未固定 revision，下一轮严格复现需补快照版本。实验配置明确验证为
V2 Model Runner、`FLASHINFER` backend、`uni` executor、chunked prefill 开启、
prefix cache 关闭。GPU 未锁频。

负载：32 条请求，短 prompt 64 token 与长 prompt 1024 token 交替；每条
固定输出 64 token，`max_num_seqs=8`、`max_model_len=4096`、
`gpu_memory_utilization=0.70`，相同负载先完整预热一次。采集阶段仍开启了
Python phase instrumentation 与 CUDA events，因此吞吐是**插桩条件下**的
探索性数值；不等于纯服务性能。

| 配置 | 4 次未开 Nsight 的输出吞吐 (token/s) | 中位数 |
|---|---|---:|
| async on | 587.745、549.666、585.062、584.762 | 584.912 |
| async off | 532.866、533.925、528.892、555.180 | 533.396 |

两组均输出 32 条请求、2048 个生成 token。最后一组 on/off 的 token 序列
SHA-256 均为 `01d1d59234da84d66f93e1608cd48f9a51aa41b092f9f823e0f6a32425de2656`。
中位数相差约 9.7%，但 on 组单次从 549.7 到 587.7 token/s，且未锁频；
这只能说明该配置下观察到异步组较快，**不能当成稳定加速或机制贡献**。

首次完整预热的探索性 phase 中位数：

| phase | async on | async off |
|---|---:|---:|
| schedule | 129.4 µs | 126.7 µs |
| execute_model（CPU 方法 wall time） | 623.5 µs | 639.4 µs |
| prepare_inputs | 262.1 µs | 255.8 µs |
| prepare_attn | 66.1 µs | 65.4 µs |
| FlashInfer metadata build | 11.7 µs | 12.0 µs |
| sample | 308.7 µs | 308.7 µs |

`execute_model` 包含其中的 prepare / attention / sample；这些行不可相加。
短 smoke 曾出现约 430 ms 的冷编译峰值，因此未拿它估计稳态 host 工作。

Nsight Systems 完整 CUDA/NVTX trace 从 `gate0:measure` 标记截取测量区间，
把 kernel 与 memcpy 的时间区间合并后得到：

| trace | 测量区间 | GPU 活动覆盖 | 内部空档总量 | ≥100 µs 空档 |
|---|---:|---:|---:|---:|
| async on | 3654.8 ms | 93.0% | 254.0 ms | 36 个，共 114.4 ms |
| async off | 3808.1 ms | 89.7% | 391.9 ms | 552 个，共 216.7 ms |

异步组有两段 57.2/44.2 ms 的长空档，都在测量尾部；同步组则分布着
较多 0.1–4 ms 的空档。上述 GPU 活动覆盖是 **Nsight 插桩 trace 的时间覆盖**，
不是 GPU SM 利用率，也不能单独证明空档由 host、plan、依赖或资源导致。

原始材料仅保存在本地 `exp_b_host_work/results/`：各次 `exp_b_mixed_async-*`
CSV、`repeat_*`/`digest_*` 日志、`mixed_async-{on,off}_fulltrace.nsys-rep`
及 SQLite。分析命令：
`python exp_b_host_work/analyze_nsys.py exp_b_host_work/results/mixed_async-on_fulltrace.sqlite`。

**B 当前判断**：vLLM + FlashInfer 的真实混合负载链路已跑通，完成了同形状
预热、async 对照、输出一致性和 GPU 时间线。`device_window` 包含可能的 host
发射间隙，`k_host_proxy=1` 只表示 schedule 到 execute 方法返回的 proxy，
两者不能填 `D` 或 `k_actual`。`H_res`、真实 in-flight step 深度、
`k_res/k_dep` 与 GPU 空档的因果归因尚未量到，**Gate 0 不能判定通过**。

## 下一步

1. 记录 graph launch 与对应 device completion 的 step 关联，测真实 `k_actual`；
   同时标出等待的资源或采样依赖，再拆 `k_res/k_dep`。
2. 对 ≥100 µs 的 GPU 空档逐段对齐 CPU NVTX 与 CUDA API，区分 host
   剩余工作、尾部 drain 与依赖等待，得到可解释的 `H_res`。
3. 若前两项支持继续，再测真实 FlashInfer plan buffer 的版本池成本与
   单图多版本路径；若不支持，按 Gate 0 止损条件调整题目。
