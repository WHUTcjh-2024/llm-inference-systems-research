# Gate 0 最小实验

目标：用 **4–6 GPU 小时**回答两个问题，决定这个选题是否值得继续。

当前进度与无卡核对结果见 [2026-11-01 交接](NO_GPU_HANDOFF_2026-11-01.md)。

1. **A：exec 克隆语义** — 分别检查图内分配节点的实例化限制，以及图外 buffer 被多个 exec 引用时的隔离和并发。
2. **B：host 工作采集** — 先记录每步 host 阶段，用 CUDA/NVTX 时间线校准 device 时间、GPU 空闲与真实 in-flight 深度，再分类残余不可图化工作。

对应文档：[选题深化](../../docs/research/2026-10-22_SafeLookahead_重叠深度_选题深化.md) 的 3.4.1 与第九节。

## 目录

```
exp_a_clone_semantics/
  clone_semantics.cu    # 同一个 graph instantiate 两次 + 并发 replay
  clone_external.cu     # 图外 buffer：共享指针与显式重绑定对照
  Makefile
  run_a.sh              # 编译、运行、分析
  analyze_a.py          # 汇总地址 / race / 回放时间
exp_b_host_work/
  instrument_vllm.py    #  monkey-patch vLLM 的关键路径
  workloads.py          # uniform / mixed / preemption
  bench_b.py            # 跑一个负载，写 CSV
  analyze_b.py          # 探索性 host 阶段汇总；不自动判定 Gate 0
  analyze_nsys.py       # 从 Nsight SQLite 提取测量区间内的 GPU 活动与空档
  run_b.sh
week1_report_template.md
```

## 实验 A：exec 克隆语义

### 需要什么

任意 NVIDIA GPU（H20 / A100 / 4090 都行），有对应架构的 CUDA toolkit 和 `nvcc`。

### 运行

```bash
cd exp_a_clone_semantics
ARCH=-arch=sm_90 ./run_a.sh      # H20
ARCH=-arch=sm_80 ./run_a.sh      # A100 / A800
ARCH=-arch=sm_89 ./run_a.sh      # RTX 4090
```

可调参数：

```bash
REPEATS=200 SPIN=50000000 ITERS=1048576 ARCH=-arch=sm_90 ./run_a.sh
```

### 看什么

- 图内 `cudaMallocAsync/cudaFreeAsync`：第二次 instantiate 是否被 CUDA 拒绝；
- 图外 buffer 的 `addr_same`：两个 exec 报告的 workspace 地址是否相同；
- `race`：并发 replay 时是否有 exec 读到别人的写入；
- `separate_serial` 与 `separate_concurrent` 的时间，以及 Nsight 节点时间线是否确认重叠。

### 判读

| 结果 | 含义 | 下一步 |
|---|---|---|
| 图含 allocation/free 节点，第二次 instantiate 被拒绝 | CUDA API 限制：该图同时只能有一个 exec | 图内分配的 exec 克隆路径排除；看图外 buffer 对照 |
| 地址相同 + race | 克隆不提供隔离 | 放弃克隆，转 `SetParams` 或单图多版本 |
| 图外 buffer 显式重绑定后地址不同、无 race | 该微基准具备隔离 | 记录每个 slot 的显存成本，再移入真实 plan 路径 |
| 地址相同 + 无 race | 可能被驱动串行化 | 用 nsys 看两个 graph 的 kernel 是否真的重叠 |

## 实验 B：残余 host 工作

### 环境（与算子开发隔离）

```bash
conda create -n vllm-research python=3.11 -y
conda activate vllm-research
# 按你要测的 vLLM commit 安装；记录 commit 到报告里
pip install vllm
pip install flashinfer-python
```

当前脚本针对 vLLM 0.29.0 的 V2 Model Runner，要求 engine 与采集器同进程。
`bench_b.py` 会检查关键 patch 是否生效，若 `rows=0` 则直接失败。

如果 FlashInfer 在这台机器上不可用，可以传 `--attn-backend ''` 让 vLLM
走默认后端；这时测到的 `H/D` 只用于功能验证，不要和 FlashInfer 的数字混用。

模型与缓存目录需按机器剩余空间选择。实验中的 AutoDL H20 持久盘只剩
约 1.4 GB，因此 Qwen3-8B 放在 `/root/safe-lookahead-runtime/Qwen3-8B`；
该路径随租用实例释放而消失。原始结果应及时复制到本地。

若持久盘有足够空间，可使用：

```bash
export HF_HOME=/data/hf
export CONDA_PKGS_DIRS=/data/conda-pkgs
```

### 运行

```bash
cd exp_b_host_work

# 先跑 uniform decode 基线
WORKLOAD=uniform ./run_b.sh

# 主场景：mixed/ragged + chunked prefill
WORKLOAD=mixed ./run_b.sh

# 第二主场景：preemption（默认会降 gpu_memory_utilization）
WORKLOAD=preemption ./run_b.sh

# 对照：关 async
ASYNC=off WORKLOAD=mixed ./run_b.sh

# 正式矩阵：每次创建独立结果目录，交错重复各配置
MODEL=/path/to/Qwen3-8B PYTHON=/path/to/python \
  REPEATS=3 WITH_NSYS=1 ./run_formal.sh
```

`run_b.sh` 默认完整预热相同负载（`WARMUP_FULL=0` 可用于短 smoke），以减少
冷启动混入测量。正式对照还需固定模型、请求形状、
`--max-num-seqs`、`--max-model-len` 和 GPU 显存利用率。`bench_b.py` 会显式
关闭 prefix cache，输出实际 vLLM 配置、生成 token 数、吞吐和输出摘要。

Nsight Systems 的采集示例：

```bash
nsys profile --output=mixed_async-on_fulltrace \
  --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none \
  ./run_b.sh --nvtx
nsys export --type sqlite --output=mixed_async-on_fulltrace.sqlite \
  mixed_async-on_fulltrace.nsys-rep
python analyze_nsys.py mixed_async-on_fulltrace.sqlite
```

分析器只统计 `gate0:measure` 区间的 GPU kernel / memcpy 时间覆盖，
不把空档直接判为 host 导致的 stall。

一次会话跑完 A + B 的最小集：

```bash
cd experiments/gate0
ARCH=-arch=sm_90 ./run_all.sh
# 需要 preemption 时：
WITH_PREEMPTION=1 ARCH=-arch=sm_90 ./run_all.sh
```

### 前置检查

两个任务共用一张卡时，计时前确认没有别的进程：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

有输出就等。计时实验不要和算子 benchmark 并行。

### 看什么

`analyze_b.py` 只输出探索性指标：各 host 方法耗时、顶层方法耗时之和、
CUDA stream 的 `device_window`、scheduler 调用间隔，以及
`k_host_proxy`（已 schedule 但 `execute_model` 尚未返回的调用数）。
`device_window` 可能包含 host 发射间隙；`k_host_proxy` 不等于 GPU 上真正的
in-flight step 数。**这些输出不能直接填入报告的 `D`、`H_res`、`k_actual`，
也不能自动触发 5% 门槛。**先用 `--nvtx` 和 Nsight Systems 对齐 CPU/GPU
时间线，确认哪些 host 工作确实不可图化、哪些 GPU 空闲由它造成。

### 判读

- 经时间线校准后，若残余 `H_res/step_time ≥ 5%` 且真实 `k>1`，再继续归因和对照；
- 若 H20 上低于 5%，记录绝对 µs，必要时在 A100/A800 复测，不能把当前探索性脚本的输出当作止损证据；
- 若真实 `k≤1`，检查 async 是否开启，再区分 `k_dep` 与 `k_res`。

## H20 注意事项

- H20 是 SM90，`ARCH=-arch=sm_90`；
- 算力偏低、带宽高：decode 看起来不错，prefill 的 `D` 很大，mixed/ragged 的 `H/D` 会偏低；
- 因此 H20 适合做**语义验证和功能开发**，效应量结论需要 A100/A800 确认；
- FlashInfer 必须装带 SM90 kernel 的版本，否则会回退到慢路径，`H/D` 失真。

## 已知限制

- `instrument_vllm.py` 依赖 vLLM 内部类名，版本升级会失效；脚本会打印每个 phase 的 patch 状态，关键 patch 缺失会报错；
- 如果 engine 在独立进程里跑，父进程的 patch 看不到 worker 调用；脚本会在 `rows=0` 时失败，先确认 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 是否生效；
- `H_res`、真实 device 时间和 GPU in-flight 深度尚未由当前脚本单独证明；必须做时间线校准；
- 实验 A 的 `race` 依赖 kernel 里人为拉长的 spin 窗口；如果 `race=0` 但地址相同，先确认两个 launch 真的并发；
- 如果 `cudaMallocAsync` 在 capture 时失败，说明默认 memory pool 不支持 capture；换一个专用 pool，或改用 `cudaGraphAddMemAllocNode` 手工建图。

## 成本

| 范围 | GPU 小时 | 约 |
|---|---|---|
| A（任意卡） | 2–3 | 20–30 元 |
| B 单负载 | 2–3 | 20–30 元 |
| 最小决策合计 | 4–6 | **40–60 元** |

## 产出

跑完后填 [`week1_report_template.md`](week1_report_template.md)，只留四件事：A 结论、B 的表、归因、go/no-go。
