# Gate 0 最小实验

目标：用 **4–6 GPU 小时**回答两个问题，决定这个选题是否值得继续。

1. **A：exec 克隆语义** — 同一张 CUDA Graph 两次 instantiate，workspace 是否共享、能否并发、显存是否 ×k。
2. **B：残余不可图化 host 工作** — 每步 host 工作里有多少不能入图，占 step time 多少，`k` 被 `k_res` 还是 `k_dep` 卡住。

对应文档：[选题深化](../../docs/research/2026-10-22_SafeLookahead_重叠深度_选题深化.md) 的 3.4.1 与第九节。

## 目录

```
exp_a_clone_semantics/
  clone_semantics.cu    # 同一个 graph instantiate 两次 + 并发 replay
  Makefile
  run_a.sh              # 编译、运行、分析
  analyze_a.py          # 汇总地址 / race / 显存
exp_b_host_work/
  instrument_vllm.py    #  monkey-patch vLLM 的关键路径
  workloads.py          # uniform / mixed / preemption
  bench_b.py            # 跑一个负载，写 CSV
  analyze_b.py          # 算 H_step / H_res / D / k
  run_b.sh
week1_report_template.md
```

## 实验 A：exec 克隆语义

### 需要什么

任意 NVIDIA GPU（H20 / A100 / 4090 都行），CUDA 12.x，`nvcc`。

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

- `addr_same`：两个 exec 报告的 workspace 地址是否相同；
- `race`：并发 replay 时是否有 exec 读到别人的写入；
- 同目录的 `.meta.txt`：e1 / e2 各自的显存增量。

### 判读

| 结果 | 含义 | 下一步 |
|---|---|---|
| 地址相同 + race | 克隆不提供隔离 | 放弃克隆，转 `SetParams` 或单图多版本 |
| 地址不同 + 显存 ×k | 隔离但贵 | 克隆只适合 `k=2`；`k>2` 转单图多版本 |
| 地址不同 + 显存不涨 | 理想 | 克隆可作第一版，最终机制仍用单图多版本 |
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

如果 `bench_b.py` 输出 `rows=0`，说明 engine 仍在独立进程里。先确认
`VLLM_ENABLE_V1_MULTIPROCESSING=0` 生效；如果这个版本不支持，下一步是把
`instrument_vllm.py` 通过 `sitecustomize.py` 注入 worker，或改用一个
in-process 的 `LLMEngine` 驱动脚本。

如果 FlashInfer 在这台机器上不可用，可以传 `--attn-backend ''` 让 vLLM
走默认后端；这时测到的 `H/D` 只用于功能验证，不要和 FlashInfer 的数字混用。

把模型和 conda 环境放在持久盘：

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
```

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

`analyze_b.py` 输出：

- `H_step`：每步 host 时间；
- `H_res`：其中不可图化的部分；
- `D_step`：device 时间；
- `H_res/step_time`：**核心指标**（`H_res / (H_step + D_step)`），≥5% 才有收益空间；
- `H_res/period`：用 scheduler 实际步间隔做分母；如果 CUDA event 测不到 worker stream 的 device 时间，就看这个；
- `H_res/H_step`：host 工作里不可图化的比例，仅作参考；
- `k_actual`：实际 in-flight step 数。

### 判读

- `H_res/step_time ≥ 5%` 且 `k > 1` → 进入 Stage 1；
- `H_res/step_time < 5%` → **不要直接判死刑**：H20 的 prefill 很慢，同样的 host µs 占比会更小；记绝对 µs，等 A100/A800 复测；
- `k ≤ 1` → 先确认 async scheduling 是否真的开着；如果 `k_dep` 主导，转 device-side planning 或 Plan B。

## H20 注意事项

- H20 是 SM90，`ARCH=-arch=sm_90`；
- 算力偏低、带宽高：decode 看起来不错，prefill 的 `D` 很大，mixed/ragged 的 `H/D` 会偏低；
- 因此 H20 适合做**语义验证和功能开发**，效应量结论需要 A100/A800 确认；
- FlashInfer 必须装带 SM90 kernel 的版本，否则会回退到慢路径，`H/D` 失真。

## 已知限制

- `instrument_vllm.py` 依赖 vLLM 内部类名，版本升级会失效；脚本会打印每个 phase 的 patch 状态，没打上的会 fallback；
- 如果 engine 在独立进程里跑，父进程的 patch 看不到 worker 调用；脚本会输出 `rows=0` 警告，先确认 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 是否生效；
- `H_res` 是启发式分类，不是证明；用 nsys 的 NVTX 时间线交叉验证；
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
