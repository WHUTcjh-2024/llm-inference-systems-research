# DashAttention E0/E1 运行包

本目录实现修订方案的本地 E0 正确性 oracle，并准备 E1：用官方 8B checkpoint 确认真实增量解码中的路由成本。**截至本地准备阶段，尚无 H20 实测结果。**

## 本地 E0（无需租 GPU）

```bash
python -m unittest discover -s tests -v
python scripts/e0_oracle.py --seed 0 --trials 100
```

E0 从固定源码逐式翻译三轮 Halley 状态机，并用分组上界执行最大值认证、逐轮暂缓和重新激活；比较每轮状态、最终概率、GQA prior 和 bitmask。代码及证明范围见 [E0 合同](docs/e0_contract.md)。随机例子的界在 CPU 双精度下构造并逐例校验；输出的少算分数数目**不是实际 GPU 加速或真实模型效果**。无 H20 时可以完成 E0，但 E1 真实瓶颈门槛和 E2 真实轨迹回放仍待测。

## 固定版本与存储

- 上游源码：`fasa-org/dash-attention` 的 `5e1c55124cd1645b5234a1af12866b7b5f8bfa2e`。
- 模型：`fasa-org/MiniCPM-4-8B-DashAttention` 的 `d1d53b1f6b6758de657a2ed98798d7165d4d011a`，16,370,657,258 字节的 `pytorch_model.bin`；下载完成后与 Hugging Face LFS SHA-256 核对。
- 依赖：上游 `pyproject.toml` 中的 PyTorch 2.6.0、Triton 3.2.0、Transformers 4.56.0 等精确版本；`verify_source.py` 会拒绝漂移。
- AutoDL 数据盘：默认 `/root/autodl-tmp/dash-router-e1`。源码、虚拟环境、HF 缓存、checkpoint、结果都在这里。脚本确认 `/root/autodl-tmp` 是可写挂载点，且剩余空间足够容纳未下载的权重及 12 GiB 环境余量。**租用实例被释放后，数据盘不能当作永久备份。**

## 一次运行

把本目录上传到 AutoDL 的 `/root/autodl-tmp/dash-router-e1`；也可在配置好私有仓库凭据后，从 `https://github.com/WHUTcjh-2024/dashattention-router-research` 克隆到该路径。若实例支持 AutoDL 的**无卡模式**，先在无卡模式执行准备步骤（官方文档标价 ¥0.1/小时，实际以控制台为准）：

```bash
cd /root/autodl-tmp/dash-router-e1
bash run_e1.sh prepare
```

准备完成后关机；切换到单卡 H20 再执行：

```bash
bash run_e1.sh run
```

若没有无卡模式，也可以在 H20 实例上一次执行全部步骤：

```bash
cd /root/autodl-tmp/dash-router-e1
bash run_e1.sh all
```

`prepare` 只做不需 GPU 的源码、依赖、checkpoint 下载与版本校验；`run` 才检查 H20、运行官方 smoke、缓存 smoke 和 E1。若上传目录本身就是默认持久盘目录，脚本会在该目录下建立 `source/`、`checkpoints/`、`cache/`、`results/`。支持设置 `PERSIST_ROOT=/root/autodl-tmp/<路径>`；不能指向系统盘。

运行顺序：检查数据盘 → checkout 固定源码 → 安装固定依赖 → 下载固定模型快照 → 核对源码、依赖、checkpoint → 检查单卡 H20 → 本项目 `cache_smoke.py` 确认 `use_cache=True` 真正调用 decode → 跑上游 `test_smoke.py`、`test_llama_dash_attn.py` → 跑官方 NIAH 4K 功能 smoke → E1。`cache_smoke.py` 在付费运行阶段执行，当前流程不会在下载大模型前执行它。

下载使用 Hugging Face `snapshot_download` 的 commit SHA 和数据盘 `local_dir`，可断点续传。若远程访问 Hugging Face 较慢，应先在本地或其他网络环境取得同一 revision，再复制完整快照到上述 checkpoint 目录；仍须保留 `e1_download_manifest.json` 并通过 `verify_source.py`。**不要改成浮动的 `main`。**

## E1 输出与口径

默认测 4K、8K、接近 16K 的 batch 1，及接近 16K 的 batch 2；实际 prompt token 数逐项写入结果。每项先 warmup 4 token，再用原版上游 router 做 3 次、每次 32 token 的增量解码。TPOT 取 logits processor 的 CUDA 事件间隔，共 31 个 decode 间隔/次；p50/p95 合并 3 次计算。prefill 不进入 TPOT。

路由成本另做一次 32 token 生成：只包装原版 `decoding.score_blocks` 调用，以 CUDA 事件测量每个 decode token、所有层合计的 `router_call_ms`，另记录 CPU 侧调用与 launch 耗时 `router_host_enqueue_ms`。分段计时再做一次 32 token 生成：在不改源码文件的情况下，包装固定源码的两个 Triton kernel launch 与 `full_attn`，得到 Stage 1、Stage 2、Stage 3 的定位数据。计时结束后用同一输入重跑原版 router 做一致性对照，bitmask 必须相同，prior 必须数值一致。事件围住函数或 kernel 调用，可能包含 GPU 空闲与 launch 间隙；采集真实张量也会改变调度。因此这些单独运行**不能代替原版 TPOT，也不能解释为纯 kernel 执行时间**。`router_call_share_of_tpot_p50_approx` 是不同运行的中位数之比，仅作 E1 停止门槛初筛。

路由密度直接从分段运行中**每个 decode token 的真实 bitmask**逐位计数，分母逐调用取实际 `nchunks+1`，再对层、batch 和 KV 头取平均；不额外生成一条不同轨迹。首个 decode token 的各层 `q_scaled`、`k_chunks`、Stage 1 `scores`、`max_val`、`bmask`、`prior` 保存为 `trace-*.npz`，供后续 A1/A2 离线验证；它们来自官方模型真实 NIAH prompt，不是随机张量。

每次运行单独建 UTC 时间戳目录：

```text
results/<run-id>/
  e1.lock.json
  session.log
  exit-status.txt
  nvidia-smi.txt
  pip-freeze.txt
  official-tests.log
  cache-smoke.log
  official-niah-smoke.log
  run.log
  validation.log
  results.json
  trace-4k-b1.npz
  trace-8k-b1.npz
  trace-16k-b1.npz
  trace-16k-b2.npz
```

准备阶段有 `results/prepare-<run-id>.log`。付费运行一开始就创建结果目录，预检失败也写入 `session.log` 和 `exit-status.txt`。`results.json` 每完成一个 case 原子保存；部分失败也保留前面的结果并以非零码退出。脚本最后执行独立的结果完整性校验，检查 case、长度、计时数量、数值范围与 trace 文件。`run` 开始时重新计算 16.4 GB 权重的 SHA-256，以排除准备后文件变化。所有原始结果和 checkpoint 均被 `.gitignore` 排除。关机/释放实例前，把结果目录另行备份到本地。

## E1 审核门槛

先看 `exit-status.txt`、`validation.log` 是否通过，`results.json` 是否 `complete`，4 项是否齐全，每项 `prompt_tokens` 是否在预期范围，`tpot_ms.count` 是否为 93，路由调用、3 个 stage 与路由密度是否各有 31 个 token，是否生成对应 trace。再看 16K 的 `router_call_share_of_tpot_p50_approx`：若低于 5%，按修订方案需谨慎投入 A1/A2；若接近门槛，必须做同机重复和更精确的端到端对照，不能直接宣称可加速 5%。同时看 `router_host_enqueue_ms`，它和 CUDA 事件时间不是可以直接相加的独立成本。

## 已知边界

- 上游 NIAH 示例指定 `use_cache=False`；它只作为**官方功能 smoke**。E1 专门用 `use_cache=True`，并在付费运行时用 tiny 模型验证实际 decode 调用。
- 首次 CUDA/Triton autotune 与模型加载不计入稳定态 TPOT；不同长度单独 warmup。E1 尚未运行到 H20，环境兼容性及显存占用须由首次 paid run 验证。
- 这阶段只有一种 NIAH 语料和一个 checkpoint。它能判断当前目标模型的路由成本，不能声称跨模型普适性。

## 仓库与数据

本项目保存在私有仓库 `WHUTcjh-2024/dashattention-router-research`，包括脚本、固定版本清单、协议、测试和将来审核后的精简数据。DashAttention 上游按 commit 固定为依赖。不要提交 checkpoint、HF 缓存、原始 `trace-*.npz`、完整运行日志或账号令牌。数据提交约定见 [data/README.md](data/README.md)。
