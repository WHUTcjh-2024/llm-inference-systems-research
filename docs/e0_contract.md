# E0：固定三轮状态机的 CPU 合同

上游参照：`fasa-org/dash-attention` commit `5e1c55124cd1645b5234a1af12866b7b5f8bfa2e`，文件 `dash_attn/decoding/stage2_block_selection.py`，锁定 SHA-256 `2b97c19728e71caf77d1772cd1f928174b606b46f4357c28c18fe096df6b80e5`。本地 oracle 使用 NumPy 双精度解释相同的控制流；它不执行 Triton，也不声称与 GPU 逐位一致。

## 输入和状态

- `z_i` 是 Stage 1 已存储的有效 chunk 分数，`x_i=z_i/2`。正常域要求至少一个 `z_i>-1e9`、全部分数有限、三个更新的分母和候选阈值有限。
- 最大值初始化沿用 Stage 1 的 `-1e9` 哨兵；`t_hi=max(-1e9,max(z))/2`，`t_lo=t_hi-1`，`t=(t_lo+t_hi)/2`。
- 每轮按固定源码计算 `f=Σ[x_i-t]_+²-1`、`f′=-2Σ[x_i-t]_+`、`f″=2Σ1{x_i>t}`，再计算 Halley 候选、更新括区间并用 `EPS=1e-5` 检查候选。
- 第三轮**更新后**的阈值用于概率、GQA 组均值、可选 `estimate_diag` 变换，以及 `nchunks` 位置恒为 1 的尾块 prior/bitmask。

## 候选认证命题

节点覆盖一组 chunk，且 `U_N≥max_{i∈N}x_i`。先用上界与已精算叶认证每头最大值。若某轮开始时参考路由与新路由的 `(t,t_lo,t_hi)` 相同，并且 `U_N≤t`，则该节点对本轮 `f/f′/f″` 的贡献均为零。对其余节点补算精确分数，可以归纳得到相同的三个实数状态。**这只允许当前轮暂缓**；下一轮阈值下降时必须重新检查。最后还要按 `t_3` 补算可能有正概率的节点，才能构造 prior 和 bitmask。

`scripts/e0_oracle.py` 以独立的全扫描路径和暂缓路径验证这个命题，并记录最大值认证成本、总精算叶数和重新激活次数。L2 球界只作为 CPU 合成例子的可计算上界；E0 会用已知精确分数检查每个界，但实际 GPU 算法不能借此检查。E0 的工作量数字不能作为 E2/E3 的性能估计。

## 不能从 E0 推出的结论

- `tl.dot(..., input_precision="ieee")`、缩放、分组约减及 FP32 舍入的保守裕量 `δ` 尚未证明。上界覆盖实数点积不等于覆盖 GPU 实际分数。
- NumPy 的约减顺序与 Triton 的 tile/warp DAG 不同；E0 状态相等不保证 GPU prior 逐位相等。正常域 GPU 目标仍是 bitmask 零失配、prior 误差不超过预注册容差。
- 全遮罩、零完整 chunk、非有限分数或退化分母留给原实现回退；回退成本须在 E3 计入。
- 尚无真实 H20 trace、实际界紧度、route 时间或 TPOT。E1 完成后才能判定是否继续 A0/A1/A2。
