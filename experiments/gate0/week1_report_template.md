# Week 1 决策报告

日期：
GPU / 驱动 / CUDA：
vLLM commit：
模型：
是否锁频：

## A. exec 克隆语义

| 项目 | 结果 |
|---|---|
| 图含 allocation/free 节点时第二次 instantiate | |
| 两次 instantiate 的 workspace 地址是否相同 | |
| 并发 replay 的 race 率 | |
| e1 显存增量 | |
| e2 显存增量 | |
| serial / concurrent 的 wall time | |
| Nsight 节点时间线是否证实两条 stream 重叠 | |
| 结论：克隆能否服务 `k` 个 in-flight step | |

## B. 残余 host 工作

| 负载 | async | H_step (µs) | H_res (µs) | D (µs) | H_res/step_time | k_actual (median) |
|---|---|---|---|---|---|---|
| uniform | on | | | | | |
| mixed | on | | | | | |
| mixed | off | | | | | |
| preemption | on | | | | | |

如果无法准确测到 device time，暂不判定 `H_res/step_time` 门槛。
`H_res/period` 只能作为辅助描述，并注明分母是 scheduler cadence。

当前探索性采集器的 `device_window` 和 `k_host_proxy` 不能直接填入 `D`、
`k_actual`；`H_res` 也须经 CUDA/NVTX 时间线确认不可图化部分后才能填。

### 归因

- `k` 主要被 `k_res` 卡住的比例：
- `k` 主要被 `k_dep` 卡住的比例：
- Python/GIL 等实现因素：
- 出现的 surprise（S1/S2/S3）：

## 决策

- [ ] 进入 Stage 1（`k_res` 主导，`H_res/step_time ≥5%`）
- [ ] 改写 L3 为 device-side planning（`k_dep` 主导但可移除）
- [ ] 转 Plan B（`k_dep` 恒为 1，或 `H_res/step_time <5%` 且 A100 复测仍低）

理由：

## 下一步（最多三条）

1.
2.
3.
