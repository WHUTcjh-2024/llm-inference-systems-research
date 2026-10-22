# Week 1 决策报告

日期：
GPU / 驱动 / CUDA：
vLLM commit：
模型：
是否锁频：

## A. exec 克隆语义

| 项目 | 结果 |
|---|---|
| 两次 instantiate 的 workspace 地址是否相同 | |
| 并发 replay 的 race 率 | |
| e1 显存增量 | |
| e2 显存增量 | |
| serial / concurrent 的 wall time | |
| 结论：克隆能否服务 `k` 个 in-flight step | |

## B. 残余 host 工作

| 负载 | async | H_step (µs) | H_res (µs) | D (µs) | H_res/step_time | k_actual (median) |
|---|---|---|---|---|---|---|
| uniform | on | | | | | |
| mixed | on | | | | | |
| mixed | off | | | | | |
| preemption | on | | | | | |

如果 device time 为 0（CUDA event 没打到 worker stream），改填 `H_res/period`，
并在备注里说明分母是 scheduler cadence。

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
