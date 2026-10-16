# E1 数据发布约定

当前没有 H20 实测数据，本目录不放占位数值。

运行完成后，可把经过审核的 `results.json` 中的版本、环境、4 个 case 的 TPOT/Stage 1–3/密度摘要整理为小型 JSON 提交。原始 `trace-*.npz`、checkpoint、HF 缓存、完整日志和可能含机器信息的 `nvidia-smi.txt` 留在受控存储，另做备份。任何论文表格都须能追溯到原始 run ID 与固定源码/模型 revision。
