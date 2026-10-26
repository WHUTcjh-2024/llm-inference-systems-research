# INVALID — unit bug

These files are from the first smoke run and are kept only as a failure
record.

`instrument_vllm.py` stored CUDA event times as `int(elapsed_time_ms * 1e6)`
while the analyzer interpreted them as microseconds. This inflated `D_step`,
`device_period`, `H_res/D_step` and `k_actual` by 1000× / broke the host-vs-
device comparison. `k_actual` in particular is meaningless here.

The subsequent run under `../smoke/` uses `int(elapsed_time_ms * 1e3)` and is
the first valid smoke sample.
