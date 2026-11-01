"""Minimal vLLM instrumentation for the Gate 0 host-work baseline.

Design goals:
  - no vLLM forks: monkey-patch class methods before the engine starts;
  - fail soft: phases that this vLLM version does not expose are reported,
    not fatal;
  - one CSV: host phase durations, a CUDA stream timing window, and a
    schedule-to-execute host backlog proxy per step.

This module must be imported *in the process that runs the engine*. The
bench script sets VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in
the same process; otherwise the parent's patches never see worker calls.

Phase hierarchy (parent column):
  schedule
    block_alloc
  execute
    prepare
    attn_metadata
      flashinfer_metadata
    sample
  postprocess

If a phase is missing in a given vLLM version, the analyzer falls back to
the parent duration.
"""

from __future__ import annotations

import csv
import functools
import importlib
import threading
import time
from collections import deque

try:
    import torch
except Exception:  # pragma: no cover - CPU-only hosts
    torch = None


# phase, parent, candidate (module, class, method) in priority order
PHASES: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "schedule",
        "",
        [("vllm.v1.core.sched.scheduler", "Scheduler", "schedule")],
    ),
    (
        "execute",
        "",
        [
            ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "execute_model"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "execute_model"),
        ],
    ),
    (
        "prepare",
        "execute",
        [
            ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "prepare_inputs"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "_prepare_inputs"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "prepare_inputs"),
        ],
    ),
    (
        "attn_metadata",
        "execute",
        [
            (
                "vllm.v1.worker.gpu.model_states.default",
                "DefaultModelState",
                "prepare_attn",
            ),
            (
                "vllm.v1.worker.gpu_model_runner",
                "GPUModelRunner",
                "_build_attention_metadata",
            ),
            (
                "vllm.v1.worker.gpu_model_runner",
                "GPUModelRunner",
                "_build_attn_metadata",
            ),
            (
                "vllm.v1.worker.gpu_model_runner",
                "GPUModelRunner",
                "build_attn_metadata",
            ),
        ],
    ),
    (
        "block_alloc",
        "schedule",
        [
            ("vllm.v1.core.kv_cache_manager", "KVCacheManager", "allocate_slots"),
            ("vllm.v1.core.kv_cache_manager", "KVCacheManager", "allocate"),
        ],
    ),
    (
        "flashinfer_metadata",
        "attn_metadata",
        [
            (
                "vllm.v1.attention.backends.flashinfer",
                "FlashInferMetadataBuilder",
                "build",
            )
        ],
    ),
    (
        "sample",
        "execute",
        [
            ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "sample"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "_sample"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "sample"),
        ],
    ),
    (
        "postprocess",
        "",
        [
            (
                "vllm.v1.engine.output_processor",
                "OutputProcessor",
                "process_outputs",
            )
        ],
    ),
    (
        "engine_step",
        "",
        [("vllm.v1.engine.core", "EngineCore", "step")],
    ),
    (
        "engine_step_bq",
        "",
        [("vllm.v1.engine.core", "EngineCore", "step_with_batch_queue")],
    ),
    (
        "engine_wait",
        "engine_step",
        [
            (
                "vllm.v1.executor.uniproc_executor",
                "AsyncOutputFuture",
                "result",
            )
        ],
    ),
    (
        "engine_update",
        "engine_step",
        [
            (
                "vllm.v1.core.sched.scheduler",
                "Scheduler",
                "update_from_output",
            )
        ],
    ),
    (
        "graph_replay",
        "execute",
        [("torch.cuda", "CUDAGraph", "replay")],
    ),
]

FIELDS = [
    "step",
    "phase",
    "parent",
    "start_ns",
    "end_ns",
    "dur_ns",
    "kind",
    "value",
]


class Recorder:
    def __init__(
        self,
        path: str,
        nvtx: bool = False,
        device_events: bool = True,
    ) -> None:
        self.path = path
        self.nvtx = bool(nvtx and torch is not None and torch.cuda.is_available())
        self.device_events = bool(
            device_events and torch is not None and torch.cuda.is_available()
        )
        self._lock = threading.Lock()
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._writer.writeheader()
        self._enabled = False
        self._scheduled = 0
        self._executed = 0
        self._pending: deque[int] = deque()
        self._events: list[tuple[int, object, object]] = []
        self._buffer: list[dict] = []
        self._rows = 0
        self._epoch = None
        self._epoch_host_ns = 0

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._enabled = True
            self._scheduled = 0
            self._executed = 0
            self._pending.clear()
            self._events.clear()
            self._epoch_host_ns = time.perf_counter_ns()
            if self.device_events and torch is not None:
                self._epoch = torch.cuda.Event(enable_timing=True)
                self._epoch.record()
                self._write_locked(
                    0, "epoch", "", 0, 0, 0, "epoch", self._epoch_host_ns
                )

    def finalize(self) -> None:
        with self._lock:
            self._enabled = False

        if self.device_events and self._events and torch is not None:
            torch.cuda.synchronize()
            with self._lock:
                for step, ev0, ev1 in self._events:
                    try:
                        start_ms = self._epoch.elapsed_time(ev0)
                        end_ms = self._epoch.elapsed_time(ev1)
                    except Exception:
                        continue
                    self._write_locked(
                        step,
                        "device_window",
                        "",
                        int(start_ms * 1e3),
                        int(end_ms * 1e3),
                        int((end_ms - start_ms) * 1e3),
                        "device_window",
                        0,
                    )
                self._events.clear()

        with self._lock:
            for entry in self._buffer:
                self._writer.writerow(entry)
            self._buffer.clear()
            self._fh.flush()
            self._fh.close()

    @property
    def rows(self) -> int:
        return self._rows

    # -- step tracking -------------------------------------------------
    def start_step(self) -> int:
        with self._lock:
            self._scheduled += 1
            sid = self._scheduled
            self._pending.append(sid)
            return sid

    def current_step(self) -> int:
        with self._lock:
            return self._pending[0] if self._pending else max(1, self._scheduled)

    def k_now(self) -> int:
        with self._lock:
            return self._scheduled - self._executed

    def last_step(self) -> int:
        with self._lock:
            return max(1, self._scheduled)

    def finish_execute(self, step: int, k_before: int) -> None:
        with self._lock:
            self._executed += 1
            if self._pending:
                self._pending.popleft()
            self._write_locked(step, "k_host_proxy", "", 0, 0, 0, "k", k_before)

    def add_resource(self, step: int, name: str, value: int) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._write_locked(step, name, "", 0, 0, 0, "resource", value)

    def add_route(self, step: int, name: str, value: str) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._write_locked(step, name, "", 0, 0, 0, "route", value)

    def record_scheduler_state(self, step: int, scheduler, output) -> None:
        """Snapshot what the scheduler was constrained by before this step."""
        def safe(fn):
            try:
                return int(fn())
            except Exception:
                return -1

        waiting = safe(lambda: len(scheduler.waiting))
        running = safe(lambda: len(scheduler.running))
        free_blocks = safe(
            lambda: scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        scheduled = safe(lambda: output.total_num_scheduled_tokens)
        preemptions = safe(
            lambda: sum(
                1 for request in scheduler.running if request.num_preemptions > 0
            )
        )
        for name, value in (
            ("waiting", waiting),
            ("running", running),
            ("free_blocks", free_blocks),
            ("scheduled_tokens", scheduled),
            ("running_preempted_ever", preemptions),
        ):
            self.add_resource(step, name, value)

    # -- writing -------------------------------------------------------
    def add_host(
        self,
        step: int,
        phase: str,
        parent: str,
        start_ns: int,
        end_ns: int,
    ) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._write_locked(
                step, phase, parent, start_ns, end_ns, end_ns - start_ns, "host", 0
            )

    def add_device_event(self, step: int, ev0: object, ev1: object) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._events.append((step, ev0, ev1))

    def _write_locked(
        self,
        step: int,
        phase: str,
        parent: str,
        start_ns: int,
        end_ns: int,
        dur_ns: int,
        kind: str,
        value: int | str,
    ) -> None:
        self._buffer.append(
            {
                "step": step,
                "phase": phase,
                "parent": parent,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "dur_ns": dur_ns,
                "kind": kind,
                "value": value,
            }
        )
        self._rows += 1


def _wrap(original, phase: str, parent: str, rec: Recorder):
    if getattr(original, "_gate0_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if not rec._enabled:  # warmup path
            return original(*args, **kwargs)

        if phase == "schedule":
            step = rec.start_step()
        elif phase in ("engine_step", "engine_step_bq"):
            step = 0  # resolved to the step scheduled inside this call
        else:
            step = rec.current_step()

        k_before = rec.k_now() if phase == "execute" else 0

        ev0 = ev1 = None
        if phase == "execute" and rec.device_events and torch is not None:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        if rec.nvtx:
            torch.cuda.nvtx.range_push(f"gate0:{phase}")

        t0 = time.perf_counter_ns()
        result = None
        try:
            result = original(*args, **kwargs)
            if phase == "flashinfer_metadata" and result is not None:
                rec.add_route(step, "prefill", type(result.prefill).__name__)
                rec.add_route(step, "decode", type(result.decode).__name__)
                rec.add_route(step, "num_prefills", str(result.num_prefills))
                rec.add_route(step, "num_decodes", str(result.num_decodes))
                decode_wrapper = getattr(result.decode, "wrapper", None)
                if decode_wrapper is not None:
                    rec.add_route(
                        step,
                        "decode_wrapper_cuda_graph",
                        str(getattr(decode_wrapper, "is_cuda_graph_enabled", "unknown")),
                    )
            return result
        finally:
            t1 = time.perf_counter_ns()
            if rec.nvtx:
                torch.cuda.nvtx.range_pop()
            if ev1 is not None:
                ev1.record()
                rec.add_device_event(step, ev0, ev1)
            if phase in ("engine_step", "engine_step_bq"):
                step = rec.last_step()
            rec.add_host(step, phase, parent, t0, t1)
            if phase == "execute":
                rec.finish_execute(step, k_before)
            if phase == "schedule" and result is not None:
                rec.record_scheduler_state(step, args[0], result)

    wrapper._gate0_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _patch_one(
    module_name: str,
    class_name: str,
    method_name: str,
    phase: str,
    parent: str,
    rec: Recorder,
) -> str:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return f"{phase}: import failed ({exc.__class__.__name__}: {exc})"

    cls = getattr(mod, class_name, None)
    if cls is None:
        return f"{phase}: class {class_name} missing in {module_name}"

    original = getattr(cls, method_name, None)
    if original is None:
        return f"{phase}: method {method_name} missing"

    if getattr(original, "_gate0_wrapped", False):
        return f"{phase}: already wrapped"

    setattr(cls, method_name, _wrap(original, phase, parent, rec))
    return f"{phase}: patched {module_name}.{class_name}.{method_name}"


def patch_all(rec: Recorder) -> list[str]:
    """Patch the first available candidate for each phase.

    Returns a human-readable status list; print it and keep it in the run log.
    """
    status: list[str] = []
    for phase, parent, candidates in PHASES:
        patched = False
        for module_name, class_name, method_name in candidates:
            msg = _patch_one(
                module_name, class_name, method_name, phase, parent, rec
            )
            status.append(msg)
            if msg.startswith(f"{phase}: patched"):
                patched = True
                break
        if not patched:
            status.append(f"{phase}: NOT PATCHED (analyzer will fall back)")
    return status
