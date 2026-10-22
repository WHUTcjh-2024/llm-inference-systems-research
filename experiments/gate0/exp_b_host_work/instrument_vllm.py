"""Minimal vLLM instrumentation for the Gate 0 host-work baseline.

Design goals:
  - no vLLM forks: monkey-patch class methods before the engine starts;
  - fail soft: phases that this vLLM version does not expose are reported,
    not fatal;
  - one CSV: host phase durations, device time, and k_actual per step.

This module must be imported *in the process that runs the engine*. The
bench script sets VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in
the same process; otherwise the parent's patches never see worker calls.

Phase hierarchy (parent column):
  schedule
    block_alloc
  execute
    prepare
      attn_metadata
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
        [("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "execute_model")],
    ),
    (
        "prepare",
        "execute",
        [
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "_prepare_inputs"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "prepare_inputs"),
        ],
    ),
    (
        "attn_metadata",
        "prepare",
        [
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
        "sample",
        "execute",
        [
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
        self._rows = 0

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._enabled = True
            self._scheduled = 0
            self._executed = 0
            self._pending.clear()
            self._events.clear()

    def finalize(self) -> None:
        with self._lock:
            self._enabled = False

        if self.device_events and self._events and torch is not None:
            torch.cuda.synchronize()
            with self._lock:
                for step, ev0, ev1 in self._events:
                    try:
                        ms = ev0.elapsed_time(ev1)
                    except Exception:
                        continue
                    self._write_locked(
                        step, "device", "", 0, 0, int(ms * 1e6), "device", 0
                    )
                self._events.clear()

        with self._lock:
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

    def finish_execute(self, step: int, k_before: int) -> None:
        with self._lock:
            self._executed += 1
            if self._pending:
                self._pending.popleft()
            self._write_locked(step, "k_actual", "", 0, 0, 0, "k", k_before)

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
        value: int,
    ) -> None:
        self._writer.writerow(
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
        try:
            return original(*args, **kwargs)
        finally:
            t1 = time.perf_counter_ns()
            if rec.nvtx:
                torch.cuda.nvtx.range_pop()
            if ev1 is not None:
                ev1.record()
                rec.add_device_event(step, ev0, ev1)
            rec.add_host(step, phase, parent, t0, t1)
            if phase == "execute":
                rec.finish_execute(step, k_before)

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
