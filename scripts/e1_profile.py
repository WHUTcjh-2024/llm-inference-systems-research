"""E1: incremental-decode TPOT, route kernels, attention, and active density.

The two timing passes are separate. CUDA event instrumentation perturbs CPU
launch latency, so TPOT is always measured with the original upstream router.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import triton
from e1_metrics import mean_active_blocks, percentile, stats_ms, summary
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def load_official_prompt_builder(source: Path):
    path = source / "example/run_niah.py"
    spec = importlib.util.spec_from_file_location("dash_official_niah", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ForwardEndEvents(LogitsProcessor):
    def __init__(self, expected_events: int) -> None:
        self.events = [
            torch.cuda.Event(enable_timing=True) for _ in range(expected_events)
        ]
        self.recorded = 0

    def __call__(self, input_ids, scores):
        if self.recorded >= len(self.events):
            raise RuntimeError("More generation callbacks than expected")
        self.events[self.recorded].record()
        self.recorded += 1
        return scores

    def intervals_ms(self) -> list[float]:
        if self.recorded != len(self.events):
            raise RuntimeError(
                f"Expected {len(self.events)} callbacks, got {self.recorded}"
            )
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for a, b in zip(self.events, self.events[1:])]


class RouteEvents:
    """Time the unmodified upstream routing call in a separate decode pass."""

    def __init__(self, upstream_decoding, expected_calls: int) -> None:
        self.decoding = upstream_decoding
        self.original = upstream_decoding.score_blocks
        self.events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(expected_calls)
        ]
        self.host_call_ms: list[float] = []
        self.recorded = 0

    def score(self, *args, **kwargs):
        if self.recorded >= len(self.events):
            raise RuntimeError("More routing calls than expected")
        start, end = self.events[self.recorded]
        host_start = time.perf_counter_ns()
        start.record()
        result = self.original(*args, **kwargs)
        end.record()
        self.host_call_ms.append((time.perf_counter_ns() - host_start) / 1e6)
        self.recorded += 1
        return result

    def __enter__(self):
        self.decoding.score_blocks = self.score
        return self

    def __exit__(self, *_):
        self.decoding.score_blocks = self.original

    def per_token_ms(self, layers: int) -> dict[str, list[float]]:
        if self.recorded != len(self.events):
            raise RuntimeError(
                f"Expected {len(self.events)} routing calls, got {self.recorded}"
            )
        torch.cuda.synchronize()
        gpu_call_ms = [start.elapsed_time(end) for start, end in self.events]
        return {
            "gpu_stream": [
                sum(gpu_call_ms[i : i + layers])
                for i in range(0, len(gpu_call_ms), layers)
            ],
            "host_enqueue": [
                sum(self.host_call_ms[i : i + layers])
                for i in range(0, len(self.host_call_ms), layers)
            ],
        }


class StageEvents:
    """Instrument the two exact Triton launches from pinned stage2 source."""

    def __init__(self, upstream_decoding, trace_layers: int) -> None:
        self.decoding = upstream_decoding
        self.original_score = upstream_decoding.score_blocks
        self.original_full = upstream_decoding.full_attn
        self.trace_layers = trace_layers
        self.trace: list[dict[str, torch.Tensor]] = []
        self.first_input = None
        self.support: list[tuple[torch.Tensor, int]] = []
        self.samples: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "stage1": [],
            "stage2": [],
            "stage3": [],
        }

    def _record(self, name: str, operation) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        self.samples[name].append((start, end))

    def score(
        self, q, k_blk, chunk_size, attn_mask=None, estimate_diag=False, sigma=1.0e6
    ):
        from dash_attn.decoding import stage2_block_selection as stage

        batch, query_heads, _, dimension = q.shape
        _, kv_heads, chunks, _ = k_blk.shape
        if chunks == 0:
            raise RuntimeError("E1 expected at least one summarized KV chunk")
        groups = query_heads // kv_heads
        words = (chunks + 1 + 31) // 32
        if attn_mask is not None:
            left_pad = attn_mask.shape[-1] - attn_mask.sum(dim=-1)
            first_valid = (left_pad // chunk_size).to(torch.int32).to(q.device)
        else:
            first_valid = q
        block_m = max(16, triton.next_power_of_2(groups))
        block_n_k1 = 64 if chunks >= 64 else max(16, triton.next_power_of_2(chunks))
        scores = torch.empty(
            batch, query_heads, chunks, device=q.device, dtype=torch.float32
        )
        max_buf = torch.empty(batch, query_heads, device=q.device, dtype=torch.float32)

        def stage1() -> None:
            stage._scores_and_max_kernel[(batch, kv_heads)](
                q,
                k_blk,
                scores,
                max_buf,
                first_valid,
                chunks,
                q.stride(0),
                q.stride(1),
                q.stride(3),
                k_blk.stride(0),
                k_blk.stride(1),
                k_blk.stride(2),
                k_blk.stride(3),
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                max_buf.stride(0),
                max_buf.stride(1),
                1.0 / math.sqrt(dimension),
                GROUPS=groups,
                D=dimension,
                BLOCK_M=block_m,
                BLOCK_N=block_n_k1,
                HAS_ATTN_MASK=attn_mask is not None,
            )

        self._record("stage1", stage1)
        prior = torch.empty(
            batch, kv_heads, 1, chunks + 1, device=q.device, dtype=torch.float32
        )
        bmask = torch.empty(
            batch, kv_heads, 1, words, device=q.device, dtype=torch.int32
        )

        def stage2() -> None:
            stage._entmax_mean_pack_kernel[(batch, kv_heads)](
                scores,
                max_buf,
                prior,
                bmask,
                chunks,
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                max_buf.stride(0),
                max_buf.stride(1),
                prior.stride(0),
                prior.stride(1),
                prior.stride(3),
                bmask.stride(0),
                bmask.stride(1),
                bmask.stride(3),
                words,
                float(sigma),
                GROUPS=groups,
                BLOCK_M=block_m,
                BLOCK_N=32,
                N_ITER=3,
                ESTIMATE_DIAG=estimate_diag,
            )

        self._record("stage2", stage2)
        if self.first_input is None:
            self.first_input = (
                q.detach().clone(),
                k_blk.detach().clone(),
                chunk_size,
                attn_mask.detach().clone() if attn_mask is not None else None,
                estimate_diag,
                sigma,
            )
        self.support.append((bmask.detach(), chunks + 1))
        if len(self.trace) < self.trace_layers:
            self.trace.append(
                {
                    "q_scaled": q.detach().clone(),
                    "k_chunks": k_blk.detach().clone(),
                    "scores": scores.detach(),
                    "bmask": bmask.detach(),
                    "prior": prior.detach(),
                }
            )
        return bmask, prior

    def validate_first(self) -> None:
        if self.first_input is None or not self.trace:
            raise RuntimeError("No routing call available for parity validation")
        reference_mask, reference_prior = self.original_score(*self.first_input)
        measured = self.trace[0]
        if not torch.equal(measured["bmask"], reference_mask) or not torch.allclose(
            measured["prior"], reference_prior, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("Instrumented router differs from pinned upstream")

    def full(self, *args, **kwargs):
        result = None

        def operation() -> None:
            nonlocal result
            result = self.original_full(*args, **kwargs)

        self._record("stage3", operation)
        return result

    def __enter__(self):
        self.decoding.score_blocks = self.score
        self.decoding.full_attn = self.full
        return self

    def __exit__(self, *_):
        self.decoding.score_blocks = self.original_score
        self.decoding.full_attn = self.original_full

    def per_token_ms(self, name: str, layers: int, decode_tokens: int) -> list[float]:
        torch.cuda.synchronize()
        events = self.samples[name]
        if len(events) != layers * decode_tokens:
            raise RuntimeError(
                f"{name}: expected {layers * decode_tokens} layer calls, got {len(events)}"
            )
        elapsed = [start.elapsed_time(end) for start, end in events]
        return [sum(elapsed[i : i + layers]) for i in range(0, len(elapsed), layers)]

    def save_trace(self, path: Path, layers: int) -> None:
        if len(self.trace) != layers:
            raise RuntimeError(f"Expected {layers} trace layers, got {len(self.trace)}")
        torch.cuda.synchronize()
        arrays = {}
        for layer_idx, values in enumerate(self.trace):
            for name, value in values.items():
                if name == "bmask":
                    arrays[f"layer{layer_idx:02d}_{name}"] = value.cpu().numpy()
                else:
                    arrays[f"layer{layer_idx:02d}_{name}"] = value.float().cpu().numpy()
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as file:
            np.savez_compressed(file, **arrays)
        temp.replace(path)

    def support_per_token(
        self, layers: int, decode_tokens: int
    ) -> dict[str, list[float]]:
        if len(self.support) != layers * decode_tokens:
            raise RuntimeError("Unexpected number of bitmask samples")
        torch.cuda.synchronize()
        densities = []
        active_blocks = []
        possible_blocks = []
        for mask, possible in self.support:
            mean_active = mean_active_blocks(mask.cpu().numpy(), possible)
            active_blocks.append(mean_active)
            possible_blocks.append(float(possible))
            densities.append(mean_active / possible)

        def mean_by_token(values: list[float]) -> list[float]:
            return [
                statistics.mean(values[i : i + layers])
                for i in range(0, len(values), layers)
            ]

        return {
            "density": mean_by_token(densities),
            "active_blocks": mean_by_token(active_blocks),
            "possible_blocks": mean_by_token(possible_blocks),
        }


def generate(
    model, tokenizer, input_ids, attention_mask, new_tokens, *, active=False, timer=None
):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "return_active_blocks": active,
    }
    if timer is not None:
        kwargs["logits_processor"] = LogitsProcessorList([timer])
    with torch.inference_mode():
        output = model.generate(**kwargs)
    torch.cuda.synchronize()
    if output.shape[-1] - input_ids.shape[-1] != new_tokens:
        raise RuntimeError("Generation stopped before the requested token count")
    return output


def case_inputs(official, tokenizer, target_tokens: int, batch_size: int, digits: str):
    prompt, _, _ = official.build_niah_s_1_prompt(tokenizer, digits, target_tokens)
    encoded = official.encode_chat_prompt(tokenizer, prompt)
    ids = encoded.input_ids
    if not target_tokens - 128 <= ids.shape[-1] <= target_tokens:
        raise RuntimeError("Official prompt builder missed the requested length band")
    return ids.repeat(batch_size, 1).to("cuda"), encoded.attention_mask.repeat(
        batch_size, 1
    ).to("cuda")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    lock_digest = hashlib.sha256(args.lock.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source))
    official = load_official_prompt_builder(args.source)
    from dash_attn import decoding
    from dash_attn.models.llama import LlamaForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1 or "H20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected one visible H20")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        LlamaForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="dash_attn"
        )
        .to("cuda")
        .eval()
    )
    if (
        model.config.max_position_embeddings != 16384
        or model.config.scaling_factor != 0.34
    ):
        raise RuntimeError("Unexpected model configuration")
    output = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock_digest,
        "upstream_commit": lock["upstream_commit"],
        "model_revision": lock["model_revision"],
        "host": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "hostname": platform.node(),
        },
        "method": {
            "tpot": "CUDA event at each logits-processor callback; differences across decode forwards; original upstream router",
            "router_call": "Separate pass wrapping the unchanged upstream score_blocks call with CUDA events; host enqueue time also recorded",
            "segments": "CUDA events around pinned Stage1, Stage2, Stage3 calls; sums across all layers per decode token; separate pass",
            "prefill": "excluded from TPOT and segment summaries",
            "prompt": "official NIAH S-1 builder; same prompt repeated for batch 2",
        },
        "cases": {},
    }
    atomic_json(args.output / "results.json", output)
    for case in lock["cases"]:
        name = case["name"]
        try:
            ids, mask = case_inputs(
                official,
                tokenizer,
                case["target_tokens"],
                case["batch_size"],
                lock["digits"],
            )
            prompt_tokens = ids.shape[-1]
            if (
                prompt_tokens + lock["new_tokens"]
                > model.config.max_position_embeddings
            ):
                raise RuntimeError("Prompt plus generation exceeds model context limit")
            generate(model, tokenizer, ids, mask, lock["warmup_tokens"])
            tpot_trials = []
            for _ in range(lock["repetitions"]):
                if not tpot_trials:
                    torch.cuda.reset_peak_memory_stats()
                timer = ForwardEndEvents(lock["new_tokens"])
                generate(model, tokenizer, ids, mask, lock["new_tokens"], timer=timer)
                intervals = timer.intervals_ms()
                if len(intervals) != lock["new_tokens"] - 1:
                    raise RuntimeError("Incorrect TPOT interval count")
                tpot_trials.append(intervals)
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            layers = model.config.num_hidden_layers
            decode_tokens = lock["new_tokens"] - 1
            with RouteEvents(decoding, layers * decode_tokens) as route_events:
                generate(model, tokenizer, ids, mask, lock["new_tokens"])
            route_call_values = route_events.per_token_ms(layers)
            # Only decode calls are replaced. Prefill still executes the upstream code.
            with StageEvents(decoding, layers) as segments:
                generate(model, tokenizer, ids, mask, lock["new_tokens"])
            segments.validate_first()
            segments.save_trace(args.output / f"trace-{name}.npz", layers)
            stage_values = {
                stage: segments.per_token_ms(stage, layers, decode_tokens)
                for stage in ("stage1", "stage2", "stage3")
            }
            route_stage_sum_values = [
                a + b for a, b in zip(stage_values["stage1"], stage_values["stage2"])
            ]
            support_values = segments.support_per_token(layers, decode_tokens)
            flat_tpot = [value for trial in tpot_trials for value in trial]
            result = {
                "status": "complete",
                "prompt_tokens": prompt_tokens,
                "batch_size": case["batch_size"],
                "new_tokens": lock["new_tokens"],
                "peak_allocated_bytes_tpot_pass": peak_memory_bytes,
                "trace_file": f"trace-{name}.npz",
                "tpot_ms": stats_ms(flat_tpot),
                "tpot_trials_ms": tpot_trials,
                "stage_ms": {
                    key: stats_ms(values) for key, values in stage_values.items()
                },
                "stage_per_token_ms": stage_values,
                "router_call_ms": stats_ms(route_call_values["gpu_stream"]),
                "router_call_per_token_ms": route_call_values["gpu_stream"],
                "router_host_enqueue_ms": stats_ms(route_call_values["host_enqueue"]),
                "router_host_enqueue_per_token_ms": route_call_values["host_enqueue"],
                "router_stage_sum_ms": stats_ms(route_stage_sum_values),
                "router_call_share_of_tpot_p50_approx": percentile(
                    route_call_values["gpu_stream"], 0.5
                )
                / percentile(flat_tpot, 0.5),
                "active": {
                    "density": summary(support_values["density"]),
                    "density_per_token": support_values["density"],
                    "active_blocks_per_token": support_values["active_blocks"],
                    "possible_blocks_per_token": support_values["possible_blocks"],
                    "basis": "Exact bit count of real decode bitmasks; mean over layers, batch and KV heads",
                },
            }
            output["cases"][name] = result
            print(
                f"{name}: TPOT p50={result['tpot_ms']['p50_ms']:.3f} ms, "
                f"router call share≈{result['router_call_share_of_tpot_p50_approx']:.2%}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - persist this case and continue the E1 matrix
            output["cases"][name] = {"status": "failed", "error": repr(error)}
            print(f"{name}: FAILED: {error!r}", flush=True)
        finally:
            atomic_json(args.output / "results.json", output)
            torch.cuda.empty_cache()
    output["status"] = (
        "complete"
        if all(case["status"] == "complete" for case in output["cases"].values())
        else "partial_failure"
    )
    output["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output / "results.json", output)
    if output["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    run()
