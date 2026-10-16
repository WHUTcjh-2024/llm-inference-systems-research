"""Fail closed if source, model snapshot, or package versions have drifted."""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

PACKAGES = {
    "torch": "2.6.0",
    "triton": "3.2.0",
    "transformers": "4.56.0",
    "tokenizers": "0.22.0",
    "datasets": "2.20.0",
    "accelerate": "1.10.1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--rehash-weight", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))

    head = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != lock["upstream_commit"]:
        raise RuntimeError(f"Upstream drift: {head}")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"Tracked upstream files were modified:\n{dirty}")

    router = args.source / "dash_attn/decoding/stage2_block_selection.py"
    digest = hashlib.sha256(router.read_bytes()).hexdigest()
    if digest != lock["router_source_sha256"]:
        raise RuntimeError(f"Router source drift: {digest}")

    for package, expected in PACKAGES.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")

    manifest = json.loads(
        (args.checkpoint / "e1_download_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest["repo"] != lock["model_repo"]
        or manifest["revision"] != lock["model_revision"]
    ):
        raise RuntimeError("Checkpoint revision mismatch")
    weight = args.checkpoint / lock["model_weight_file"]
    if weight.stat().st_size != lock["model_weight_bytes"]:
        raise RuntimeError("Checkpoint weight size mismatch")
    if (
        manifest["weights_bytes"] != lock["model_weight_bytes"]
        or manifest["weights_sha256"] != lock["model_weight_sha256"]
    ):
        raise RuntimeError("Checkpoint SHA-256 manifest mismatch")
    if args.rehash_weight:
        digest = hashlib.sha256()
        with weight.open("rb") as file:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != lock["model_weight_sha256"]:
            raise RuntimeError("Checkpoint bytes changed after preparation")

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "max_position_embeddings": 16384,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "chunk_size": 64,
        "estimate_diagonal": True,
        "scaling_factor": 0.34,
        "sigma": 100_000_000,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError(
            "Checkpoint configuration is not the expected DashAttention model"
        )
    print("Pinned source, package versions, and checkpoint verified.")


if __name__ == "__main__":
    main()
