"""Download the immutable model snapshot directly onto the AutoDL data disk."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if args.repo != lock["model_repo"] or args.revision != lock["model_revision"]:
        raise RuntimeError("Download target differs from the E1 lock")

    args.dest.mkdir(parents=True, exist_ok=True)
    actual = Path(
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=args.dest,
        )
    )
    weight = actual / lock["model_weight_file"]
    if weight.stat().st_size != lock["model_weight_bytes"]:
        raise RuntimeError("Checkpoint byte size differs from the pinned HF LFS object")
    digest = hashlib.sha256()
    with weight.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != lock["model_weight_sha256"]:
        raise RuntimeError("Checkpoint SHA-256 differs from the pinned HF LFS object")
    metadata = {
        "repo": args.repo,
        "revision": args.revision,
        "snapshot_path": str(actual.resolve()),
        "weights_bytes": weight.stat().st_size,
        "weights_sha256": digest.hexdigest(),
    }
    (actual / "e1_download_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
