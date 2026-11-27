"""Poll checkpoint directory and atomically upload completed checkpoints to HF Hub."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

DEFAULT_HF_REPO = "Dhiraj45/laughlm-135m-v2"
_FINAL_MARKERS = ("trainer_state.pt", "config.json", "model.safetensors", "pytorch_model.bin")


def is_completed_checkpoint(path: Path) -> bool:
    """Return True only for finalized checkpoint directories with no `.tmp` artifacts inside."""
    name = path.name
    if name.startswith(".") or ".tmp" in name or name.endswith(".tmp"):
        return False
    if not path.is_dir():
        return False
    entries = list(path.iterdir())
    if any(".tmp" in child.name or child.name.endswith(".tmp") for child in entries):
        return False
    return any((path / marker).is_file() for marker in _FINAL_MARKERS)


def scan_completed_checkpoints(checkpoint_dir: Path) -> tuple[Path, ...]:
    """Return sorted finalized checkpoint directories under ``checkpoint_dir``."""
    if not checkpoint_dir.is_dir():
        return ()
    return tuple(sorted(p for p in checkpoint_dir.iterdir() if is_completed_checkpoint(p)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch checkpoint dir and upload to HF Hub.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--once", action="store_true", help="Upload once and exit.")
    return parser.parse_args(argv)


def upload_new_checkpoints(
    checkpoint_dir: Path,
    *,
    hf_repo: str,
    uploaded: set[str],
    private: bool = False,
    upload_fn: object | None = None,
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi() if upload_fn is None else None
    if api is not None:
        api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)
    newly_uploaded: list[str] = []
    for ckpt in scan_completed_checkpoints(checkpoint_dir):
        if ckpt.name in uploaded:
            continue
        if upload_fn is not None:
            upload_fn(ckpt, hf_repo)
        else:
            assert api is not None
            api.upload_folder(
                folder_path=str(ckpt),
                path_in_repo=ckpt.name,
                repo_id=hf_repo,
                repo_type="model",
                ignore_patterns=["*.tmp", "*.tmp.*", ".*.tmp.*"],
                commit_message=f"Upload checkpoint {ckpt.name}",
            )
        uploaded.add(ckpt.name)
        newly_uploaded.append(ckpt.name)
        print(f"[Uploader] Uploaded {ckpt} -> {hf_repo}/{ckpt.name}", flush=True)
    return newly_uploaded


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.checkpoint_dir)
    uploaded: set[str] = set()
    while True:
        upload_new_checkpoints(
            root,
            hf_repo=args.hf_repo,
            uploaded=uploaded,
            private=args.private,
        )
        if args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
