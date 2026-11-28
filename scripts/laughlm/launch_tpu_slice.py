"""Multi-chip Cloud TPU (v5e-8 / v6e-8) slice launcher via torchrun."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.laughlm.common import configure_preimport_tpu_compile_cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multi-chip LaughLM training across an 8-chip v5e-8/v6e-8 TPU slice."
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--arch", default="laughlm_loop", choices=("dense", "laughlm_loop"))
    parser.add_argument("--loop-preset", default="trc-35b-v6e8", choices=("pilot-1b", "trc-35b-v6e8"))
    parser.add_argument("--gcs-shard-dir", default=None)
    parser.add_argument("--repo-id", "--dataset-repo", dest="repo_id", default=None)
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-stop", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--compile-cache-dir",
        "--tpu-cache-dir",
        dest="tpu_cache_dir",
        default=str(Path.home() / ".cache" / "trainlm" / "tpu_compile"),
    )
    parser.add_argument("--output-dir", default="runs/laughlm_loop_v6e8")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--distributed-strategy", default="ddp", choices=("auto", "ddp", "fsdp2", "none"))
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--deep-init-scale", action="store_true")
    parser.add_argument("--loop-depth-schedule", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the torchrun command & cache env without spawning.")
    return parser.parse_args(argv)


def build_launch_command(args: argparse.Namespace) -> list[str]:
    script_path = str((REPO_ROOT / "scripts" / "laughlm" / "stage1_base_30b.py").resolve())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc_per_node}",
        script_path,
        "--arch",
        args.arch,
        "--loop-preset",
        args.loop_preset,
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--distributed-strategy",
        args.distributed_strategy,
        "--compile-cache-dir",
        args.tpu_cache_dir,
        "--output-dir",
        args.output_dir,
    ]
    if args.gcs_shard_dir:
        cmd.extend(["--gcs-shard-dir", args.gcs_shard_dir])
    elif args.repo_id:
        cmd.extend(["--repo-id", args.repo_id, "--subfolder", args.subfolder])
    if getattr(args, "shard_start", 0):
        cmd.extend(["--shard-start", str(args.shard_start)])
    if getattr(args, "shard_stop", None) is not None:
        cmd.extend(["--shard-stop", str(args.shard_stop)])
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.grad_accum is not None:
        cmd.extend(["--grad-accum", str(args.grad_accum)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if getattr(args, "max_steps", None) is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if getattr(args, "deep_init_scale", False):
        cmd.append("--deep-init-scale")
    if getattr(args, "loop_depth_schedule", None):
        cmd.extend(["--loop-depth-schedule", str(args.loop_depth_schedule)])
    return cmd


def build_tpu_cache_env(
    tpu_cache_dir: str, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    return configure_preimport_tpu_compile_cache(tpu_cache_dir, base_env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = build_launch_command(args)
    env = build_tpu_cache_env(args.tpu_cache_dir)
    if args.dry_run:
        print(
            "[TPU Slice Launcher Dry-Run] "
            f"TORCH_TPU_CACHE_DIR={env['TORCH_TPU_CACHE_DIR']} "
            f"TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT={env['TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT']} "
            f"XLA_COMPILATION_CACHE_PATH={env['XLA_COMPILATION_CACHE_PATH']} | "
            + " ".join(cmd),
            flush=True,
        )
        return 0
    if not str(args.tpu_cache_dir).startswith("gs://"):
        Path(args.tpu_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
