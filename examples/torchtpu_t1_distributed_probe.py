"""Opt-in T1.4 single-host TPU collective smoke; no training or checkpoints.

Run --inspect without a TPU to see only the current launch-environment shape.
Run --run on an isolated v5e-8 host after the single-device checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import sys
from datetime import timedelta


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = runpy.run_path(
    str(ROOT / "src/trainlm/runtime/torchtpu_distributed_probe.py")
)
RUN_PROCESS = runpy.run_path(
    str(ROOT / "src/trainlm/runtime/torchtpu_probe_process.py")
)["run_probe_process"]
PREFIX = "TRAINLM_T1_DISTRIBUTED="


def worker() -> int:
    import torch
    import torch.distributed as dist

    try:
        dist.init_process_group(backend="tpu_dist", timeout=timedelta(seconds=120))
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        value = torch.ones(1, device=torch.device("tpu")) * (rank + 1)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world * (world + 1) // 2
        observed = value.item()
        report = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world,
            "collective": "passed" if observed == expected else "failed",
        }
        print(PREFIX + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["collective"] == "passed" else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(expected_world: int, timeout: int) -> int:
    if expected_world < 2 or timeout < 1:
        raise ValueError("expected world must be >= 2 and timeout must be positive")
    # Import registers TorchTPU, but do not allocate a tensor in this parent.
    import torch

    report = CONTRACT["prepare_launch_environment"](torch, os.environ)
    print("launch_environment=" + json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "ready" or report["world_size"] != expected_world:
        print("result=launch environment unavailable or world size mismatch")
        return 1

    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", f"--nproc-per-node={expected_world}",
        str(Path(__file__).resolve()), "--worker",
    ]
    process = RUN_PROCESS(command, cwd=ROOT, timeout=timeout)
    if process["status"] in {"timeout", "interrupted"}:
        print("collective_summary=" + json.dumps({
            key: process[key] for key in (
                "status", "error_type", "cleanup_completed", "exit_code"
            )
        }, sort_keys=True))
        return 1

    records = []
    for line in process["stdout"].splitlines():
        if line.startswith(PREFIX):
            try:
                record = json.loads(line[len(PREFIX):])
                if not isinstance(record, dict):
                    raise ValueError("worker record must be an object")
                records.append(record)
            except ValueError:
                print("result=malformed worker report")
                return 1
    summary = CONTRACT["summarize_workers"](records, expected_world)
    if process["status"] != "completed":
        summary["status"] = "failed"
    summary["worker_exit_code"] = process["exit_code"]
    summary["fatal_markers"] = process["fatal_markers"]
    summary["error_type"] = process["error_type"]
    print("collective_summary=" + json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inspect", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.worker:
        return worker()
    if args.inspect:
        print(json.dumps(CONTRACT["inspect_launch_environment"](os.environ), sort_keys=True))
        return 0
    previous_handler = signal.signal(signal.SIGTERM, _interrupt_on_term)
    try:
        return launch(args.expected_world_size, args.timeout)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _interrupt_on_term(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Runtime exception text may include coordinator addresses or secrets.
        print("probe_error_type=" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
