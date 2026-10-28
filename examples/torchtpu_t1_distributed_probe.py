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
import subprocess
import sys
from datetime import timedelta


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = runpy.run_path(
    str(ROOT / "src/trainlm/runtime/torchtpu_distributed_probe.py")
)
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
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        print("result=timeout; worker process group terminated")
        return 1

    records = []
    for line in stdout.splitlines():
        if line.startswith(PREFIX):
            try:
                records.append(json.loads(line[len(PREFIX):]))
            except json.JSONDecodeError:
                print("result=malformed worker report")
                return 1
    summary = CONTRACT["summarize_workers"](records, expected_world)
    if process.returncode != 0:
        summary["status"] = "failed"
        summary["worker_exit_code"] = process.returncode
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
    return launch(args.expected_world_size, args.timeout)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Runtime exception text may include coordinator addresses or secrets.
        print("probe_error_type=" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
