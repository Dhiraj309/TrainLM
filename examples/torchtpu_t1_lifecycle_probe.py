#!/usr/bin/env python3
"""Opt-in, bounded native TorchTPU single-device restart/cleanup probe.

The default invocation only prints a plan. ``--run`` starts two or three
separate probe processes on a TPU host; it never falls back to CPU. This is
T1 lifecycle evidence, not native trainer or distributed certification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "examples/torchtpu_t1_v5e1_probe.py"
NORMALIZE = runpy.run_path(
    str(ROOT / "src/trainlm/runtime/torchtpu_discovery.py")
)["normalize_worker_report"]
PASS_MARKERS = (
    "tpu_detection=PASS (native PyTorch TPU tensor allocation)",
    "forward=PASS", "autograd=PASS", "optimizer_update=PASS",
    "result=single-device smoke passed; no distributed or performance claim",
)


def _terminate_group(process: object, kill_group: object) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            kill_group(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue


def _error_type(stderr: str) -> str | None:
    if "PyTorch import failed:" in stderr:
        return "ImportError"
    matches = re.findall(
        r"(?:^|\n)([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", stderr
    )
    return matches[-1] if matches else None


def run_once(
    index: int, output_dir: Path, timeout: int, *,
    process_factory: object = subprocess.Popen, kill_group: object = os.killpg,
) -> dict:
    """Launch exactly one device-owning child and retain only bounded facts."""
    report_path = output_dir / f"attempt-{index:02d}.json"
    command = [
        sys.executable, str(PROBE), "--device", "tpu", "--steps", "1",
        "--report-json", str(report_path),
    ]
    try:
        process = process_factory(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
    except Exception as exc:
        return {"attempt": index, "status": "failed", "report": None,
                "error_type": type(exc).__name__}
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_group(process, kill_group)
        return {"attempt": index, "status": "timeout", "report": None,
                "error_type": "TimeoutExpired"}
    except KeyboardInterrupt:
        _terminate_group(process, kill_group)
        raise
    except Exception as exc:
        _terminate_group(process, kill_group)
        return {"attempt": index, "status": "failed", "report": None,
                "error_type": type(exc).__name__}

    report = None
    if report_path.is_file():
        try:
            report = NORMALIZE(json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            report = {"schema_version": 1, "status": "invalid_report"}
    lines = set(stdout.splitlines())
    passed = (
        process.returncode == 0
        and report is not None and report["status"] == "ready"
        and all(marker in lines for marker in PASS_MARKERS)
    )
    return {
        "attempt": index,
        "status": "passed" if passed else "failed",
        "exit_code": process.returncode,
        "report": report,
        "missing_markers": [marker.split("=", 1)[0] for marker in PASS_MARKERS
                            if marker not in lines],
        "error_type": _error_type(stderr),
    }


def run_lifecycle(
    output_dir: Path, *, repeats: int = 2, timeout: int = 300,
    process_factory: object = subprocess.Popen, kill_group: object = os.killpg,
) -> dict:
    """Run sequentially; stop at the first failure and compare observations."""
    if not 2 <= repeats <= 3 or not 1 <= timeout <= 900:
        raise ValueError("repeats must be 2–3 and timeout must be 1–900 seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(tempfile.mkdtemp(prefix="session-", dir=output_dir))
    attempts = []
    for index in range(1, repeats + 1):
        try:
            attempt = run_once(index, session_dir, timeout,
                               process_factory=process_factory,
                               kill_group=kill_group)
        except KeyboardInterrupt:
            attempt = {"attempt": index, "status": "interrupted", "report": None,
                       "error_type": "KeyboardInterrupt"}
        attempts.append(attempt)
        if attempt["status"] != "passed":
            break
    properties = ("device_name", "topology", "device_count_visible",
                  "torch_version", "torch_tpu_version", "libtpu_version")
    fingerprints = [json.dumps({key: attempt["report"].get(key) for key in properties},
                               sort_keys=True) for attempt in attempts
                    if attempt["status"] == "passed"]
    consistent = len(fingerprints) == repeats and len(set(fingerprints)) == 1
    passed = len(attempts) == repeats and all(
        attempt["status"] == "passed" for attempt in attempts
    ) and consistent
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": "single_device_process_restart",
        "artifact_dir": str(session_dir),
        "requested_attempts": repeats,
        "completed_attempts": len(attempts),
        "observations_consistent": consistent,
        "attempts": attempts,
    }
    destination = session_dir / "summary.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=session_dir,
        prefix="summary.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)
    return summary


def _interrupt_on_term(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run on a TPU host.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("t1_lifecycle_reports"))
    args = parser.parse_args()
    if not 2 <= args.repeats <= 3 or not 1 <= args.timeout <= 900:
        parser.error("repeats must be 2–3 and timeout must be 1–900 seconds")
    if not args.run:
        print(json.dumps({"run_enabled": False, "device": "tpu",
                          "repeats": args.repeats, "timeout_seconds": args.timeout},
                         sort_keys=True))
        return 0
    if sys.platform != "linux":
        parser.error("the opt-in lifecycle probe requires Linux process groups")
    previous_handler = signal.signal(signal.SIGTERM, _interrupt_on_term)
    try:
        report = run_lifecycle(args.output_dir, repeats=args.repeats,
                               timeout=args.timeout)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
