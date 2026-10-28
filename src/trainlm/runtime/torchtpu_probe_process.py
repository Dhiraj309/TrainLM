"""Linux process isolation for opt-in native TPU probes; no torch import.

Raw child output stays local. Callers should export only normalized evidence,
never this internal result dictionary wholesale.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
from typing import Any, Callable, Sequence


_FATAL_MARKERS = {
    "core_dump": "RAW: Dumping core",
    "shutdown_hang": "exit() hanging: exiting process",
    "python_fatal": "Fatal Python error:",
}


def fatal_runtime_markers(stdout: str, stderr: str) -> list[str]:
    """Return stable failure labels, not the potentially sensitive log lines."""
    return [label for label, marker in _FATAL_MARKERS.items()
            if marker in stdout or marker in stderr]


def _error_type(stderr: str) -> str | None:
    if "PyTorch import failed:" in stderr:
        return "ImportError"
    matches = re.findall(
        r"(?:^|\n)([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", stderr
    )
    return matches[-1] if matches else None


def terminate_process_group(
    process: Any, kill_group: Callable, *, grace: int = 5,
) -> bool:
    """Reap the leader and kill surviving descendants even if it exits first."""
    clean = True
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            kill_group(process.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            clean = False
        try:
            process.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                clean = False
        except Exception:
            clean = False
        # Do not return after SIGTERM: a child may have closed its pipes and
        # ignored SIGTERM while the leader exited successfully.
    return clean


def run_probe_process(
    command: Sequence[str], *, cwd: Path, timeout: int,
    process_factory: Callable = subprocess.Popen,
    kill_group: Callable = os.killpg, shutdown_grace: int = 5,
) -> dict[str, Any]:
    """Bound a fresh process group and reject fatal runtime teardown output.

    ``completed`` describes successful process exit only. Each probe still has
    to verify its own semantic checks before reporting a passed result.
    """
    if not 0 < timeout <= 3600 or not 1 <= shutdown_grace <= 30:
        raise ValueError("timeout must be 1–3600s and shutdown grace 1–30s")
    result: dict[str, Any] = {
        "status": "failed", "exit_code": None, "error_type": None,
        "fatal_markers": [], "cleanup_completed": None,
        "stdout": "", "stderr": "",
    }
    try:
        process = process_factory(
            list(command), cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        return result
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (KeyboardInterrupt, Exception) as exc:
        result["status"] = (
            "timeout" if isinstance(exc, subprocess.TimeoutExpired)
            else "interrupted" if isinstance(exc, KeyboardInterrupt)
            else "failed"
        )
        result["error_type"] = type(exc).__name__
        if isinstance(exc, subprocess.TimeoutExpired):
            for key, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                result[key] = value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result["cleanup_completed"] = terminate_process_group(
            process, kill_group, grace=shutdown_grace,
        )
        result["exit_code"] = process.returncode
        return result
    result.update(stdout=stdout, stderr=stderr, exit_code=process.returncode)
    result["fatal_markers"] = fatal_runtime_markers(stdout, stderr)
    result["error_type"] = _error_type(stderr)
    if process.returncode == 0 and not result["fatal_markers"]:
        result["status"] = "completed"
    return result
