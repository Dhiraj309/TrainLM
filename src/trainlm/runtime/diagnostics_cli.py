"""Read-only one-command package and native TPU diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any

from .torchtpu_discovery import discover_tpu, normalize_worker_report


def _source_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _package_profile(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"status": "source_unavailable", "scope": "installed_metadata_only"}
    script = root / "examples/torchtpu_t1_install_preflight.py"
    try:
        inspect = runpy.run_path(str(script))["inspect_profile"]
        report = inspect(root)
    except Exception as exc:
        return {"status": "check_failed", "error_type": type(exc).__name__}
    return {
        "status": report["status"],
        "scope": report["scope"],
        "issues": report["issues"],
        "installed_versions": report["installed_versions"],
        "pip_check": report["pip_check"],
    }


def collect_diagnostics() -> dict[str, Any]:
    """Collect a single read-only report; never installs packages or trains."""
    root = _source_root()
    revision = None
    if root is not None:
        try:
            candidate = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True, timeout=5, stderr=subprocess.DEVNULL,
            ).strip()
            if len(candidate) in (40, 64) and all(c in "0123456789abcdef" for c in candidate):
                revision = candidate
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import torch
    except Exception as exc:
        discovery = {"status": "torch_import_failed", "error_type": type(exc).__name__}
    else:
        discovery = normalize_worker_report(discover_tpu(torch))
    package_profile = _package_profile(root)
    statuses = {"incompatible", "check_failed", "torch_import_failed",
                "query_failed", "allocation_failed", "invalid_report"}
    status = "failed" if package_profile["status"] in statuses or discovery["status"] in statuses \
        else "diagnostics_complete"
    return {
        "schema_version": 1,
        "status": status,
        "scope": "read_only_runtime_diagnostics",
        "source_revision": revision,
        "python_version": sys.version.split()[0],
        "package_profile": package_profile,
        "native_tpu": discovery,
        "limitations": ["does_not_train", "does_not_certify_performance"],
    }
