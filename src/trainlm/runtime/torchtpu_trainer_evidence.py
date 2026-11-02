"""Sanitize evidence from the isolated native trainer smoke process.

This module has no Torch import and can be loaded by a notebook before TPU
initialization. Raw child logs are deliberately excluded from its result.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import runpy
from typing import Any, Mapping


_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9._+\-]{1,96}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}\Z")
_CHECKS = ("train", "export", "compile", "resume", "exact_weight_match")
_MARKERS = {
    "train": "train=PASS", "export": "export=PASS",
    "compile": "compile=PASS", "resume": "resume=PASS",
    "exact_weight_match": "exact_weight_match=PASS",
}
_PACKAGES = ("torch", "torch_tpu", "libtpu", "transformers")
_FATAL_MARKERS = {"core_dump", "shutdown_hang", "python_fatal"}


def build_trainer_report(
    process: Mapping[str, Any], *, source_revision: str,
    compile_requested: bool, resume_requested: bool,
) -> dict[str, Any]:
    """Require matching process, phase, and device evidence before a pass."""
    if not _SHA.fullmatch(source_revision):
        raise ValueError("source_revision must be a full Git SHA")
    if type(compile_requested) is not bool or type(resume_requested) is not bool:
        raise ValueError("requested modes must be booleans")
    if compile_requested and resume_requested:
        raise ValueError("compile and resume checks must run separately")

    raw_status = process.get("status")
    process_status = raw_status if isinstance(raw_status, str) and raw_status in {
        "completed", "failed", "timeout", "interrupted",
    } else "failed"
    exit_code = process.get("exit_code")
    if type(exit_code) is not int:
        exit_code = None
    error_type = process.get("error_type")
    if not isinstance(error_type, str) or not _SAFE_TEXT.fullmatch(error_type):
        error_type = None
    raw_fatal_markers = process.get("fatal_markers")
    invalid_fatal_markers = (not isinstance(raw_fatal_markers, list)
                             or any(not isinstance(marker, str)
                                    or marker not in _FATAL_MARKERS
                                    for marker in raw_fatal_markers))
    fatal_markers = (sorted(set(raw_fatal_markers))
                     if not invalid_fatal_markers else [])
    cleanup_completed = process.get("cleanup_completed")
    if type(cleanup_completed) is not bool:
        cleanup_completed = None
    stdout = process.get("stdout")
    if not isinstance(stdout, str):
        stdout = ""
    lines = stdout.splitlines()
    summaries = [line[len("trainer_summary="):] for line in lines
                 if line.startswith("trainer_summary=")]
    issues = []
    summary = None
    if len(summaries) != 1 or len(summaries[0]) > 8192:
        issues.append("missing_or_duplicate_summary")
    else:
        try:
            summary = json.loads(summaries[0])
        except (ValueError, TypeError):
            issues.append("invalid_summary")

    checks = {name: "unknown" for name in _CHECKS}
    device = None
    model = None
    packages = {name: None for name in _PACKAGES}
    if (isinstance(summary, dict)
            and type(summary.get("schema_version")) is int
            and summary["schema_version"] == 1):
        normalize = runpy.run_path(str(Path(__file__).with_name(
            "torchtpu_discovery.py")))["normalize_worker_report"]
        device = normalize(summary.get("device"))
        if (device.get("status") != "ready"
                or device.get("device_type") != "tpu"
                or device.get("device_count_visible") != 1):
            issues.append("invalid_device")
            device = None
        raw_model = summary.get("model")
        if raw_model is not None:
            if (not isinstance(raw_model, dict)
                    or not isinstance(raw_model.get("id"), str)
                    or not _MODEL_ID.fullmatch(raw_model["id"])
                    or not isinstance(raw_model.get("revision"), str)
                    or not _SHA.fullmatch(raw_model["revision"])):
                issues.append("invalid_model_revision")
            else:
                model = {"id": raw_model["id"], "revision": raw_model["revision"]}
        raw_packages = summary.get("packages")
        if not isinstance(raw_packages, dict):
            issues.append("invalid_packages")
        else:
            for name in _PACKAGES:
                version = raw_packages.get(name)
                if not isinstance(version, str) or not _SAFE_TEXT.fullmatch(version):
                    issues.append("invalid_packages")
                    break
                packages[name] = version
        raw_checks = summary.get("checks")
        if not isinstance(raw_checks, dict):
            issues.append("invalid_checks")
        else:
            for name in _CHECKS:
                value = raw_checks.get(name)
                if value not in {"passed", "not_run"}:
                    issues.append("invalid_checks")
                    break
                checks[name] = value
    elif summary is not None:
        issues.append("invalid_summary")

    expected = {
        "train": "passed", "export": "passed",
        "compile": "passed" if compile_requested else "not_run",
        "resume": "passed" if resume_requested else "not_run",
        "exact_weight_match": "passed" if resume_requested else "not_run",
    }
    if checks != expected:
        issues.append("phase_mismatch")
    required_markers = {_MARKERS[name] for name in _CHECKS
                        if expected[name] == "passed"}
    required_markers.add("result=trainer_smoke_passed")
    if not required_markers.issubset(set(lines)):
        issues.append("missing_markers")
    if (process_status != "completed" or exit_code != 0 or fatal_markers
            or invalid_fatal_markers):
        issues.append("process_failed")
    return {
        "schema_version": 1,
        "scope": "single_device_native_trainer_smoke",
        "status": "passed" if not issues else "failed",
        "source_revision": source_revision,
        "requested": {"compile": compile_requested, "resume": resume_requested},
        "process": {"status": process_status, "exit_code": exit_code,
                    "error_type": error_type, "fatal_markers": fatal_markers,
                    "cleanup_completed": cleanup_completed},
        "device": device,
        "model": model,
        "packages": packages,
        "checks": checks,
        "issues": sorted(set(issues)),
        "limitations": ["smoke_only", "does_not_certify_performance"],
    }
