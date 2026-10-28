"""Shareable T1 observations without importing torch or exporting raw logs.

This source-level contract consumes the existing probes. It does not launch
training, change dependencies, or promote observations to certification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import runpy
from typing import Any


NORMALIZE = runpy.run_path(str(Path(__file__).with_name("torchtpu_discovery.py")))[
    "normalize_worker_report"
]
FATAL = runpy.run_path(str(Path(__file__).with_name("torchtpu_probe_process.py")))[
    "fatal_runtime_markers"
]
_SMOKE_MARKERS = ("forward=PASS", "autograd=PASS", "optimizer_update=PASS",
                  "result=single-device smoke passed; no distributed or performance claim")
_FATAL_LABELS = {"core_dump", "shutdown_hang", "python_fatal"}
_VERSION = re.compile(r"[0-9][A-Za-z0-9.!+_-]{0,95}\Z")
_PACKAGES = {"trainlm", "torch", "torch-tpu", "libtpu", "transformers",
             "huggingface-hub", "pyyaml", "tpu-info", "torch-xla", "jax", "jaxlib"}


def _labels(value: Any, allowed: set[str]) -> set[str]:
    return {item for item in value if isinstance(item, str) and item in allowed} \
        if isinstance(value, list) else set()


def _process_fields(process: dict) -> dict:
    status = process.get("status")
    error_type = process.get("error_type")
    return {
        "status": status if status in {"failed", "timeout", "interrupted"} else "failed",
        "exit_code": process.get("exit_code") if type(process.get("exit_code")) is int else None,
        "error_type": error_type if isinstance(error_type, str) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,63}", error_type) else None,
        "fatal_markers": sorted(set(FATAL(process.get("stdout", ""),
                                          process.get("stderr", "")))
                                | _labels(process.get("fatal_markers"), _FATAL_LABELS)),
        "cleanup_completed": process.get("cleanup_completed")
        if type(process.get("cleanup_completed")) is bool else None,
    }


def _json_record(stdout: str, prefix: str) -> dict | None:
    records = [line[len(prefix):] for line in stdout.splitlines()
               if line.startswith(prefix)]
    if len(records) != 1 or len(records[0]) > 65536:
        return None
    try:
        record = json.loads(records[0])
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def smoke_evidence(process: dict | None, *, compiled: bool = False) -> dict:
    if process is None:
        return {"status": "not_run"}
    result = _process_fields(process)
    lines = set(process.get("stdout", "").splitlines())
    devices = {line.removeprefix("device=").split(":", 1)[0] for line in lines
               if line.startswith("device=")}
    device = next(iter(devices)) if len(devices) == 1 else None
    device = device if device in {"tpu", "cpu", "cuda"} else None
    record = _json_record(process.get("stdout", ""), "tpu_report=")
    report = NORMALIZE(record) if record is not None else None
    markers = (*_SMOKE_MARKERS, "compile=PASS", "compile=torch.compile enabled") \
        if compiled else (*_SMOKE_MARKERS, "compile=eager")
    missing = [marker.split("=", 1)[0] for marker in markers if marker not in lines]
    passed = (process.get("status") == "completed" and result["exit_code"] == 0
              and not result["fatal_markers"] and not missing and device is not None)
    if device == "tpu":
        passed = passed and report is not None and report["status"] == "ready"
    result.update(device=device, discovery=report, missing_checks=sorted(set(missing)),
                  scope="native_tpu" if device == "tpu" else "non_tpu")
    if passed:
        result["status"] = "passed"
    return result


def profile_evidence(process: dict | None) -> dict:
    if process is None:
        return {"status": "not_run"}
    result = _process_fields(process)
    try:
        raw = process.get("stdout", "")
        report = json.loads(raw) if len(raw) <= 65536 else None
    except ValueError:
        report = None
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1):
        return result
    status = report.get("status")
    valid = {"metadata_compatible", "incomplete", "incompatible", "check_failed"}
    expected_exit = 0 if status == "metadata_compatible" else 1
    if (not isinstance(status, str) or status not in valid or result["exit_code"] != expected_exit
            or process.get("status") not in {"completed", "failed"}
            or result["fatal_markers"]):
        return result
    if (not isinstance(report.get("installed_versions"), dict)
            or not isinstance(report.get("issues"), list)
            or any(not isinstance(report.get(field), list) for field in (
                "source_requirements", "torch_tpu_requirements"))):
        return dict(result, status="invalid_report")
    result.update(status=status, scope="installed_metadata_only")
    known_issues = {"source_python_mismatch", "torch_tpu_python_mismatch",
                    "requirement_conflict", "pip_conflicts", "missing_distributions",
                    "pip_check_failed"}
    result["issues"] = sorted(_labels(report["issues"], known_issues))
    result["installed_versions"] = {
        key: version for key, version in report.get("installed_versions", {}).items()
        if key in _PACKAGES and (version is None or isinstance(version, str)
                                and _VERSION.fullmatch(version))
    }
    # Keep the useful resolver comparison without URLs or arbitrary metadata.
    for field in ("source_requirements", "torch_tpu_requirements"):
        result[field] = []
        for item in report.get(field, [])[:64]:
            if (not isinstance(item, dict) or not isinstance(item.get("package"), str)
                    or item["package"] not in _PACKAGES):
                continue
            required = item.get("required")
            if not isinstance(required, str) or not re.fullmatch(
                r"(?:any|[0-9A-Za-z.!<>=~*+, -]{1,128})", required
            ):
                continue
            state = item.get("status")
            if isinstance(state, str) and state in {"matched", "mismatch", "missing", "direct_url_not_checked"}:
                result[field].append({"package": item["package"],
                                      "required": required, "status": state})
    return result


def lifecycle_evidence(report: dict | None) -> dict:
    if report is None:
        return {"status": "not_run"}
    invalid = {"status": "invalid_report"}
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int \
            or report["schema_version"] != 1:
        return invalid
    requested, completed = report.get("requested_attempts"), report.get("completed_attempts")
    if type(requested) is not int or not 2 <= requested <= 3 \
            or type(completed) is not int or not 1 <= completed <= requested:
        return invalid
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != completed:
        return invalid
    normalized = []
    for index, attempt in enumerate(attempts, 1):
        if (not isinstance(attempt, dict) or type(attempt.get("attempt")) is not int
                or attempt["attempt"] != index):
            return invalid
        status = attempt.get("status")
        if not isinstance(status, str) or status not in {"passed", "failed", "timeout", "interrupted"}:
            return invalid
        discovery = NORMALIZE(attempt.get("report")) if attempt.get("report") else None
        fields = _process_fields(attempt)
        if status == "passed" and (discovery is None or discovery["status"] != "ready"
                                   or fields["fatal_markers"] or fields["exit_code"] != 0):
            return invalid
        normalized.append(dict(fields, attempt=index, status=status, discovery=discovery))
    fingerprints = {json.dumps({key: attempt["discovery"].get(key) for key in (
        "device_name", "topology", "device_count_visible", "torch_version",
        "torch_tpu_version", "libtpu_version")}, sort_keys=True)
        for attempt in normalized if attempt["status"] == "passed"}
    passed = (report.get("status") == "passed" and completed == requested
              and all(attempt["status"] == "passed" for attempt in normalized)
              and len(fingerprints) == 1 and report.get("observations_consistent") is True)
    return {"status": "passed" if passed else "failed", "scope": "process_restart",
            "requested_attempts": requested, "completed_attempts": completed,
            "observations_consistent": passed, "attempts": normalized}


def distributed_evidence(process: dict | None) -> dict:
    if process is None:
        return {"status": "not_run"}
    result = _process_fields(process)
    summary = _json_record(process.get("stdout", ""), "collective_summary=")
    if summary is None:
        return result
    result["fatal_markers"] = sorted(set(result["fatal_markers"]) |
                                    _labels(summary.get("fatal_markers"), _FATAL_LABELS))
    expected = summary.get("expected_world_size")
    ranks, local_ranks = summary.get("ranks"), summary.get("local_ranks")
    valid = (type(expected) is int and 2 <= expected <= 1024
             and type(summary.get("reports_received")) is int
             and summary["reports_received"] == expected
             and all(isinstance(values, list) and len(values) == expected
                     and all(type(rank) is int for rank in values)
                     and sorted(values) == list(range(expected))
                     for values in (ranks, local_ranks)))
    if valid:
        result.update(expected_world_size=expected, ranks=sorted(ranks),
                      local_ranks=sorted(local_ranks))
    passed = (valid and summary.get("status") == "passed"
              and summary.get("worker_exit_code") == 0
              and process.get("status") == "completed" and result["exit_code"] == 0
              and not result["fatal_markers"])
    if passed:
        result["status"] = "passed"
    elif result["status"] == "failed" and summary.get("status") in {"timeout", "interrupted"}:
        result["status"] = summary["status"]
    result["scope"] = "single_host_collective_only"
    return result


def build_evidence(
    *, source_revision: str, notebook_host: str,
    package_process: dict | None = None, eager_process: dict | None = None,
    compile_process: dict | None = None, lifecycle_report: dict | None = None,
    distributed_process: dict | None = None,
) -> dict[str, Any]:
    """Build a new snapshot; callers must not mix results from different runs."""
    if not isinstance(source_revision, str) or not re.fullmatch(
        r"[a-f0-9]{40}|[a-f0-9]{64}", source_revision
    ):
        raise ValueError("source_revision must be a full Git commit SHA")
    return {
        "schema_version": 1, "scope": "t1_feasibility_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision,
        "python_version": platform.python_version(),
        "notebook_host": notebook_host if notebook_host in {"Colab", "Kaggle"} else "other",
        "native_training": "not_integrated", "review": "owner_review_required",
        "checks": {
            "package_profile": profile_evidence(package_process),
            "eager": smoke_evidence(eager_process),
            "compile": smoke_evidence(compile_process, compiled=True),
            "lifecycle": lifecycle_evidence(lifecycle_report),
            "distributed": distributed_evidence(distributed_process),
        },
    }
