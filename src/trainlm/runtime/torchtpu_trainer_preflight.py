"""Classify read-only package metadata before the experimental TPU trainer smoke.

This module has no Torch import. A permitted run is an experiment, not a
declaration that an incompatible dependency profile is supported.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


_VERSION = re.compile(r"[A-Za-z0-9._+\-]{1,96}\Z")
_PACKAGES = ("torch", "torch-tpu", "libtpu", "transformers",
             "huggingface-hub", "pyyaml")
_TRAINING_STACK = set(_PACKAGES) | {"trainlm"}
_KNOWN_ISSUES = {
    "source_python_mismatch", "torch_tpu_python_mismatch",
    "requirement_conflict", "pip_conflicts", "missing_distributions",
    "pip_check_failed",
}


def assess_trainer_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only a narrow, explicit source-Torch mismatch experiment.

    An unrelated, fully parsed pip conflict may coexist with that experiment;
    missing dependencies, upstream mismatches, and unknown conflicts block it.
    """
    issues = profile.get("issues")
    versions = profile.get("installed_versions")
    source_checks = profile.get("source_requirements")
    upstream_checks = profile.get("torch_tpu_requirements")
    pip_check = profile.get("pip_check")
    reasons: list[str] = []
    if (type(profile.get("schema_version")) is not int
            or profile.get("schema_version") != 1
            or profile.get("scope") != "installed_metadata_only"
            or profile.get("status") not in {
                "metadata_compatible", "incompatible", "incomplete", "check_failed",
            }
            or not isinstance(issues, list)
            or any(issue not in _KNOWN_ISSUES for issue in issues)
            or not isinstance(versions, dict)
            or not isinstance(source_checks, list)
            or not isinstance(upstream_checks, list)
            or not isinstance(pip_check, dict)):
        reasons.append("invalid_profile")
        issues = []
        versions = {}
        source_checks = []
        upstream_checks = []
        pip_check = {}

    installed_versions = {}
    for name in _PACKAGES:
        version = versions.get(name)
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            reasons.append("missing_or_invalid_package")
            installed_versions[name] = None
        else:
            installed_versions[name] = version

    source_torch_mismatch = False
    for check in source_checks:
        if not isinstance(check, dict):
            reasons.append("invalid_requirement")
            continue
        name, status = check.get("package"), check.get("status")
        if name == "torch" and status == "mismatch":
            source_torch_mismatch = True
        elif status != "matched":
            reasons.append("source_requirement_conflict")
    if not source_checks:
        reasons.append("missing_source_requirements")
    if not upstream_checks or any(
        not isinstance(check, dict) or check.get("status") != "matched"
        for check in upstream_checks
    ):
        reasons.append("torch_tpu_requirement_conflict")

    if any(issue in issues for issue in (
        "source_python_mismatch", "torch_tpu_python_mismatch",
        "missing_distributions", "pip_check_failed",
    )):
        reasons.append("profile_compatibility_failure")

    pip_status = pip_check.get("status")
    if pip_status == "conflicts":
        conflicts = pip_check.get("conflicts")
        count = pip_check.get("problem_count")
        if (not isinstance(conflicts, list) or type(count) is not int
                or count != len(conflicts)):
            reasons.append("unclassified_pip_conflict")
        else:
            for conflict in conflicts:
                if (not isinstance(conflict, dict)
                        or conflict.get("issue") not in {
                            "missing_dependency", "version_mismatch",
                        }
                        or conflict.get("package") in _TRAINING_STACK
                        or conflict.get("dependency") in _TRAINING_STACK
                        or not isinstance(conflict.get("package"), str)
                        or not isinstance(conflict.get("dependency"), str)):
                    reasons.append("training_stack_pip_conflict")
                    break
    elif pip_status != "passed":
        reasons.append("pip_check_unavailable")
    if "pip_conflicts" in issues and pip_status != "conflicts":
        reasons.append("invalid_profile")
    if "requirement_conflict" in issues and not source_torch_mismatch:
        reasons.append("requirement_conflict")
    if profile.get("status") == "metadata_compatible" and issues:
        reasons.append("invalid_profile")
    if profile.get("status") == "incompatible" and not issues:
        reasons.append("invalid_profile")

    status = ("blocked" if reasons else
              "experimental_mismatch" if source_torch_mismatch or pip_status == "conflicts"
              else "metadata_compatible")
    return {
        "schema_version": 1,
        "scope": "native_trainer_package_preflight",
        "status": status,
        "proceed": status != "blocked",
        "installed_versions": installed_versions,
        "source_torch_mismatch": source_torch_mismatch,
        "unrelated_pip_conflict_count": (
            pip_check.get("problem_count") if pip_status == "conflicts"
            and "training_stack_pip_conflict" not in reasons
            and "unclassified_pip_conflict" not in reasons else 0
        ),
        "reasons": sorted(set(reasons)),
        "limitations": ["metadata_only", "experimental_run_is_not_supported_profile"],
    }
