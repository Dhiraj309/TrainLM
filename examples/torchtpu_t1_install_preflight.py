#!/usr/bin/env python3
"""Read-only native TorchTPU package-profile check for a TrainLM checkout.

Uses installed distribution metadata and ``pip check``. It never imports
torch/torch_tpu, installs packages, or initializes an accelerator. Passing
metadata checks is not a TPU runtime or TrainLM training certification.
"""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 with the documented dev extra.
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("trainlm", "torch", "torch_tpu", "libtpu", "transformers", "huggingface-hub")
OPTIONAL_PACKAGES = ("tpu-info", "torch-xla", "jax", "jaxlib")


def _version(name: str, provider: object) -> str | None:
    try:
        return str(Version(provider.version(name)))
    except metadata.PackageNotFoundError:
        return None
    except InvalidVersion:
        return None


def _requirements(
    entries: list[str], versions: dict[str, str | None], provider: object
) -> list[dict]:
    checks = []
    for entry in entries:
        try:
            requirement = Requirement(entry)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            name = canonicalize_name(requirement.name)
            installed = versions.setdefault(name, _version(name, provider))
            if requirement.url:
                status = "direct_url_not_checked"
            elif installed is None:
                status = "missing"
            else:
                status = "matched" if Version(installed) in requirement.specifier else "mismatch"
            checks.append({
                "package": name,
                "required": str(requirement.specifier) or "any",
                "installed": installed,
                "status": status,
            })
        except Exception:
            # Malformed upstream metadata is reported without echoing its text.
            checks.append({"package": None, "required": None, "installed": None,
                           "status": "invalid_requirement"})
    return checks


def inspect_profile(
    root: Path = ROOT, *, provider: object = metadata,
    pip_runner: object = subprocess.run,
) -> dict:
    """Compare this source tree and TorchTPU metadata against installed wheels."""
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    versions = {canonicalize_name(name): _version(name, provider) for name in PACKAGES}
    optional_versions = {
        canonicalize_name(name): _version(name, provider) for name in OPTIONAL_PACKAGES
    }
    source_requires_python = project.get("requires-python", "")
    python_version = platform.python_version()
    python_compatible = (
        Version(python_version) in SpecifierSet(source_requires_python)
        if source_requires_python else None
    )
    source_checks = _requirements(project.get("dependencies", []), versions, provider)

    upstream_checks = []
    upstream_python_compatible = None
    if versions["torch-tpu"] is not None:
        try:
            upstream = provider.distribution("torch_tpu")
            upstream_checks = _requirements(upstream.requires or [], versions, provider)
            upstream_python = upstream.metadata.get("Requires-Python")
            if upstream_python:
                upstream_python_compatible = (
                    Version(python_version) in SpecifierSet(upstream_python)
                )
        except Exception:
            upstream_checks = [{"package": None, "required": None,
                                "installed": None, "status": "invalid_requirement"}]

    pip_status = "not_run"
    pip_problem_count = None
    pip_conflict_packages = []
    pip_conflict_details = []
    try:
        completed = pip_runner(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=45,
        )
        pip_status = (
            "passed" if completed.returncode == 0
            else "conflicts" if completed.returncode == 1 and completed.stdout.strip()
            else "failed"
        )
        pip_problem_count = (
            sum(bool(line.strip()) for line in completed.stdout.splitlines())
            if completed.returncode == 1 else 0 if completed.returncode == 0 else None
        )
        if pip_status == "conflicts":
            pip_conflict_packages = sorted({
                canonicalize_name(match.group(1))
                for line in completed.stdout.splitlines()
                if (match := re.match(r"^([A-Za-z0-9_.-]+)\s", line))
            })
            for line in completed.stdout.splitlines()[:32]:
                match = re.match(r"^([A-Za-z0-9_.-]+)\s+(\S+) has requirement (.+)\.$", line)
                if not match:
                    continue
                package, version, detail = match.groups()
                missing = detail.endswith(", which is not installed")
                mismatch = re.match(r"^(.+), but you have ([A-Za-z0-9_.-]+)\s+(\S+)$", detail)
                requirement_text = detail[:-len(", which is not installed")] if missing else mismatch.group(1) if mismatch else ""
                if not requirement_text:
                    continue
                try:
                    requirement = Requirement(requirement_text)
                    if not requirement.url:
                        pip_conflict_details.append({
                            "package": canonicalize_name(package),
                            "installed_version": version[:96],
                            "dependency": canonicalize_name(requirement.name),
                            "required": str(requirement.specifier) or "any",
                            "issue": "missing_dependency" if missing else "version_mismatch",
                            "installed_dependency_version": mismatch.group(3)[:96] if mismatch else None,
                        })
                except Exception:
                    continue
    except subprocess.TimeoutExpired:
        pip_status = "timeout"
    except Exception:
        pip_status = "failed"

    checks = source_checks + upstream_checks
    issues = []
    if python_compatible is False:
        issues.append("source_python_mismatch")
    if upstream_python_compatible is False:
        issues.append("torch_tpu_python_mismatch")
    if any(check["status"] in {"mismatch", "invalid_requirement", "direct_url_not_checked"}
           for check in checks):
        issues.append("requirement_conflict")
    if pip_status == "conflicts":
        issues.append("pip_conflicts")
    if any(value is None for value in versions.values()) or any(
        check["status"] == "missing" for check in checks
    ):
        issues.append("missing_distributions")
    if pip_status in {"failed", "timeout"}:
        issues.append("pip_check_failed")
    status = (
        "incompatible" if any(issue in issues for issue in (
            "source_python_mismatch", "torch_tpu_python_mismatch",
            "requirement_conflict", "pip_conflicts"))
        else "check_failed" if "pip_check_failed" in issues
        else "incomplete" if "missing_distributions" in issues
        else "metadata_compatible"
    )
    return {
        "schema_version": 1,
        "status": status,
        "scope": "installed_metadata_only",
        "python_version": python_version,
        "source_python_requirement": source_requires_python,
        "source_python_compatible": python_compatible,
        "torch_tpu_python_compatible": upstream_python_compatible,
        "installed_versions": versions,
        "optional_versions": optional_versions,
        "source_requirements": source_checks,
        "torch_tpu_requirements": upstream_checks,
        "pip_check": {"status": pip_status, "problem_count": pip_problem_count,
                      "affected_packages": pip_conflict_packages,
                      "conflicts": pip_conflict_details},
        "issues": sorted(set(issues)),
    }


def main() -> int:
    report = inspect_profile()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "metadata_compatible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
