"""Read-only dependency checks for the public from-scratch workflow.

The profile verifies installed distribution metadata only. A compatible profile
does not certify that an accelerator can be allocated or that training works on
that hardware.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


# Keep these in sync with pyproject.toml. A contract test checks both the base
# project requirements and the optional pretraining extra.
_BASE_REQUIREMENTS = (
    "huggingface-hub>=1.5,<2",
    "PyYAML>=6,<7",
    "torch>=2.9,<2.15",
    "transformers>=5.15,<6",
)
_PRETRAINING_REQUIREMENTS = (
    "datasets>=3,<5",
    "tokenizers>=0.21,<1",
)


class PretrainingPackagePreflightError(RuntimeError):
    """A scratch run whose installed package metadata is incomplete/invalid."""

    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = profile
        failures = [
            item for item in profile["requirements"]
            if item["status"] != "matched"
        ]
        details = []
        for item in failures:
            package = item.get("package") or "TorchTPU requirement metadata"
            required = item.get("required") or "valid metadata"
            installed = item.get("installed") or "not installed"
            details.append(f"{package} requires {required}; installed: {installed}")
        message = "; ".join(details) or "package metadata could not be verified"
        super().__init__(
            "Scratch pretraining package preflight failed: " + message + ". "
            "Use a runtime with compatible TrainLM pretraining dependencies. "
            "This check did not install or change packages and does not certify "
            "accelerator execution."
        )


def _version(distribution: Any) -> str | None:
    try:
        return str(Version(distribution.version))
    except (InvalidVersion, AttributeError, TypeError):
        return None


def _check_requirement(
    raw: str, provider: Any,
) -> dict[str, str | None] | None:
    try:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            return None
        package = canonicalize_name(requirement.name)
    except (InvalidRequirement, ValueError, TypeError):
        return {"package": None, "required": None, "installed": None,
                "status": "invalid_requirement"}

    required = str(requirement.specifier) or "any"
    if requirement.url:
        return {"package": package, "required": "direct URL", "installed": None,
                "status": "direct_url_not_checked"}
    try:
        distribution = provider.distribution(package)
    except metadata.PackageNotFoundError:
        return {"package": package, "required": required, "installed": None,
                "status": "missing"}
    except Exception:
        return {"package": package, "required": required, "installed": None,
                "status": "metadata_unavailable"}
    installed = _version(distribution)
    if installed is None:
        status = "invalid_version"
    else:
        status = (
            "matched" if Version(installed) in requirement.specifier else "mismatch"
        )
    return {"package": package, "required": required, "installed": installed,
            "status": status}


def inspect_pretraining_profile(
    *, accelerator: str, provider: Any | None = None,
) -> dict[str, Any]:
    """Check the package metadata needed by a scratch run without importing it."""
    metadata_provider = metadata if provider is None else provider
    checks = [
        result
        for raw in (*_BASE_REQUIREMENTS, *_PRETRAINING_REQUIREMENTS)
        if (result := _check_requirement(raw, metadata_provider)) is not None
    ]

    if accelerator == "native_tpu":
        try:
            torch_tpu = metadata_provider.distribution("torch-tpu")
        except metadata.PackageNotFoundError:
            checks.append({"package": "torch-tpu", "required": "installed",
                           "installed": None, "status": "missing"})
        except Exception:
            checks.append({"package": "torch-tpu", "required": "installed",
                           "installed": None, "status": "metadata_unavailable"})
        else:
            torch_tpu_version = _version(torch_tpu)
            checks.append({
                "package": "torch-tpu", "required": "installed",
                "installed": torch_tpu_version,
                "status": (
                    "matched" if torch_tpu_version is not None else "invalid_version"
                ),
            })
            for raw in torch_tpu.requires or ():
                result = _check_requirement(raw, metadata_provider)
                if result is not None:
                    checks.append(result)

    statuses = {item["status"] for item in checks}
    status = (
        "incompatible" if statuses & {
            "mismatch", "invalid_requirement", "invalid_version",
            "direct_url_not_checked", "metadata_unavailable",
        }
        else "incomplete" if "missing" in statuses
        else "metadata_compatible"
    )
    return {
        "schema_version": 1,
        "scope": "installed_metadata_only",
        "status": status,
        "accelerator": accelerator,
        "requirements": checks,
        "limitations": ["metadata_only", "does_not_certify_accelerator_execution"],
    }


def require_pretraining_profile(
    *, accelerator: str, provider: Any | None = None,
) -> dict[str, Any]:
    """Raise an actionable error unless the required package metadata matches."""
    profile = inspect_pretraining_profile(accelerator=accelerator, provider=provider)
    if profile["status"] != "metadata_compatible":
        raise PretrainingPackagePreflightError(profile)
    return profile
