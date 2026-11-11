"""Installed package metadata gates the public scratch-training workflow."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 dev extra.
    import tomli as tomllib

import pytest
from packaging.utils import canonicalize_name

from trainlm.runtime.pretraining_profile import (
    PretrainingPackagePreflightError,
    _BASE_REQUIREMENTS,
    _PRETRAINING_REQUIREMENTS,
    inspect_pretraining_profile,
    require_pretraining_profile,
)


class FakeDistribution:
    def __init__(self, version: str, requires: list[str] | None = None):
        self.version = version
        self.requires = requires or []


class FakeProvider:
    def __init__(self, versions: dict[str, str], requirements=None):
        self.versions = {
            canonicalize_name(name): version for name, version in versions.items()
        }
        self.requirements = requirements or {}

    def distribution(self, name):
        normalized = canonicalize_name(name)
        if normalized not in self.versions:
            raise metadata.PackageNotFoundError(name)
        return FakeDistribution(
            self.versions[normalized], self.requirements.get(normalized, []),
        )


def _compatible_provider(*, native_tpu=False):
    versions = {
        "huggingface-hub": "1.33.0",
        "pyyaml": "6.0.3",
        "torch": "2.14.1+cpu",
        "transformers": "5.20.2",
        "datasets": "3.6.0",
        "tokenizers": "0.22.2",
    }
    requirements = {}
    if native_tpu:
        versions.update({"torch-tpu": "0.2.0", "libtpu": "0.0.50"})
        requirements["torch-tpu"] = [
            "torch>=2.11,<2.16",
            "libtpu>=0.0.50; sys_platform == 'linux'",
            "unused-extra>=1; extra == 'cuda'",
        ]
    return FakeProvider(versions, requirements)


def test_requirements_match_project_metadata_and_pretraining_extra():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert tuple(project["dependencies"]) == _BASE_REQUIREMENTS
    assert tuple(project["optional-dependencies"]["pretraining"]) == (
        _PRETRAINING_REQUIREMENTS
    )


def test_base_pretraining_profile_is_read_only_and_metadata_compatible():
    profile = inspect_pretraining_profile(
        accelerator="cpu", provider=_compatible_provider(),
    )

    assert profile["status"] == "metadata_compatible"
    assert profile["scope"] == "installed_metadata_only"
    assert {item["status"] for item in profile["requirements"]} == {"matched"}
    assert all(
        "installed" in item and "package" in item
        for item in profile["requirements"]
    )
    assert "does_not_certify_accelerator_execution" in profile["limitations"]


def test_missing_or_incompatible_pretraining_packages_are_reported():
    provider = _compatible_provider()
    provider.versions["torch"] = "2.15.0"
    del provider.versions["datasets"]

    profile = inspect_pretraining_profile(accelerator="cpu", provider=provider)
    statuses = {item["package"]: item["status"] for item in profile["requirements"]}

    assert profile["status"] == "incompatible"
    assert statuses["torch"] == "mismatch"
    assert statuses["datasets"] == "missing"


def test_native_tpu_profile_checks_upstream_requirements_and_ignores_other_extras():
    profile = inspect_pretraining_profile(
        accelerator="native_tpu", provider=_compatible_provider(native_tpu=True),
    )

    assert profile["status"] == "metadata_compatible"
    packages = {item["package"] for item in profile["requirements"]}
    assert {"torch-tpu", "torch", "libtpu"} <= packages
    assert "unused-extra" not in packages


def test_native_tpu_profile_requires_torch_tpu_distribution():
    profile = inspect_pretraining_profile(
        accelerator="native_tpu", provider=_compatible_provider(),
    )

    assert profile["status"] == "incomplete"
    assert any(
        item["package"] == "torch-tpu" and item["status"] == "missing"
        for item in profile["requirements"]
    )


def test_native_tpu_profile_rejects_torch_outside_upstream_range():
    provider = _compatible_provider(native_tpu=True)
    provider.versions["torch"] = "2.10.0"

    profile = inspect_pretraining_profile(accelerator="native_tpu", provider=provider)
    torch_statuses = [
        item["status"] for item in profile["requirements"]
        if item["package"] == "torch"
    ]

    assert profile["status"] == "incompatible"
    assert torch_statuses == ["matched", "mismatch"]


def test_require_profile_raises_with_machine_readable_report_and_no_install():
    provider = _compatible_provider()
    provider.versions["torch"] = "2.15.0"

    with pytest.raises(PretrainingPackagePreflightError) as raised:
        require_pretraining_profile(accelerator="cpu", provider=provider)

    assert raised.value.profile["status"] == "incompatible"
    assert "torch requires <2.15,>=2.9" in str(raised.value)
    assert "did not install or change packages" in str(raised.value)
