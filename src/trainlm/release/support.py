"""Validation and explanation agreement for the dense-AR support manifest."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

_LEVELS = {
    "unverified": 0,
    "compatible": 1,
    "experimental": 1,
    "optimized": 2,
    "certified": 3,
}
_EXPLANATION_LEVELS = {"unverified", "compatible", "optimized", "certified"}


@dataclass(frozen=True, slots=True)
class SupportManifest:
    schema_version: int
    release: str
    package_versions: Mapping[str, str]
    hardware: tuple[Mapping[str, str], ...]
    execution_paths: tuple[Mapping[str, str], ...]
    providers: tuple[Mapping[str, str], ...]
    fallbacks: tuple[str, ...]
    caveats: tuple[str, ...]
    torchtpu: Mapping[str, str]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("SupportManifest supports schema_version=1 only.")
        if not isinstance(self.release, str) or not self.release:
            raise ValueError("release cannot be empty.")
        if not self.package_versions or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in self.package_versions.items()
        ):
            raise ValueError("package_versions must contain non-empty strings.")
        self._validate_records(
            "hardware", self.hardware, "backend", "support_level"
        )
        self._validate_records(
            "execution_paths",
            self.execution_paths,
            "path",
            "maximum_certification",
        )
        self._validate_records(
            "providers", self.providers, "provider", "support_level"
        )
        for name in ("fallbacks", "caveats"):
            values = getattr(self, name)
            if not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        if not isinstance(self.torchtpu, Mapping):
            raise TypeError("torchtpu must be a mapping.")
        if self.torchtpu.get("status") not in {
            "deferred",
            "experimental",
            "supported",
        }:
            raise ValueError("torchtpu status is invalid.")
        if not isinstance(self.torchtpu.get("milestone"), str) or not self.torchtpu.get(
            "milestone"
        ):
            raise ValueError("torchtpu milestone cannot be empty.")

    @staticmethod
    def _validate_records(
        name: str,
        records: tuple[Mapping[str, str], ...],
        identity: str,
        level: str,
    ) -> None:
        if not records:
            raise ValueError(f"{name} cannot be empty.")
        identities = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(f"{name} entries must be mappings.")
            value = record.get(identity)
            support = record.get(level)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} {identity} cannot be empty.")
            if support not in _LEVELS:
                raise ValueError(f"{name} has invalid support level {support!r}.")
            identities.append(value)
        if len(identities) != len(set(identities)):
            raise ValueError(f"{name} identities must be unique.")


@dataclass(frozen=True, slots=True)
class SupportExplanationEvaluation:
    agrees: bool
    reasons: tuple[str, ...]


def load_support_manifest(path: str | Path) -> SupportManifest:
    """Load a strict support manifest without accepting silent schema drift."""

    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a path.")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError("path must reference an existing support manifest.")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid support manifest: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Support manifest root must be an object.")
    expected = {field.name for field in fields(SupportManifest)}
    if set(data) != expected:
        raise ValueError("Support manifest keys must match schema version 1.")
    values: dict[str, Any] = dict(data)
    _validate_json_object("package_versions", values["package_versions"])
    _validate_record_array(
        "hardware",
        values["hardware"],
        {"backend", "hardware", "support_level"},
    )
    _validate_record_array(
        "execution_paths",
        values["execution_paths"],
        {"path", "maximum_certification"},
    )
    _validate_record_array(
        "providers",
        values["providers"],
        {"provider", "support_level"},
    )
    for name in ("hardware", "execution_paths", "providers"):
        values[name] = tuple(values[name])
    for name in ("fallbacks", "caveats"):
        items = values[name]
        if isinstance(items, (str, bytes)) or not isinstance(items, list):
            raise ValueError(f"{name} must be a JSON array.")
        values[name] = tuple(items)
    _validate_json_object("torchtpu", values["torchtpu"])
    if set(values["torchtpu"]) != {"status", "milestone"}:
        raise ValueError("torchtpu keys must match the schema.")
    try:
        return SupportManifest(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid support manifest values: {exc}") from exc


def _validate_json_object(name: str, value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")


def _validate_record_array(
    name: str,
    value: object,
    expected_keys: set[str],
) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array.")
    for index, record in enumerate(value):
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise ValueError(f"{name}[{index}] keys must match the schema.")


def evaluate_explanation_support(
    manifest: SupportManifest,
    explanation: Mapping[str, Any],
) -> SupportExplanationEvaluation:
    """Ensure a public explanation does not exceed published support."""

    if not isinstance(manifest, SupportManifest):
        raise TypeError("manifest must be a SupportManifest.")
    if not isinstance(explanation, Mapping):
        raise TypeError("explanation must be a mapping.")
    reasons: list[str] = []
    backend = explanation.get("backend")
    path = explanation.get("selected_path")
    certification = explanation.get("certification")
    hardware = {item["backend"]: item for item in manifest.hardware}
    paths = {item["path"]: item for item in manifest.execution_paths}
    if backend not in hardware:
        reasons.append(f"backend {backend!r} is absent from the support manifest")
    if path not in paths:
        reasons.append(f"execution path {path!r} is absent from the support manifest")
    if certification not in _EXPLANATION_LEVELS:
        reasons.append(f"certification {certification!r} is invalid")
    elif (
        backend in hardware
        and _LEVELS[certification] > _LEVELS[hardware[backend]["support_level"]]
    ):
        reasons.append("explanation certification exceeds hardware support")
    elif (
        path in paths
        and _LEVELS[certification] > _LEVELS[paths[path]["maximum_certification"]]
    ):
        reasons.append("explanation certification exceeds execution-path support")
    return SupportExplanationEvaluation(agrees=not reasons, reasons=tuple(reasons))


__all__ = [
    "SupportExplanationEvaluation",
    "SupportManifest",
    "evaluate_explanation_support",
    "load_support_manifest",
]
