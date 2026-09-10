"""Versioned compatibility contract for TrainLM's public package surface."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PublicAPIContract:
    """Published names and configuration compatibility for one API version."""

    schema_version: int
    api_version: str
    public_symbols: tuple[str, ...]
    config_keys: tuple[str, ...]
    deprecated_config_keys: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("PublicAPIContract supports schema_version=1 only.")
        if not self.api_version:
            raise ValueError("api_version cannot be empty.")
        for name in ("public_symbols", "config_keys"):
            values = getattr(self, name)
            if not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} entries must be unique.")
        if any(
            not isinstance(old, str)
            or not old
            or not isinstance(new, str)
            or not new
            for old, new in self.deprecated_config_keys.items()
        ):
            raise ValueError("deprecated_config_keys must map non-empty strings.")


@dataclass(frozen=True, slots=True)
class PublicAPICompatibility:
    compatible: bool
    reasons: tuple[str, ...]


def load_public_api_contract(path: str | Path) -> PublicAPIContract:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Public API contract root must be an object.")
    data["public_symbols"] = tuple(data.get("public_symbols", ()))
    data["config_keys"] = tuple(data.get("config_keys", ()))
    return PublicAPIContract(**data)


def evaluate_public_api_compatibility(
    contract: PublicAPIContract,
    *,
    api_version: str,
    public_symbols: tuple[str, ...],
    config_keys: tuple[str, ...],
    deprecated_config_keys: Mapping[str, str],
) -> PublicAPICompatibility:
    """Compare the installed public surface with its versioned contract."""

    reasons: list[str] = []
    if api_version != contract.api_version:
        reasons.append("public API version changed")
    if set(public_symbols) != set(contract.public_symbols):
        reasons.append("public package symbols changed")
    if set(config_keys) != set(contract.config_keys):
        reasons.append("public configuration keys changed")
    if dict(deprecated_config_keys) != dict(contract.deprecated_config_keys):
        reasons.append("deprecated configuration aliases changed")
    return PublicAPICompatibility(compatible=not reasons, reasons=tuple(reasons))


__all__ = [
    "PublicAPICompatibility",
    "PublicAPIContract",
    "evaluate_public_api_compatibility",
    "load_public_api_contract",
]
