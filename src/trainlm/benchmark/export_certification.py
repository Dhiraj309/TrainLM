"""Evidence gate for plain-Transformers optimized checkpoint interoperability."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PlainHFExportEvidence:
    transformers_version: str
    trainlm_installed: bool
    canonical_state_dict_only: bool
    tied_aliases_preserved: bool
    logits_max_abs_error: float
    loss_abs_error: float
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transformers_version, str)
            or not self.transformers_version
        ):
            raise ValueError("transformers_version cannot be empty.")
        for name in (
            "trainlm_installed",
            "canonical_state_dict_only",
            "tied_aliases_preserved",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        for name in ("logits_max_abs_error", "loss_abs_error"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("missing_keys", "unexpected_keys"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings.")


@dataclass(frozen=True, slots=True)
class PlainHFExportEvaluation:
    passed: bool
    reasons: tuple[str, ...]
    tolerance: float


def evaluate_plain_hf_export(
    evidence: PlainHFExportEvidence,
    *,
    tolerance: float = 1e-5,
) -> PlainHFExportEvaluation:
    """Require a clean Transformers-only reload with numerical parity."""

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and non-negative.")
    reasons = []
    if evidence.trainlm_installed:
        reasons.append("TrainLM was installed in the interoperability environment")
    if not evidence.canonical_state_dict_only:
        reasons.append("export contains non-canonical internal state-dict keys")
    if not evidence.tied_aliases_preserved:
        reasons.append("tied parameter aliases were not preserved")
    if evidence.missing_keys:
        reasons.append("plain Transformers reload reported missing keys")
    if evidence.unexpected_keys:
        reasons.append("plain Transformers reload reported unexpected keys")
    if evidence.logits_max_abs_error > tolerance:
        reasons.append("reloaded logits exceed numerical tolerance")
    if evidence.loss_abs_error > tolerance:
        reasons.append("reloaded loss exceeds numerical tolerance")
    return PlainHFExportEvaluation(
        passed=not reasons,
        reasons=tuple(reasons),
        tolerance=float(tolerance),
    )


def load_plain_hf_export_evaluation(
    artifact_path: str | Path,
) -> PlainHFExportEvaluation:
    """Load a strict interoperability artifact produced by a clean HF process."""

    if not isinstance(artifact_path, (str, Path)):
        raise TypeError("artifact_path must be a path.")
    path = Path(artifact_path)
    if not path.is_file():
        raise ValueError("artifact_path must reference an existing file.")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid plain-HF export artifact: {exc}") from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("Plain-HF export artifact must contain a JSON object.")
    if set(artifact) != {"schema_version", "evidence", "tolerance"}:
        raise ValueError("Plain-HF export artifact keys must match schema version 1.")
    version = artifact["schema_version"]
    if isinstance(version, bool) or version != 1:
        raise ValueError("Plain-HF export artifact supports schema_version=1 only.")

    evidence = artifact["evidence"]
    expected_evidence = {field.name for field in fields(PlainHFExportEvidence)}
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence:
        raise ValueError("Plain-HF export evidence keys must match the schema.")
    values: dict[str, Any] = dict(evidence)
    for name in ("missing_keys", "unexpected_keys"):
        keys = values[name]
        if isinstance(keys, (str, bytes)) or not isinstance(keys, (list, tuple)):
            raise ValueError(f"{name} must be a JSON array of non-empty strings.")
        values[name] = tuple(keys)
    try:
        loaded_evidence = PlainHFExportEvidence(**values)
        return evaluate_plain_hf_export(
            loaded_evidence,
            tolerance=artifact["tolerance"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid plain-HF export evidence: {exc}") from exc


__all__ = [
    "PlainHFExportEvaluation",
    "PlainHFExportEvidence",
    "evaluate_plain_hf_export",
    "load_plain_hf_export_evaluation",
]
