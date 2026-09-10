"""Evidence gate for plain-Transformers optimized checkpoint interoperability."""

from __future__ import annotations

from dataclasses import dataclass
import math


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
        if not isinstance(self.transformers_version, str) or not self.transformers_version:
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


__all__ = [
    "PlainHFExportEvaluation",
    "PlainHFExportEvidence",
    "evaluate_plain_hf_export",
]
