"""Deterministic semantic comparison for the locked 135M parity workload."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

REQUIRED_NUMERICAL_PATHS = (
    "architecture.position_encoding",
    "architecture.normalization",
    "architecture.normalization_placement",
    "architecture.residual_layout",
    "initialization.method",
    "initialization.parameter_std",
    "initialization.embedding_std",
    "initialization.attention_std",
    "initialization.mlp_std",
    "initialization.residual_scale",
    "loss.label_shift",
    "loss.z_loss",
    "loss.ignore_index",
    "loss.reduction",
    "optimizer.type",
    "optimizer.learning_rate",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.epsilon",
    "optimizer.weight_decay",
    "optimizer.gradient_clip_norm",
    "optimizer.first_moment_dtype",
    "optimizer.second_moment_dtype",
    "scheduler.type",
    "scheduler.horizon_tokens",
    "scheduler.warmup_fraction",
    "scheduler.stable_fraction",
    "scheduler.minimum_learning_rate_ratio",
    "precision.parameter_dtype",
    "precision.compute_dtype",
    "precision.output_dtype",
)


@dataclass(frozen=True, slots=True)
class NumericalDifference:
    path: str
    reference: Any
    candidate: Any
    status: str
    justification: str | None = None


@dataclass(frozen=True, slots=True)
class NumericalAlignmentReport:
    aligned: bool
    differences: tuple[NumericalDifference, ...]
    deterministic_update_max_abs_errors: tuple[float, ...]
    update_tolerance: float


def compare_numerical_alignment(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    justifications: Mapping[str, str] | None = None,
    deterministic_update_max_abs_errors: tuple[float, ...] = (),
    update_tolerance: float = 1e-6,
) -> NumericalAlignmentReport:
    """Compare required semantics and early deterministic update evidence."""

    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("reference and candidate must be mappings.")
    justifications = justifications or {}
    if any(
        path not in REQUIRED_NUMERICAL_PATHS
        or not isinstance(reason, str)
        or not reason.strip()
        for path, reason in justifications.items()
    ):
        raise ValueError("Justifications require a known path and non-empty reason.")
    if (
        isinstance(update_tolerance, bool)
        or not isinstance(update_tolerance, (int, float))
        or not math.isfinite(update_tolerance)
        or update_tolerance < 0
    ):
        raise ValueError("update_tolerance must be finite and non-negative.")
    if any(
        isinstance(error, bool)
        or not isinstance(error, (int, float))
        or not math.isfinite(error)
        or error < 0
        for error in deterministic_update_max_abs_errors
    ):
        raise ValueError("Deterministic update errors must be finite and non-negative.")

    differences = []
    for path in REQUIRED_NUMERICAL_PATHS:
        reference_value = _lookup(reference, path)
        candidate_value = _lookup(candidate, path)
        if _equal(reference_value, candidate_value):
            continue
        justification = justifications.get(path)
        differences.append(
            NumericalDifference(
                path=path,
                reference=reference_value,
                candidate=candidate_value,
                status="justified" if justification else "mismatch",
                justification=justification,
            )
        )
    updates_aligned = bool(deterministic_update_max_abs_errors) and all(
        error <= update_tolerance for error in deterministic_update_max_abs_errors
    )
    semantics_aligned = all(item.status == "justified" for item in differences)
    return NumericalAlignmentReport(
        aligned=semantics_aligned and updates_aligned,
        differences=tuple(differences),
        deterministic_update_max_abs_errors=deterministic_update_max_abs_errors,
        update_tolerance=float(update_tolerance),
    )


def _lookup(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return "<missing>"
        current = current[part]
    return current


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0.0)
    return left == right


__all__ = [
    "NumericalAlignmentReport",
    "NumericalDifference",
    "REQUIRED_NUMERICAL_PATHS",
    "compare_numerical_alignment",
]
