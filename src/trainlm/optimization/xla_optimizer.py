"""Policy and evidence gate for the XLA AdamW state/update path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Literal

GradientReduction = Literal["mean", "sum"]


@dataclass(frozen=True, slots=True)
class XLAAdamWPolicy:
    """Serializable optimizer semantics that must survive compile and resume."""

    schema_version: int = 1
    first_moment_dtype: str = "bfloat16"
    second_moment_dtype: str = "float32"
    gradient_clip_norm: float | None = None
    decoupled_weight_decay: bool = True
    gradient_reduction: GradientReduction = "mean"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("XLAAdamWPolicy supports schema_version=1 only.")
        if self.first_moment_dtype not in {"bfloat16", "float32"}:
            raise ValueError("first_moment_dtype must be bfloat16 or float32.")
        if self.second_moment_dtype != "float32":
            raise ValueError("second_moment_dtype must remain float32.")
        if self.gradient_clip_norm is not None and (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive.")
        if self.decoupled_weight_decay is not True:
            raise ValueError("XLA optimizer path requires decoupled weight decay.")
        if self.gradient_reduction not in {"mean", "sum"}:
            raise ValueError("gradient_reduction must be mean or sum.")

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class XLAOptimizerEvidence:
    update_matches_reference: bool
    resume_matches_reference: bool
    graph_stable: bool
    cpu_fallback_count: int
    step_seconds: float
    reference_step_seconds: float
    peak_hbm_gib: float
    reference_peak_hbm_gib: float

    def __post_init__(self) -> None:
        for name in (
            "update_matches_reference",
            "resume_matches_reference",
            "graph_stable",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if (
            isinstance(self.cpu_fallback_count, bool)
            or not isinstance(self.cpu_fallback_count, int)
            or self.cpu_fallback_count < 0
        ):
            raise ValueError("cpu_fallback_count must be a non-negative integer.")
        for name in (
            "step_seconds",
            "reference_step_seconds",
            "peak_hbm_gib",
            "reference_peak_hbm_gib",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True, slots=True)
class XLAOptimizerEvaluation:
    passed: bool
    reasons: tuple[str, ...]
    step_change_fraction: float
    hbm_change_fraction: float


def evaluate_xla_optimizer_path(
    policy: XLAAdamWPolicy,
    evidence: XLAOptimizerEvidence,
    *,
    maximum_step_regression: float = 0.05,
) -> XLAOptimizerEvaluation:
    """Require correctness, resume, graph, fallback, step, and HBM evidence."""

    if not isinstance(policy, XLAAdamWPolicy):
        raise TypeError("policy must be an XLAAdamWPolicy.")
    if not isinstance(evidence, XLAOptimizerEvidence):
        raise TypeError("evidence must be XLAOptimizerEvidence.")
    if (
        isinstance(maximum_step_regression, bool)
        or not isinstance(maximum_step_regression, (int, float))
        or not math.isfinite(maximum_step_regression)
        or maximum_step_regression < 0
    ):
        raise ValueError("maximum_step_regression must be finite and non-negative.")
    step_change = evidence.step_seconds / evidence.reference_step_seconds - 1.0
    hbm_change = evidence.peak_hbm_gib / evidence.reference_peak_hbm_gib - 1.0
    reasons = []
    if not evidence.update_matches_reference:
        reasons.append("optimizer update does not match the reference")
    if not evidence.resume_matches_reference:
        reasons.append("optimizer resume does not match the reference")
    if not evidence.graph_stable:
        reasons.append("optimizer graph is unstable")
    if evidence.cpu_fallback_count:
        reasons.append("optimizer path contains CPU fallbacks")
    if step_change > maximum_step_regression:
        reasons.append("optimizer step-time regression exceeds budget")
    if hbm_change >= 0:
        reasons.append("optimizer peak HBM is not reduced")
    return XLAOptimizerEvaluation(
        passed=not reasons,
        reasons=tuple(reasons),
        step_change_fraction=step_change,
        hbm_change_fraction=hbm_change,
    )


__all__ = [
    "GradientReduction",
    "XLAAdamWPolicy",
    "XLAOptimizerEvaluation",
    "XLAOptimizerEvidence",
    "evaluate_xla_optimizer_path",
]
