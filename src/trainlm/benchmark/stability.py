"""Evidence gate for the 200-update real-shard stability run."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RealShardStabilityEvidence:
    completed_updates: int
    real_data_revision: str
    shard_count: int
    evaluation_completed: bool
    checkpoint_resumed: bool
    integrity_checks_passed: bool
    data_cursor_continuous: bool
    export_completed: bool
    unexpected_compile_count: int
    cpu_fallback_count: int
    minimum_loss: float
    maximum_loss: float
    minimum_gradient_norm: float
    maximum_gradient_norm: float

    def __post_init__(self) -> None:
        for name in ("completed_updates", "shard_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.real_data_revision, str) or not self.real_data_revision:
            raise ValueError("real_data_revision cannot be empty.")
        for name in (
            "evaluation_completed",
            "checkpoint_resumed",
            "integrity_checks_passed",
            "data_cursor_continuous",
            "export_completed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        for name in ("unexpected_compile_count", "cpu_fallback_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in (
            "minimum_loss",
            "maximum_loss",
            "minimum_gradient_norm",
            "maximum_gradient_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.minimum_loss > self.maximum_loss:
            raise ValueError("minimum_loss cannot exceed maximum_loss.")
        if self.minimum_gradient_norm > self.maximum_gradient_norm:
            raise ValueError(
                "minimum_gradient_norm cannot exceed maximum_gradient_norm."
            )


@dataclass(frozen=True, slots=True)
class RealShardStabilityEvaluation:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_real_shard_stability(
    evidence: RealShardStabilityEvidence,
    *,
    required_updates: int = 200,
    minimum_shards: int = 2,
) -> RealShardStabilityEvaluation:
    """Require lifecycle, integrity, continuity, graph, and export evidence."""

    for name, value in (
        ("required_updates", required_updates),
        ("minimum_shards", minimum_shards),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    reasons = []
    if evidence.completed_updates < required_updates:
        reasons.append(f"run completed fewer than {required_updates} updates")
    if evidence.shard_count < minimum_shards:
        reasons.append("run did not cover diverse real-data shards")
    if not evidence.evaluation_completed:
        reasons.append("scheduled evaluation did not complete")
    if not evidence.checkpoint_resumed:
        reasons.append("checkpoint resume was not exercised")
    if not evidence.integrity_checks_passed:
        reasons.append("integrity checks did not pass")
    if not evidence.data_cursor_continuous:
        reasons.append("data cursor continuity was not proven")
    if evidence.unexpected_compile_count:
        reasons.append("unexpected compilation occurred after warmup")
    if evidence.cpu_fallback_count:
        reasons.append("CPU fallback counters were observed")
    if not evidence.export_completed:
        reasons.append("canonical export did not complete")
    return RealShardStabilityEvaluation(passed=not reasons, reasons=tuple(reasons))


__all__ = [
    "RealShardStabilityEvaluation",
    "RealShardStabilityEvidence",
    "evaluate_real_shard_stability",
]
