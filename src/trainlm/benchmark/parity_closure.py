"""Final evidence gate for closing measured HLO and host bottlenecks."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .result import BenchmarkResult


@dataclass(frozen=True, slots=True)
class ParityClosureEvidence:
    result: BenchmarkResult
    hlo_fingerprint: str
    transpose_count: int
    layout_copy_count: int
    host_sync_count: int
    full_logits_materialized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.result, BenchmarkResult):
            raise TypeError("result must be a BenchmarkResult.")
        if not isinstance(self.hlo_fingerprint, str) or not self.hlo_fingerprint:
            raise ValueError("hlo_fingerprint cannot be empty.")
        for name in ("transpose_count", "layout_copy_count", "host_sync_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.full_logits_materialized, bool):
            raise TypeError("full_logits_materialized must be boolean.")


@dataclass(frozen=True, slots=True)
class ParityClosureEvaluation:
    passed: bool
    reasons: tuple[str, ...]
    global_tokens_per_second: float
    non_embedding_mfu: float
    collective_fraction: float


def evaluate_parity_closure(
    evidence: ParityClosureEvidence,
    *,
    minimum_global_tokens_per_second: float = 912_600.0,
    minimum_non_embedding_mfu: float = 0.478,
    maximum_input_idle_fraction: float = 0.05,
    maximum_collective_fraction: float = 0.10,
) -> ParityClosureEvaluation:
    """Require the M11 hard gate and absence of known graph/host bottlenecks."""

    for name, value in (
        ("minimum_global_tokens_per_second", minimum_global_tokens_per_second),
        ("minimum_non_embedding_mfu", minimum_non_embedding_mfu),
        ("maximum_input_idle_fraction", maximum_input_idle_fraction),
        ("maximum_collective_fraction", maximum_collective_fraction),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be non-negative.")
    if maximum_input_idle_fraction > 1 or maximum_collective_fraction > 1:
        raise ValueError("Fraction budgets cannot exceed one.")
    result = evidence.result
    collective_fraction = (
        result.collective_seconds_median / result.total_step_seconds_median
    )
    reasons = []
    if result.measurement_kind != "steady_state":
        reasons.append("result must be a steady_state measurement")
    if result.accelerator_type != "v5e-8":
        reasons.append("parity closure requires v5e-8")
    if result.global_tokens_per_second < minimum_global_tokens_per_second:
        reasons.append("global throughput is below the 912.6K hard gate")
    if result.non_embedding_mfu < minimum_non_embedding_mfu:
        reasons.append("non-embedding MFU is below the 47.8% hard gate")
    if result.unexpected_compile_count:
        reasons.append("unexpected compilation occurred after warmup")
    if result.cpu_fallback_count:
        reasons.append("CPU fallback counters were observed")
    if evidence.transpose_count:
        reasons.append("unresolved transpose operations remain")
    if evidence.layout_copy_count:
        reasons.append("unresolved layout copies remain")
    if evidence.host_sync_count:
        reasons.append("unplanned host synchronization remains")
    if evidence.full_logits_materialized:
        reasons.append("full logits remain materialized")
    if result.input_idle_fraction > maximum_input_idle_fraction:
        reasons.append("input idle fraction exceeds budget")
    if collective_fraction > maximum_collective_fraction:
        reasons.append("collective time fraction exceeds budget")
    return ParityClosureEvaluation(
        passed=not reasons,
        reasons=tuple(reasons),
        global_tokens_per_second=result.global_tokens_per_second,
        non_embedding_mfu=result.non_embedding_mfu,
        collective_fraction=collective_fraction,
    )


__all__ = [
    "ParityClosureEvaluation",
    "ParityClosureEvidence",
    "evaluate_parity_closure",
]
