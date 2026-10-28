"""Evidence gate for the combined optimized loss and attention stage."""

from __future__ import annotations

from dataclasses import dataclass

from .result import BenchmarkResult


@dataclass(frozen=True, slots=True)
class AttentionStageEvaluation:
    passed: bool
    reasons: tuple[str, ...]
    global_tokens_per_second: float
    peak_hbm_gib: float
    reference_peak_hbm_gib: float
    throughput_gate: float


def evaluate_attention_stage(
    optimized: BenchmarkResult,
    reference: BenchmarkResult,
    *,
    minimum_global_tokens_per_second: float = 850_000.0,
) -> AttentionStageEvaluation:
    """Evaluate matched M10 evidence without executing or profiling a run."""

    if minimum_global_tokens_per_second <= 0:
        raise ValueError("minimum_global_tokens_per_second must be positive.")
    reasons: list[str] = []
    if optimized.measurement_kind != "steady_state":
        reasons.append("optimized result must be a steady_state measurement")
    for name, label in (
        ("workload_id", "workload ID"),
        ("workload_version", "workload version"),
        ("accelerator_type", "accelerator type"),
        ("device_count", "device count"),
        ("data_parallel_replicas", "data-parallel replica count"),
        ("scheduled_tokens_per_update", "scheduled tokens per update"),
    ):
        if getattr(optimized, name) != getattr(reference, name):
            reasons.append(f"{label} does not match the reference")
    if optimized.accelerator_type != "v5e-8":
        reasons.append("attention stage certification requires v5e-8")
    if optimized.global_tokens_per_second < minimum_global_tokens_per_second:
        reasons.append("global throughput is below the 850K stage gate")
    if optimized.peak_hbm_gib >= reference.peak_hbm_gib:
        reasons.append("peak HBM is not lower than the matched reference")
    if optimized.unexpected_compile_count:
        reasons.append("unexpected compilation occurred after warmup")
    if optimized.cpu_fallback_count:
        reasons.append("CPU fallback counters were observed")
    metadata = optimized.metadata
    if metadata.get("full_logits_materialized") is not False:
        reasons.append("full-logits elimination is not proven")
    if not isinstance(metadata.get("attention_provider"), str) or not metadata.get(
        "attention_provider"
    ):
        reasons.append("attention provider evidence is missing")
    if not isinstance(metadata.get("loss_provider"), str) or not metadata.get(
        "loss_provider"
    ):
        reasons.append("loss provider evidence is missing")
    if not isinstance(metadata.get("hlo_fingerprint"), str) or not metadata.get(
        "hlo_fingerprint"
    ):
        reasons.append("HLO fingerprint evidence is missing")
    return AttentionStageEvaluation(
        passed=not reasons,
        reasons=tuple(reasons),
        global_tokens_per_second=optimized.global_tokens_per_second,
        peak_hbm_gib=optimized.peak_hbm_gib,
        reference_peak_hbm_gib=reference.peak_hbm_gib,
        throughput_gate=float(minimum_global_tokens_per_second),
    )


__all__ = ["AttentionStageEvaluation", "evaluate_attention_stage"]
