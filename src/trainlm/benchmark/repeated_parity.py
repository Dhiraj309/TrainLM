"""Three-run v5e parity certification summary and hard-gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from .result import BenchmarkResult


@dataclass(frozen=True, slots=True)
class RepeatedParityEvaluation:
    passed: bool
    preferred_passed: bool
    reasons: tuple[str, ...]
    median_global_tokens_per_second: float
    median_non_embedding_mfu: float
    throughput_relative_spread: float
    peak_hbm_gib_max: float
    hlo_fingerprint: str | None


def evaluate_repeated_parity(
    results: tuple[BenchmarkResult, ...],
    *,
    hard_throughput: float = 912_600.0,
    hard_mfu: float = 0.478,
    preferred_throughput: float = 963_300.0,
    preferred_mfu: float = 0.504,
) -> RepeatedParityEvaluation:
    """Require three matched runs and summarize reproducibility evidence."""

    if len(results) != 3:
        raise ValueError("Repeated parity certification requires exactly three runs.")
    for name, value in (
        ("hard_throughput", hard_throughput),
        ("hard_mfu", hard_mfu),
        ("preferred_throughput", preferred_throughput),
        ("preferred_mfu", preferred_mfu),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")
    reference = results[0]
    matched_fields = (
        "workload_id",
        "workload_version",
        "accelerator_type",
        "device_count",
        "data_parallel_replicas",
        "scheduled_tokens_per_update",
    )
    reasons = []
    for index, result in enumerate(results, start=1):
        if any(getattr(result, name) != getattr(reference, name) for name in matched_fields):
            reasons.append(f"run {index} does not match the reference run geometry")
        if result.measurement_kind != "steady_state":
            reasons.append(f"run {index} is not a steady_state measurement")
        if result.cache_state != "warm":
            reasons.append(f"run {index} does not use a warm compilation cache")
        if result.accelerator_type != "v5e-8":
            reasons.append(f"run {index} does not use v5e-8")
        if result.global_tokens_per_second < hard_throughput:
            reasons.append(f"run {index} is below the hard throughput gate")
        if result.non_embedding_mfu < hard_mfu:
            reasons.append(f"run {index} is below the hard MFU gate")
        if result.unexpected_compile_count:
            reasons.append(f"run {index} recompiled after warmup")
        if result.cpu_fallback_count:
            reasons.append(f"run {index} contains CPU fallbacks")
    fingerprints = tuple(result.metadata.get("hlo_fingerprint") for result in results)
    if any(not isinstance(value, str) or not value for value in fingerprints):
        reasons.append("one or more runs are missing an HLO fingerprint")
        hlo_fingerprint = None
    elif len(set(fingerprints)) != 1:
        reasons.append("HLO fingerprints differ across repeated runs")
        hlo_fingerprint = None
    else:
        hlo_fingerprint = fingerprints[0]
    throughputs = tuple(result.global_tokens_per_second for result in results)
    mfus = tuple(result.non_embedding_mfu for result in results)
    median_throughput = statistics.median(throughputs)
    median_mfu = statistics.median(mfus)
    relative_spread = (max(throughputs) - min(throughputs)) / median_throughput
    return RepeatedParityEvaluation(
        passed=not reasons,
        preferred_passed=(
            not reasons
            and median_throughput >= preferred_throughput
            and median_mfu >= preferred_mfu
        ),
        reasons=tuple(reasons),
        median_global_tokens_per_second=median_throughput,
        median_non_embedding_mfu=median_mfu,
        throughput_relative_spread=relative_spread,
        peak_hbm_gib_max=max(result.peak_hbm_gib for result in results),
        hlo_fingerprint=hlo_fingerprint,
    )


def load_repeated_parity_evaluation(
    manifest_path: str | Path,
) -> RepeatedParityEvaluation:
    """Load exactly three benchmark artifacts from a confined manifest."""

    if not isinstance(manifest_path, (str, Path)):
        raise TypeError("manifest_path must be a path.")
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("manifest_path must reference an existing file.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid repeated-parity manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Repeated-parity manifest must contain a JSON object.")
    if set(manifest) != {"schema_version", "result_paths", "thresholds"}:
        raise ValueError("Repeated-parity manifest keys must match schema version 1.")
    if manifest["schema_version"] != 1:
        raise ValueError("Repeated-parity manifest supports schema_version=1 only.")
    result_paths = manifest["result_paths"]
    if (
        isinstance(result_paths, (str, bytes))
        or not isinstance(result_paths, (list, tuple))
        or len(result_paths) != 3
        or any(not isinstance(value, str) or not value for value in result_paths)
    ):
        raise ValueError("result_paths must contain exactly three non-empty paths.")
    if len(set(result_paths)) != 3:
        raise ValueError("result_paths must identify three distinct artifacts.")
    thresholds = manifest["thresholds"]
    expected_thresholds = {
        "hard_throughput",
        "hard_mfu",
        "preferred_throughput",
        "preferred_mfu",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != expected_thresholds:
        raise ValueError("thresholds must match the versioned parity gate.")
    root = path.resolve().parent
    results = []
    for relative in result_paths:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("Benchmark result paths must be relative to the manifest.")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("Benchmark result path escapes the manifest or is missing.")
        try:
            results.append(BenchmarkResult.from_json(resolved.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid benchmark result {relative!r}: {exc}") from exc
    return evaluate_repeated_parity(tuple(results), **dict(thresholds))


__all__ = [
    "RepeatedParityEvaluation",
    "evaluate_repeated_parity",
    "load_repeated_parity_evaluation",
]
