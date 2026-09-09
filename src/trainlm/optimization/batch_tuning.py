"""Deterministic production batch, accumulation, and prefetch selection."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class BatchPrefetchGeometry:
    geometry_id: str
    sequence_length: int
    micro_batch_per_device: int
    gradient_accumulation_steps: int
    data_parallel_replicas: int
    prefetch_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_id, str) or not self.geometry_id:
            raise ValueError("geometry_id cannot be empty.")
        for name in (
            "sequence_length",
            "micro_batch_per_device",
            "gradient_accumulation_steps",
            "data_parallel_replicas",
            "prefetch_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def scheduled_tokens_per_update(self) -> int:
        return (
            self.sequence_length
            * self.micro_batch_per_device
            * self.gradient_accumulation_steps
            * self.data_parallel_replicas
        )


@dataclass(frozen=True, slots=True)
class BatchPrefetchMeasurement:
    geometry: BatchPrefetchGeometry
    global_tokens_per_second: float
    peak_hbm_gib: float
    input_idle_fraction: float
    graph_stable: bool
    cpu_fallback_count: int
    real_data: bool

    def __post_init__(self) -> None:
        for name in ("global_tokens_per_second", "peak_hbm_gib"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(self.input_idle_fraction, bool)
            or not isinstance(self.input_idle_fraction, (int, float))
            or not math.isfinite(self.input_idle_fraction)
            or not 0 <= self.input_idle_fraction <= 1
        ):
            raise ValueError("input_idle_fraction must be between zero and one.")
        for name in ("graph_stable", "real_data"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if (
            isinstance(self.cpu_fallback_count, bool)
            or not isinstance(self.cpu_fallback_count, int)
            or self.cpu_fallback_count < 0
        ):
            raise ValueError("cpu_fallback_count must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class BatchPrefetchSelection:
    selected: BatchPrefetchMeasurement
    rejected: tuple[tuple[str, str], ...]
    expected_tokens_per_update: int


def select_batch_prefetch_geometry(
    measurements: tuple[BatchPrefetchMeasurement, ...],
    *,
    expected_tokens_per_update: int,
    maximum_peak_hbm_gib: float,
    maximum_input_idle_fraction: float = 0.05,
) -> BatchPrefetchSelection:
    """Select fastest valid real-data geometry with stable deterministic ties."""

    if (
        isinstance(expected_tokens_per_update, bool)
        or not isinstance(expected_tokens_per_update, int)
        or expected_tokens_per_update < 1
    ):
        raise ValueError("expected_tokens_per_update must be a positive integer.")
    for name, value in (
        ("maximum_peak_hbm_gib", maximum_peak_hbm_gib),
        ("maximum_input_idle_fraction", maximum_input_idle_fraction),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative.")
    if maximum_peak_hbm_gib == 0:
        raise ValueError("maximum_peak_hbm_gib must be greater than zero.")
    if maximum_input_idle_fraction > 1:
        raise ValueError("maximum_input_idle_fraction cannot exceed one.")
    if not measurements:
        raise ValueError("At least one batch/prefetch measurement is required.")
    ids = [item.geometry.geometry_id for item in measurements]
    if len(ids) != len(set(ids)):
        raise ValueError("Batch/prefetch geometry IDs must be unique.")
    eligible = []
    rejected = []
    for item in sorted(measurements, key=lambda value: value.geometry.geometry_id):
        reason = None
        if item.geometry.scheduled_tokens_per_update != expected_tokens_per_update:
            reason = "scheduled token geometry does not match"
        elif not item.real_data:
            reason = "measurement did not use real data"
        elif not item.graph_stable:
            reason = "compiled graph is unstable"
        elif item.cpu_fallback_count:
            reason = "CPU fallback counters were observed"
        elif item.peak_hbm_gib > maximum_peak_hbm_gib:
            reason = "peak HBM exceeds budget"
        elif item.input_idle_fraction > maximum_input_idle_fraction:
            reason = "input idle fraction exceeds budget"
        if reason is None:
            eligible.append(item)
        else:
            rejected.append((item.geometry.geometry_id, reason))
    if not eligible:
        raise ValueError("No batch/prefetch geometry satisfies the production gates.")
    selected = min(
        eligible,
        key=lambda item: (
            -item.global_tokens_per_second,
            item.peak_hbm_gib,
            item.input_idle_fraction,
            item.geometry.geometry_id,
        ),
    )
    return BatchPrefetchSelection(
        selected=selected,
        rejected=tuple(rejected),
        expected_tokens_per_update=expected_tokens_per_update,
    )


__all__ = [
    "BatchPrefetchGeometry",
    "BatchPrefetchMeasurement",
    "BatchPrefetchSelection",
    "select_batch_prefetch_geometry",
]
