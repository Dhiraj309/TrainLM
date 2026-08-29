"""Synchronized, host-boundary telemetry for training measurements."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median


def _non_negative(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class StepTelemetry:
    """One already-synchronized measurement recorded outside the hot path."""

    supervised_tokens: int
    wall_seconds: float
    device_seconds: float
    compile_seconds: float = 0.0
    peak_hbm_gib: float = 0.0
    input_idle_fraction: float = 0.0
    collective_seconds: float = 0.0
    compile_count: int = 0
    cpu_fallback_count: int = 0
    synchronized: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.supervised_tokens, bool)
            or not isinstance(self.supervised_tokens, int)
            or self.supervised_tokens <= 0
        ):
            raise ValueError("supervised_tokens must be a positive integer.")
        for name in (
            "wall_seconds",
            "device_seconds",
            "compile_seconds",
            "peak_hbm_gib",
            "collective_seconds",
        ):
            _non_negative(name, getattr(self, name))
        if self.wall_seconds <= 0 or self.device_seconds <= 0:
            raise ValueError("step timings must be greater than zero.")
        if self.device_seconds > self.wall_seconds:
            raise ValueError("device_seconds cannot exceed wall_seconds.")
        if not 0 <= self.input_idle_fraction <= 1:
            raise ValueError("input_idle_fraction must be between zero and one.")
        for name in ("compile_count", "cpu_fallback_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.synchronized is not True:
            raise ValueError("Telemetry samples must be synchronized.")


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Immutable aggregate suitable for callbacks and benchmark serialization."""

    measured_steps: int
    supervised_tokens: int
    wall_seconds: float
    device_seconds: float
    global_tokens_per_second: float
    device_tokens_per_second: float
    compile_seconds: float
    compile_count: int
    peak_hbm_gib: float
    input_idle_fraction: float
    collective_seconds_median: float
    cpu_fallback_count: int
    synchronized: bool = True


class TelemetryRecorder:
    """Collect explicit synchronized samples without tensor/scalar extraction."""

    def __init__(self) -> None:
        self._steps: list[StepTelemetry] = []

    def record_step(self, sample: StepTelemetry) -> None:
        if not isinstance(sample, StepTelemetry):
            raise TypeError("sample must be StepTelemetry.")
        self._steps.append(sample)

    def snapshot(self) -> TelemetrySnapshot:
        if not self._steps:
            raise RuntimeError("No synchronized telemetry samples recorded.")
        steps = tuple(self._steps)
        wall_seconds = sum(item.wall_seconds for item in steps)
        device_seconds = sum(item.device_seconds for item in steps)
        tokens = sum(item.supervised_tokens for item in steps)
        return TelemetrySnapshot(
            measured_steps=len(steps),
            supervised_tokens=tokens,
            wall_seconds=wall_seconds,
            device_seconds=device_seconds,
            global_tokens_per_second=tokens / wall_seconds,
            device_tokens_per_second=tokens / device_seconds,
            compile_seconds=sum(item.compile_seconds for item in steps),
            compile_count=sum(item.compile_count for item in steps),
            peak_hbm_gib=max(item.peak_hbm_gib for item in steps),
            input_idle_fraction=float(
                median(item.input_idle_fraction for item in steps)
            ),
            collective_seconds_median=float(
                median(item.collective_seconds for item in steps)
            ),
            cpu_fallback_count=sum(item.cpu_fallback_count for item in steps),
        )

    def clear(self) -> None:
        self._steps.clear()
