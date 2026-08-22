"""Training diagnostics collection policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    """Configure metric collection independently from presentation."""

    enabled: bool = True
    compile_metrics: bool = True
    memory_metrics: bool = True
    training_integrity: bool = False
    integrity_interval_steps: int = 100

    def __post_init__(self) -> None:
        if self.integrity_interval_steps < 1:
            raise ValueError(
                "'monitoring.integrity_interval_steps' must be at least 1."
            )

