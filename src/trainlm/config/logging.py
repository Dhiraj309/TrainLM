"""
Logging configuration.

This module defines the logging policy for a training run.

Logging is responsible for presenting and exporting metrics collected
during training. Metric collection itself is handled independently by
the MetricsStore.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """
    Logging configuration.
    """

    log_every_steps: int = 10

    console: bool = True

    jsonl: bool = True

    tensorboard: bool = False

    wandb: bool = False

    output_dir: Path = Path("runs")
