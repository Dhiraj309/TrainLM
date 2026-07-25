"""
Dataset configuration.

This module defines the configuration required to construct datasets and
data loaders for TrainLM.

The configuration describes *what* data should be loaded and *how* it
should be presented to the trainer. Dataset implementations remain
separate from the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """
    A single dataset source.

    Parameters
    ----------
    path:
        Path to the dataset.

    weight:
        Sampling weight used when mixing multiple datasets.
    """

    path: Path

    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """
    Dataset pipeline configuration.
    """

    sources: list[DatasetSource] = field(default_factory=list)

    format: Literal[
        "bin",
    ] = "bin"

    sequence_length: int = 2048

    shuffle: bool = True

    drop_last: bool = True

    num_workers: int = 4

    pin_memory: bool = True

    prefetch_factor: int = 2

    persistent_workers: bool = True

    packing: bool = False
