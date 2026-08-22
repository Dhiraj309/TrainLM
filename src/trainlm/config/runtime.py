"""
Runtime configuration.

The runtime is responsible for executing training independently of the
underlying hardware backend (CPU, CUDA, XLA, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """
    Runtime execution configuration.

    Notes
    -----
    This configuration only describes *how* training is executed.

    It does not contain trainer behaviour such as checkpoint intervals,
    logging cadence, or evaluation scheduling.
    """

    device: Literal["auto", "cpu", "cuda", "xla"] = "auto"

    precision: Literal[
        "fp32",
        "fp16",
        "bf16",
    ] = "bf16"

    strategy: Literal[
        "replicated",
        "sharded",
    ] = "replicated"

