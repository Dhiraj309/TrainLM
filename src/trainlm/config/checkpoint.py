"""
Checkpoint configuration.

This module defines the checkpointing policy used by the Trainer.

TrainLM distinguishes between two checkpoint types:

- Training checkpoints
    Used for fault tolerance and exact training resumption.

- Inference checkpoints
    Used for evaluation and deployment.

The checkpoint manager is responsible for implementing this policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """
    Checkpoint configuration.
    """

    output_dir: Path = Path("checkpoints")

    save_training_every_steps: int | None = None

    save_training_every_tokens: int | None = None

    save_inference_every_steps: int | None = None

    save_inference_every_tokens: int | None = None

    keep_last: int = 3

    save_optimizer: bool = True

    save_rng_state: bool = True
