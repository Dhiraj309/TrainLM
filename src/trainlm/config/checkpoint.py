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

    def __post_init__(self) -> None:
        for name in (
            "save_training_every_steps",
            "save_training_every_tokens",
            "save_inference_every_steps",
            "save_inference_every_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"'{name}' must be positive when configured.")
        if (
            isinstance(self.keep_last, bool)
            or not isinstance(self.keep_last, int)
            or self.keep_last < 1
        ):
            raise ValueError("'keep_last' must be positive.")
        if self.save_optimizer is not True or self.save_rng_state is not True:
            raise ValueError(
                "Exact training checkpoints require optimizer and RNG state; "
                "use a Hugging Face export for model-only persistence."
            )
