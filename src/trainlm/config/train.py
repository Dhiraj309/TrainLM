"""
Root training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoint import CheckpointConfig
from .configuration_trainlm import TrainLMConfig
from .dataset import DatasetConfig
from .evaluation import EvaluationConfig
from .logging import LoggingConfig
from .optimizer import OptimizerConfig
from .runtime import RuntimeConfig
from .scheduler import SchedulerConfig
from .trainer import TrainerConfig

__all__ = ["TrainConfig"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """
    Root configuration for TrainLM training.
    """

    model: TrainLMConfig = field(default_factory=TrainLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """
        Validate cross-domain configuration invariants.

        Raises
        ------
        ValueError
            If the configuration is internally inconsistent.
        """

        from .validation import validate_config

        validate_config(self)
