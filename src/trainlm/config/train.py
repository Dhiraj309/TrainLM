"""
Root training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoint import CheckpointConfig
from .dataset import DatasetConfig
from .evaluation import EvaluationConfig
from .loss import LossConfig
from .logging import LoggingConfig
from .model import ModelSourceConfig
from .monitoring import MonitoringConfig
from .optimization import OptimizationConfig
from .optimizer import OptimizerConfig
from .parallelism import ParallelismConfig
from .runtime import RuntimeConfig
from .scheduler import SchedulerConfig
from .trainer import TrainerConfig

__all__ = ["TrainConfig"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """
    Root configuration for TrainLM training.
    """

    model: ModelSourceConfig = field(default_factory=ModelSourceConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    optimizations: OptimizationConfig = field(default_factory=OptimizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
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
