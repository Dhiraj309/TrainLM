from __future__ import annotations

from collections.abc import Sequence

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from trainlm.config import TrainConfig
from trainlm.runtime import Runtime

from .callback import TrainerCallback
from .callback_handler import CallbackHandler
from .control import TrainerControl
from .state import TrainerState


class Trainer:
    """Coordinates the end-to-end training lifecycle."""

    def __init__(
        self,
        *,
        config: TrainConfig,
        model: nn.Module,
        runtime: Runtime,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        self.config = config

        self.model = model
        self.runtime = runtime

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.state = TrainerState()
        self.control = TrainerControl()

        self.callback_handler = CallbackHandler(callbacks)

    def train(self) -> TrainerState:
        raise NotImplementedError

    def _train_step(self) -> None:
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
