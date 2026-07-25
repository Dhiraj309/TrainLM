from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
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

        self.model = runtime.prepare_model(model)
        self.runtime = runtime

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.state = TrainerState()
        self.control = TrainerControl()

        self.callback_handler = CallbackHandler(callbacks)

        self._train_iterator: Iterator | None = None

    def train(self) -> TrainerState:
        self.state.is_training = True

        self.model.train()

        self.callback_handler.on_train_begin(
            self.state,
            self.control,
        )

        try:
            while not self._should_stop():
                self.control.reset()

                self.callback_handler.on_step_begin(
                    self.state,
                    self.control,
                )

                self._train_step()

                self.callback_handler.on_step_end(
                    self.state,
                    self.control,
                )

        finally:
            self.state.is_training = False

            self.callback_handler.on_train_end(
                self.state,
                self.control,
            )

        return self.state

    def _should_stop(self) -> bool:
        if self.control.should_stop:
            return True

        return self.state.step >= self.config.trainer.max_steps

    def _next_batch(self):
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_dataloader)

        try:
            return next(self._train_iterator)

        except StopIteration:
            self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)

    def _compute_loss(self, batch) -> torch.Tensor:
        batch = self.runtime.prepare_batch(batch)

        with self.runtime.autocast():
            outputs = self.model(**batch)

        if not hasattr(outputs, "loss"):
            raise ValueError(
                "Model output must define a 'loss' attribute."
            )

        return outputs.loss

    def _train_step(self) -> None:
        batch = self._next_batch()

        self.runtime.zero_grad(self.optimizer)

        loss = self._compute_loss(batch)

        self.runtime.backward(loss)

        self.runtime.clip_gradients(
            self.model.parameters(),
            self.config.trainer.max_grad_norm,
        )

        self.runtime.optimizer_step(
            self.optimizer,
        )

        self.scheduler.step()

        self.runtime.synchronize()

        self.state.step += 1
        self.state.loss = loss.detach().item()
        self.state.learning_rate = self.scheduler.get_last_lr()[0]

    def evaluate(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
