from __future__ import annotations

from collections.abc import Iterator, Sequence
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from trainlm.config import TrainConfig
from trainlm.runtime import ExecutionBackend
from trainlm.tasks import (
    LanguageModelTask,
    LossTaskAdapter,
    TaskResult,
)

from .callback import TrainerCallback
from .callback_handler import CallbackHandler
from .control import TrainerControl
from .loss import Loss
from .state import TrainerState


class Trainer:
    """Coordinates the end-to-end training lifecycle."""

    def __init__(
        self,
        *,
        config: TrainConfig,
        model: nn.Module,
        runtime: ExecutionBackend,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataloader: DataLoader,
        task: LanguageModelTask | None = None,
        loss_fn: Loss | None = None,
        eval_dataloader: DataLoader | None = None,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        self.config = config

        self.runtime = runtime
        self.model = runtime.prepare_model(model)

        self.optimizer = runtime.prepare_optimizer(optimizer)
        self.scheduler = scheduler
        if task is not None and loss_fn is not None:
            raise ValueError("Set either 'task' or legacy 'loss_fn', not both.")
        if task is None:
            if loss_fn is None:
                raise ValueError("Trainer requires a language-model 'task'.")
            task = LossTaskAdapter(loss_fn)
        self.task = task

        self.train_dataloader = runtime.prepare_dataloader(train_dataloader)
        self.eval_dataloader = (
            runtime.prepare_dataloader(eval_dataloader)
            if eval_dataloader is not None
            else None
        )

        self.state = TrainerState()
        self.control = TrainerControl()

        self.callback_handler = CallbackHandler(callbacks)

        self._train_iterator: Iterator | None = None

    def train(self) -> TrainerState:
        self.runtime.initialize()
        self.state.is_training = True

        try:
            self.model.train()

            self.runtime.on_train_begin()

            self.callback_handler.on_train_begin(
                self.state,
                self.control,
            )

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

            try:
                self.callback_handler.on_train_end(
                    self.state,
                    self.control,
                )
            finally:
                try:
                    self.runtime.on_train_end()
                finally:
                    self.runtime.finalize()

        return self.state

    def _should_stop(self) -> bool:
        if self.control.should_stop:
            return True

        trainer = self.config.trainer
        if trainer.max_steps is not None and self.state.step >= trainer.max_steps:
            return True
        if (
            trainer.max_tokens is not None
            and self.state.tokens_seen >= trainer.max_tokens
        ):
            return True
        return False

    def _next_batch(self):
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_dataloader)

        try:
            return next(self._train_iterator)

        except StopIteration:
            self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)

    def _current_learning_rate(self) -> float:
        """Return the current learning rate."""

        return self.scheduler.get_last_lr()[0]

    def _update_state(
        self,
        *,
        result: TaskResult,
    ) -> None:
        """Update trainer state after a completed optimization step."""

        self.state.step += 1
        self.state.tokens_seen += result.tokens.supervised_tokens
        self.state.samples_seen += result.tokens.sequences
        self.state.loss = result.loss.detach().item()
        self.state.learning_rate = self._current_learning_rate()

    def _train_step(self) -> None:
        batch = self._next_batch()

        self.runtime.on_step_begin(self.state.step)

        self.runtime.zero_grad(self.optimizer)

        result = self.task.training_step(
            self.model,
            batch,
            self.runtime,
        )

        self.runtime.backward(result.loss)

        self.runtime.clip_gradients(
            self.model.parameters(),
            self.config.trainer.max_grad_norm,
        )

        self.runtime.optimizer_step(
            self.optimizer,
        )

        self.scheduler.step()

        self.runtime.synchronize()

        self._update_state(
            result=result,
        )

        self.runtime.on_step_end(self.state.step)

    def _evaluation_step(self, batch) -> TaskResult:
        """Dispatch one evaluation batch through the selected task."""

        return self.task.evaluation_step(
            self.model,
            batch,
            self.runtime,
        )

    def evaluate(self) -> dict[str, float]:
        """Run evaluation over the evaluation dataloader."""

        if self.eval_dataloader is None:
            raise RuntimeError(
                "Evaluation requested but no evaluation dataloader is configured."
            )

        was_training = self.model.training
        self.model.eval()

        results: list[TaskResult] = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                result = self._evaluation_step(batch)
                results.append(result)

        if was_training:
            self.model.train()

        return self.task.aggregate_evaluation(results)

    def save_model(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
