from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
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
from .state import TrainerPhase, TrainerState


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
        checkpoint_saver: Callable[["Trainer", object | None], object] | None = None,
        checkpoint_loader: Callable[["Trainer", object], object] | None = None,
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

        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader

        self._train_iterator: Iterator | None = None

    def prepare(self) -> TrainerState:
        """Initialize backend resources and enter the prepared phase."""

        if self.state.phase == TrainerPhase.PREPARED:
            return self.state
        if self.state.phase != TrainerPhase.CREATED:
            raise RuntimeError(
                f"Cannot prepare trainer in phase '{self.state.phase.value}'."
            )
        try:
            configure_shapes = getattr(
                self.runtime,
                "configure_static_shapes",
                None,
            )
            if callable(configure_shapes):
                accumulation_steps = getattr(
                    self.config.trainer,
                    "gradient_accumulation_steps",
                    1,
                )
                configure_shapes(
                    accumulation_steps=accumulation_steps,
                )
            self.runtime.initialize()
            self.state.transition(TrainerPhase.PREPARED)
        except BaseException as exc:
            self.state.mark_failed(exc)
            try:
                self.runtime.finalize()
            finally:
                raise
        return self.state

    def train(self) -> TrainerState:
        self.prepare()
        self.state.transition(TrainerPhase.TRAINING)
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
                self._emit_step_metrics()

        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            self._finish_training()

        return self.state

    def _finish_training(self) -> None:
        """Run end hooks and finalize resources exactly once."""

        self.state.is_training = False
        self.state.should_stop = self.state.should_stop or self.control.should_stop
        if self.state.phase != TrainerPhase.FAILED:
            self.state.transition(TrainerPhase.STOPPING)
        try:
            self.callback_handler.on_train_end(
                self.state,
                self.control,
            )
        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            try:
                self.runtime.on_train_end()
            except BaseException as exc:
                self.state.mark_failed(exc)
                raise
            finally:
                try:
                    self.runtime.finalize()
                except BaseException as exc:
                    self.state.mark_failed(exc)
                    raise
                finally:
                    if self.state.phase == TrainerPhase.FAILED:
                        self.state.transition(TrainerPhase.STOPPING)
                    if self.state.phase == TrainerPhase.STOPPING:
                        self.state.transition(TrainerPhase.FINALIZED)

    def _should_stop(self) -> bool:
        if self.control.should_stop or self.state.should_stop:
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

    def _emit_step_metrics(self) -> None:
        """Emit a sparse host-only snapshot according to logging policy."""

        logging_config = getattr(self.config, "logging", None)
        interval = getattr(logging_config, "log_every_steps", None)
        if interval is None:
            return
        if (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval < 1
        ):
            raise ValueError("logging.log_every_steps must be positive.")
        if self.state.step % interval != 0:
            return
        metrics: dict[str, float] = {
            "step": float(self.state.step),
            "tokens_seen": float(self.state.tokens_seen),
            "samples_seen": float(self.state.samples_seen),
            "learning_rate": self.state.learning_rate,
        }
        if self.state.loss is not None:
            metrics["loss"] = self.state.loss
        self.callback_handler.on_metrics(self.state, self.control, metrics)

    def request_stop(self) -> None:
        """Request a graceful stop at the next completed-step boundary."""

        self.control.request_stop()
        self.state.should_stop = True

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

    def _update_accumulated_state(
        self,
        *,
        total_tokens: int,
        total_sequences: int,
        loss_numerator: torch.Tensor,
        exact_tokens: bool,
    ) -> None:
        """Commit one optimizer update after all microbatches are reduced."""

        self.state.step += 1
        self.state.tokens_seen += total_tokens
        self.state.samples_seen += total_sequences
        self.state.global_batch_size = total_sequences
        loss_value = loss_numerator.detach().item()
        self.state.loss = (
            loss_value / total_tokens if exact_tokens else loss_value
        )
        self.state.learning_rate = self._current_learning_rate()

    def _train_step(self) -> None:
        self.runtime.on_step_begin(self.state.step)
        self.runtime.zero_grad(self.optimizer)

        accumulation_steps = getattr(
            self.config.trainer,
            "gradient_accumulation_steps",
            1,
        )
        if (
            isinstance(accumulation_steps, bool)
            or not isinstance(accumulation_steps, int)
            or accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive.")

        total_tokens = 0
        total_sequences = 0
        loss_numerator: torch.Tensor | None = None
        exact_tokens = True
        micro_steps = 0

        for _ in range(accumulation_steps):
            result = self.task.training_step(
                self.model,
                self._next_batch(),
                self.runtime,
            )
            self.state.micro_step += 1
            micro_steps += 1
            token_count = result.tokens.supervised_tokens
            if not result.tokens.exact or token_count <= 0:
                if accumulation_steps != 1:
                    raise ValueError(
                        "Gradient accumulation requires exact positive "
                        "supervised-token counts."
                    )
                exact_tokens = False
                self.runtime.backward(result.loss)
                loss_numerator = result.loss.detach()
                total_sequences = result.tokens.sequences
                break

            weighted_loss = result.loss * token_count
            self.runtime.backward(weighted_loss)
            total_tokens += token_count
            total_sequences += result.tokens.sequences
            detached_weighted_loss = weighted_loss.detach()
            loss_numerator = (
                detached_weighted_loss
                if loss_numerator is None
                else loss_numerator + detached_weighted_loss
            )

            if (
                self.config.trainer.max_tokens is not None
                and self.state.tokens_seen + total_tokens
                >= self.config.trainer.max_tokens
            ):
                break

        if loss_numerator is None:
            raise RuntimeError("Training update produced no loss.")
        observe_accumulation = getattr(
            self.runtime,
            "observe_accumulation_steps",
            None,
        )
        if callable(observe_accumulation):
            observe_accumulation(micro_steps)
        if exact_tokens:
            if total_tokens <= 0:
                raise ValueError("Training update produced no supervised tokens.")
            self.runtime.scale_gradients(
                self.model.parameters(),
                1.0 / total_tokens,
            )

        self.runtime.clip_gradients(
            self.model.parameters(),
            self.config.trainer.max_grad_norm,
        )

        self.runtime.optimizer_step(
            self.optimizer,
        )

        self.runtime.synchronize()

        self._update_accumulated_state(
            total_tokens=total_tokens,
            total_sequences=total_sequences,
            loss_numerator=loss_numerator,
            exact_tokens=exact_tokens,
        )
        self._advance_scheduler(total_tokens=total_tokens)
        self.state.learning_rate = self._current_learning_rate()

        self.runtime.on_step_end(self.state.step)

    def _advance_scheduler(self, *, total_tokens: int) -> None:
        """Advance token-indexed schedules by cumulative consumed tokens."""

        step_tokens = getattr(self.scheduler, "step_tokens", None)
        if callable(step_tokens):
            # State accounting is rank-local; the schedule horizon describes
            # the global training corpus. Replicated DP uses equal token counts.
            world_size = getattr(self.runtime, "world_size", 1)
            step_tokens(self.state.tokens_seen * world_size)
        else:
            del total_tokens
            self.scheduler.step()

    def _evaluation_step(self, batch) -> TaskResult:
        """Dispatch one evaluation batch through the selected task."""

        return self.task.evaluation_step(
            self.model,
            batch,
            self.runtime,
        )

    def _evaluation_results(self):
        """Yield evaluation results without retaining the evaluation set."""

        for batch in self.eval_dataloader:
            yield self._evaluation_step(batch)

    def evaluate(self) -> dict[str, float]:
        """Run evaluation over the evaluation dataloader."""

        if self.eval_dataloader is None:
            raise RuntimeError(
                "Evaluation requested but no evaluation dataloader is configured."
            )

        previous_phase = self.state.phase
        if previous_phase not in {
            TrainerPhase.CREATED,
            TrainerPhase.PREPARED,
            TrainerPhase.TRAINING,
        }:
            raise RuntimeError(
                f"Cannot evaluate trainer in phase '{previous_phase.value}'."
            )
        self.state.transition(TrainerPhase.EVALUATING)
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                stream_aggregator = getattr(
                    self.task,
                    "aggregate_evaluation_stream",
                    None,
                )
                if callable(stream_aggregator):
                    metrics = stream_aggregator(self._evaluation_results())
                else:
                    results: list[TaskResult] = []
                    for result in self._evaluation_results():
                        results.append(result)
                    metrics = self.task.aggregate_evaluation(results)
            self.callback_handler.on_evaluate(self.state, self.control)
            self.callback_handler.on_metrics(
                self.state,
                self.control,
                metrics,
            )
            return metrics
        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            if was_training:
                self.model.train()
            if self.state.phase == TrainerPhase.EVALUATING:
                self.state.transition(previous_phase)

    def save_model(self):
        raise NotImplementedError

    def save_checkpoint(self, destination: object | None = None) -> object:
        """Run a checkpoint save hook within backend coordination barriers."""

        if self._checkpoint_saver is None:
            raise NotImplementedError(
                "Provide checkpoint_saver; checkpoint file formats are owned by M7."
            )
        previous_phase = self.state.phase
        if previous_phase not in {
            TrainerPhase.CREATED,
            TrainerPhase.PREPARED,
            TrainerPhase.TRAINING,
        }:
            raise RuntimeError(
                f"Cannot save checkpoint in phase '{previous_phase.value}'."
            )
        name = str(destination) if destination is not None else "manual"
        self.state.transition(TrainerPhase.SAVING)
        success = False
        try:
            self.runtime.before_checkpoint(name)
            result = self._checkpoint_saver(self, destination)
            self.callback_handler.on_save_checkpoint(self.state, self.control)
            success = True
            return result
        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            try:
                self.runtime.after_checkpoint(name, success=success)
            except BaseException as exc:
                self.state.mark_failed(exc)
                raise
            finally:
                if self.state.phase == TrainerPhase.SAVING:
                    self.state.transition(previous_phase)

    def load_checkpoint(self, source: object | None = None) -> object:
        """Restore through a checkpoint hook and publish a resume event."""

        if self._checkpoint_loader is None:
            raise NotImplementedError(
                "Provide checkpoint_loader; checkpoint file formats are owned by M7."
            )
        if self.state.phase == TrainerPhase.CREATED:
            self.prepare()
        if self.state.phase != TrainerPhase.PREPARED:
            raise RuntimeError(
                f"Cannot resume trainer in phase '{self.state.phase.value}'."
            )
        self.state.transition(TrainerPhase.RESUMING)
        success = False
        try:
            self.runtime.before_checkpoint("resume")
            result = self._checkpoint_loader(self, source)
            self.callback_handler.on_resume(self.state, self.control)
            success = True
            return result
        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            try:
                self.runtime.after_checkpoint("resume", success=success)
            except BaseException as exc:
                self.state.mark_failed(exc)
                raise
            finally:
                if self.state.phase == TrainerPhase.RESUMING:
                    self.state.transition(TrainerPhase.PREPARED)

    def finalize(self) -> TrainerState:
        """Finalize a prepared trainer that is not actively training."""

        if self.state.phase == TrainerPhase.FINALIZED:
            return self.state
        if self.state.phase == TrainerPhase.TRAINING:
            raise RuntimeError("Request stop and let train() finalize the trainer.")
        if self.state.phase != TrainerPhase.FAILED:
            self.state.transition(TrainerPhase.STOPPING)
        try:
            self.runtime.finalize()
        finally:
            if self.state.phase == TrainerPhase.FAILED:
                self.state.transition(TrainerPhase.STOPPING)
            if self.state.phase == TrainerPhase.STOPPING:
                self.state.transition(TrainerPhase.FINALIZED)
        return self.state
