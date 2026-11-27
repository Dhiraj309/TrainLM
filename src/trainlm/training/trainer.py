from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import time
from typing import Any
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
        try:
            self._train_batches_per_epoch = len(self.train_dataloader)
        except (TypeError, NotImplementedError, ValueError):
            self._train_batches_per_epoch = None
        checkpoint_config = getattr(config, "checkpoint", None)
        evaluation_config = getattr(config, "evaluation", None)
        if (
            getattr(config.trainer, "stop_at_epoch_end", False)
            or getattr(checkpoint_config, "save_training_at_epoch_end", False)
            or getattr(evaluation_config, "eval_at_epoch_end", False)
        ) and self._train_batches_per_epoch is None:
            raise ValueError(
                "Epoch-based trainer actions require a finite training loader."
            )
        self.eval_dataloader = (
            runtime.prepare_dataloader(eval_dataloader)
            if eval_dataloader is not None
            else None
        )
        evaluation_config = getattr(config, "evaluation", None)
        if (
            getattr(evaluation_config, "eval_at_epoch_end", False)
            or getattr(evaluation_config, "eval_every_steps", None) is not None
        ) and self.eval_dataloader is None:
            raise ValueError(
                "Scheduled evaluation requires an evaluation dataloader."
            )

        self.state = TrainerState()
        self.control = TrainerControl()

        self.callback_handler = CallbackHandler(callbacks)

        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader
        self._last_checkpoint_step: int | None = None

        self._train_iterator: Iterator | None = None
        self._active_prefetcher: Any = None
        self._prefetch_batches: bool = False
        self._global_tokens_seen: int | None = None
        self._epoch_end_pending = False
        self._compiled_single_step_fn = None
        self._compiled_single_step_active = False
        self._compiled_microbatch_step_fn = None
        self._compiled_microbatch_step_active = False
        self._lr_tensors: list[torch.Tensor] | None = None
        self._lr_values: list[float] | None = None
        self._performance_model = model
        self._performance_window_started: float | None = None
        self._performance_window_tokens = 0
        self._performance_window_step = 0
        self._performance_window_data_wait = 0.0
        self._performance = {
            "measured_optimizer_steps": 0,
            "supervised_tokens": 0,
            "wall_seconds": 0.0,
            "tokens_per_second": 0.0,
            "dataloader_wait_seconds": 0.0,
            "dataloader_wait_fraction": 0.0,
        }
        self._last_performance_window: dict[str, float] = {}
        self._performance_windows: list[dict[str, float]] = []

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
            self._initialize_global_token_budget()
            self.model.train()

            self.runtime.on_train_begin()

            self.callback_handler.on_train_begin(
                self.state,
                self.control,
            )
            self._start_performance_measurement()

            while not self._should_stop():
                self.control.reset()

                self.callback_handler.on_step_begin(
                    self.state,
                    self.control,
                )

                previous_epoch = self.state.epoch
                self._train_step()
                self._epoch_end_pending = self.state.epoch > previous_epoch

                self.callback_handler.on_step_end(
                    self.state,
                    self.control,
                )
                self._emit_step_metrics()
                self._run_scheduled_actions()

        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            self._finish_training()

        return self.state

    def _run_scheduled_actions(self) -> None:
        """Honor configured and callback-requested save/evaluation boundaries."""

        evaluation = getattr(self.config, "evaluation", None)
        eval_interval = getattr(evaluation, "eval_every_steps", None)
        should_evaluate = self.control.should_evaluate or (
            eval_interval is not None and self.state.step % eval_interval == 0
        ) or (
            getattr(evaluation, "eval_at_epoch_end", False)
            and self._epoch_end_pending
        )
        checkpoint = getattr(self.config, "checkpoint", None)
        save_interval = getattr(checkpoint, "save_training_every_steps", None)
        if should_evaluate:
            self.evaluate()
        # Evaluation callbacks may request a checkpoint for a newly selected
        # best model. Read the control flag after evaluation so that checkpoint
        # is published at the same optimizer step as its metric.
        should_save = self.control.should_save_checkpoint or (
            save_interval is not None and self.state.step % save_interval == 0
        ) or (
            getattr(checkpoint, "save_training_at_epoch_end", False)
            and self._epoch_end_pending
        )
        if should_save:
            self.save_checkpoint(f"checkpoint-{self.state.step}")
        self._epoch_end_pending = False

    def _finish_training(self) -> None:
        """Run end hooks and finalize resources exactly once."""

        self.state.is_training = False
        self.state.should_stop = self.state.should_stop or self.control.should_stop
        try:
            if self.state.phase != TrainerPhase.FAILED:
                self._record_performance_window(emit_metrics=True)
            self._save_final_training_checkpoint()
            self.state.transition(TrainerPhase.STOPPING)
            self.callback_handler.on_train_end(
                self.state,
                self.control,
            )
        except BaseException as exc:
            self.state.mark_failed(exc)
            raise
        finally:
            self._close_prefetcher()
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

    def _save_final_training_checkpoint(self) -> None:
        """Save a successful terminal step when periodic saves are enabled."""
        checkpoint = getattr(self.config, "checkpoint", None)
        save_interval = getattr(checkpoint, "save_training_every_steps", None)
        save_at_epoch_end = getattr(checkpoint, "save_training_at_epoch_end", False)
        if (
            self.state.phase == TrainerPhase.FAILED
            or (save_interval is None and not save_at_epoch_end)
            or self._checkpoint_saver is None
            or self.state.step < 1
            or self._last_checkpoint_step == self.state.step
        ):
            return
        self.save_checkpoint(f"checkpoint-{self.state.step}")

    def _should_stop(self) -> bool:
        if self.control.should_stop or self.state.should_stop:
            return True

        trainer = self.config.trainer
        if trainer.max_steps is not None and self.state.step >= trainer.max_steps:
            return True
        max_train_batches = getattr(trainer, "max_train_batches", None)
        if (
            max_train_batches is not None
            and self.state.micro_step >= max_train_batches
        ):
            return True
        if (
            trainer.max_tokens is not None
            and (
                self._global_tokens_seen
                if self._global_tokens_seen is not None
                else self.state.tokens_seen
            )
            >= trainer.max_tokens
        ):
            return True
        return False

    def _initialize_global_token_budget(self) -> None:
        """Restore the global token counter from per-rank checkpoint state."""

        if self.config.trainer.max_tokens is None:
            self._global_tokens_seen = None
            return
        self._global_tokens_seen = self._sum_tokens(self.state.tokens_seen)

    def _sum_tokens(self, local_tokens: int) -> int:
        """Return a global token count while preserving single-rank behavior."""

        if not self.runtime.is_distributed:
            return local_tokens
        value = torch.tensor(
            local_tokens,
            dtype=torch.int64,
            device=self.runtime.device,
        )
        return int(self.runtime.reduce_sum(value).item())

    def _sum_progress_counters(self) -> tuple[int, int]:
        """Return global progress values for distributed logging snapshots."""

        if not self.runtime.is_distributed:
            return self.state.tokens_seen, self.state.samples_seen
        counters = torch.tensor(
            (self.state.tokens_seen, self.state.samples_seen),
            dtype=torch.int64,
            device=self.runtime.device,
        )
        totals = self.runtime.reduce_sum(counters)
        return int(totals[0].item()), int(totals[1].item())

    def _global_microbatch_tokens(self, *, local_tokens: int, exact: bool) -> int:
        """Synchronize token counts and validate exact global-budget inputs."""

        if not self.runtime.is_distributed:
            if not exact:
                raise ValueError(
                    "max_tokens stopping requires exact supervised-token counts."
                )
            return local_tokens

        counts = torch.tensor(
            (local_tokens, int(exact)),
            dtype=torch.int64,
            device=self.runtime.device,
        )
        reduced = self.runtime.reduce_sum(counts)
        global_tokens = int(reduced[0].item())
        exact_ranks = int(reduced[1].item())
        if exact_ranks != self.runtime.world_size:
            raise ValueError(
                "Distributed max_tokens stopping requires exact supervised-token "
                "counts on every rank."
            )
        if global_tokens <= 0:
            raise ValueError(
                "Distributed max_tokens update produced no supervised tokens."
            )
        return global_tokens

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
        self._record_performance_window(emit_metrics=False)
        tokens_seen, samples_seen = self._sum_progress_counters()
        metrics: dict[str, float] = {
            "step": float(self.state.step),
            "tokens_seen": float(tokens_seen),
            "samples_seen": float(samples_seen),
            "learning_rate": self.state.learning_rate,
        }
        if self.state.loss is not None:
            metrics["loss"] = self.state.loss
        if self.state.grad_norm is not None:
            metrics["grad_norm"] = self.state.grad_norm
        metrics.update(self._last_performance_window)
        self.callback_handler.on_metrics(self.state, self.control, metrics)

    @property
    def performance_measurements(self) -> dict[str, float | int]:
        """Return aggregate synchronized training-window observations."""

        return dict(self._performance)

    def performance_summary(self) -> dict[str, object]:
        """Return measured throughput, input-wait, and supported MFU details."""

        from .performance import build_performance_summary

        measurements = {
            **self._performance,
            "recent_windows": list(self._performance_windows),
        }
        wall_seconds = float(self._performance.get("wall_seconds", 0.0))
        global_tokens = float(self._performance.get("global_supervised_tokens", 0))
        if self.runtime.world_size > 1 and wall_seconds > 0 and global_tokens > 0:
            measurements["global_tokens_per_second"] = global_tokens / wall_seconds
        return build_performance_summary(
            measurements=measurements,
            model=self._performance_model,
            runtime=self.runtime,
            sequence_length=int(
                getattr(getattr(self.config, "dataset", None), "sequence_length", 0)
            ),
        )

    def _start_performance_measurement(self) -> None:
        self._performance_window_started = time.perf_counter()
        self._performance_window_tokens = self.state.tokens_seen
        self._performance_window_step = self.state.step

    def _record_performance_window(self, *, emit_metrics: bool) -> None:
        """Close a sparse synchronized wall-clock window and update counters."""

        started = self._performance_window_started
        if started is None:
            return
        self.runtime.synchronize()
        ended = time.perf_counter()
        wall_seconds = ended - started
        tokens = self.state.tokens_seen - self._performance_window_tokens
        steps = self.state.step - self._performance_window_step
        data_wait = min(self._performance_window_data_wait, wall_seconds)
        self._performance_window_started = ended
        self._performance_window_tokens = self.state.tokens_seen
        self._performance_window_step = self.state.step
        self._performance_window_data_wait = 0.0
        if wall_seconds <= 0 or tokens <= 0 or steps <= 0:
            return

        throughput = tokens / wall_seconds
        global_tokens = (
            self._sum_tokens(tokens) if self.runtime.world_size > 1 else tokens
        )
        global_throughput = global_tokens / wall_seconds
        self._performance["measured_optimizer_steps"] += steps
        self._performance["supervised_tokens"] += tokens
        self._performance["global_supervised_tokens"] = (
            int(self._performance.get("global_supervised_tokens", 0)) + global_tokens
        )
        self._performance["wall_seconds"] += wall_seconds
        self._performance["tokens_per_second"] = (
            self._performance["supervised_tokens"] / self._performance["wall_seconds"]
        )
        self._performance["dataloader_wait_seconds"] += data_wait
        self._performance["dataloader_wait_fraction"] = (
            self._performance["dataloader_wait_seconds"]
            / self._performance["wall_seconds"]
        )
        self._last_performance_window = {
            "train_wall_seconds": wall_seconds,
            "train_tokens_per_second": throughput,
            "dataloader_wait_seconds": data_wait,
            "dataloader_wait_fraction": data_wait / wall_seconds,
        }
        from .performance import build_performance_summary

        window_measurements: dict[str, object] = {
            "measured_optimizer_steps": steps,
            "supervised_tokens": tokens,
            "wall_seconds": wall_seconds,
            "tokens_per_second": throughput,
            "dataloader_wait_seconds": data_wait,
            "dataloader_wait_fraction": data_wait / wall_seconds,
        }
        if self.runtime.world_size > 1:
            window_measurements["global_tokens_per_second"] = global_throughput
        summary = build_performance_summary(
            measurements=window_measurements,
            model=self._performance_model,
            runtime=self.runtime,
            sequence_length=int(
                getattr(getattr(self.config, "dataset", None), "sequence_length", 0)
            ),
        )
        if summary.get("mfu_status") == "measured":
            self._last_performance_window["non_embedding_mfu"] = float(
                summary["non_embedding_mfu"]
            )
            self._last_performance_window["logits_inclusive_mfu"] = float(
                summary["logits_inclusive_mfu"]
            )
        self._performance_windows.append({
            "step": float(self.state.step),
            "supervised_tokens": float(tokens),
            **self._last_performance_window,
        })
        if len(self._performance_windows) > 100:
            del self._performance_windows[:-100]
        if emit_metrics:
            tokens_seen, samples_seen = self._sum_progress_counters()
            metrics = {
                "step": float(self.state.step),
                "tokens_seen": float(tokens_seen),
                "samples_seen": float(samples_seen),
                "learning_rate": self.state.learning_rate,
                **self._last_performance_window,
            }
            if self.state.loss is not None:
                metrics["loss"] = self.state.loss
            self.callback_handler.on_metrics(self.state, self.control, metrics)

    def request_stop(self) -> None:
        """Request a graceful stop at the next completed-step boundary."""

        self.control.request_stop()
        self.state.should_stop = True

    def _close_prefetcher(self) -> None:
        prefetcher = getattr(self, "_active_prefetcher", None)
        if prefetcher is not None:
            try:
                prefetcher.close()
            finally:
                self._active_prefetcher = None

    def _new_train_iterator(self) -> Iterator:
        self._close_prefetcher()
        raw_iter = iter(self.train_dataloader)
        if getattr(self, "_prefetch_batches", False):
            from trainlm.runtime.prefetch import AsyncBatchPrefetcher

            prefetcher = AsyncBatchPrefetcher(raw_iter, prefetch_depth=2)
            self._active_prefetcher = prefetcher
            return prefetcher
        return raw_iter

    def _next_batch(self):
        if self._train_iterator is None:
            self._train_iterator = self._new_train_iterator()

        started = time.perf_counter()
        try:
            return next(self._train_iterator)

        except StopIteration:
            self._train_iterator = self._new_train_iterator()
            return next(self._train_iterator)
        finally:
            if self._performance_window_started is not None:
                self._performance_window_data_wait += time.perf_counter() - started

    def _current_learning_rate(self) -> float:
        """Return the current learning rate."""

        lr_values = getattr(self, "_lr_values", None)
        if lr_values:
            return float(lr_values[0])
        lr = self.scheduler.get_last_lr()[0]
        return float(lr.item() if isinstance(lr, torch.Tensor) else lr)

    def _update_state(
        self,
        *,
        result: TaskResult,
    ) -> None:
        """Update trainer state after a completed optimization step."""

        self.state.step += 1
        self.state.tokens_seen += result.tokens.supervised_tokens
        self.state.samples_seen += result.tokens.sequences
        self.state.loss = (
            result.loss.detach().item()
            if self._should_materialize_loss()
            else None
        )
        self.state.learning_rate = self._current_learning_rate()

    def _should_materialize_loss(self) -> bool:
        """Whether this completed update should synchronize its loss.

        XLA executes asynchronously. Materializing a scalar with ``item()``
        is a synchronization point, so high-throughput runs should request
        loss snapshots at the same cadence as host logging rather than every
        optimizer update.
        """

        interval = getattr(
            self.config.trainer,
            "materialize_loss_every_steps",
            1,
        )
        # Always retain a final loss/gradient snapshot, even when the run ends
        # between logging intervals or a token budget stops mid-accumulation.
        return self.state.step % interval == 0 or self._should_stop()

    def _update_accumulated_state(
        self,
        *,
        total_tokens: int,
        total_sequences: int,
        loss_numerator: torch.Tensor,
        exact_tokens: bool,
        grad_norm: torch.Tensor | None,
        global_loss_numerator: torch.Tensor | None = None,
        global_tokens: int | None = None,
        global_sequences: int | None = None,
    ) -> None:
        """Commit one optimizer update after all microbatches are reduced."""

        self.state.step += 1
        self.state.tokens_seen += total_tokens
        self.state.samples_seen += total_sequences
        self.state.global_batch_size = (
            global_sequences if global_sequences is not None else total_sequences
        )
        if self._should_materialize_loss():
            if global_loss_numerator is not None and global_tokens is not None:
                loss_value = global_loss_numerator.detach().item()
                loss_denominator = global_tokens
            else:
                loss_value = loss_numerator.detach().item()
                loss_denominator = total_tokens
            self.state.loss = (
                loss_value / loss_denominator
                if exact_tokens and loss_denominator > 0
                else loss_value
            )
            self.state.grad_norm = (
                float(grad_norm.detach().item()) if grad_norm is not None else None
            )
        else:
            self.state.loss = None
            self.state.grad_norm = None
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

        microbatch_limit = accumulation_steps
        max_train_batches = getattr(self.config.trainer, "max_train_batches", None)
        stop_at_epoch_end = getattr(
            self.config.trainer, "stop_at_epoch_end", False
        )
        if max_train_batches is not None or stop_at_epoch_end:
            if not self._train_batches_per_epoch:
                raise ValueError(
                    "Epoch-bounded accumulation requires a finite training loader."
                )
            if max_train_batches is not None:
                remaining_batches = max_train_batches - self.state.micro_step
                if remaining_batches < 1:
                    raise RuntimeError(
                        "The configured training-batch budget is exhausted."
                    )
                microbatch_limit = min(microbatch_limit, remaining_batches)
            batches_to_epoch_end = self._train_batches_per_epoch - (
                self.state.micro_step % self._train_batches_per_epoch
            )
            microbatch_limit = min(microbatch_limit, batches_to_epoch_end)

        if (
            microbatch_limit == 1
            and not self.runtime.is_distributed
            and callable(getattr(self.task, "prepare_batch_on_host", None))
            and callable(getattr(self.task, "training_step_prepared", None))
        ):
            raw_batch = self._next_batch()
            if self._can_prepare_on_host(raw_batch):
                task_batch, counts = self.task.prepare_batch_on_host(raw_batch)
                self.state.micro_step += 1
                if self._train_batches_per_epoch:
                    self.state.epoch = (
                        self.state.micro_step // self._train_batches_per_epoch
                    )
                token_count = counts.supervised_tokens
                if self.config.trainer.max_tokens is not None:
                    global_microbatch_tokens = self._global_microbatch_tokens(
                        local_tokens=token_count,
                        exact=counts.exact,
                    )
                    if self._global_tokens_seen is None:
                        raise RuntimeError("Global token budget was not initialized.")
                    self._global_tokens_seen += global_microbatch_tokens
                step_fn = self._resolve_single_step_fn()
                task_batch = self.runtime.prepare_batch(task_batch)
                mean_loss, grad_norm = step_fn(task_batch, counts)
                observe_accumulation = getattr(
                    self.runtime,
                    "observe_accumulation_steps",
                    None,
                )
                if callable(observe_accumulation):
                    observe_accumulation(1)
                self._update_accumulated_state(
                    total_tokens=token_count,
                    total_sequences=counts.sequences,
                    loss_numerator=mean_loss,
                    exact_tokens=False,
                    grad_norm=grad_norm,
                )
                self._advance_scheduler(total_tokens=token_count)
                self.state.learning_rate = self._current_learning_rate()
                self.runtime.on_step_end(self.state.step)
                return
            self._train_iterator = self._prepend_batch(raw_batch)

        total_tokens = 0
        total_sequences = 0
        loss_numerator: torch.Tensor | None = None
        exact_tokens = True
        micro_steps = 0
        runtime_dev_type = getattr(getattr(self.runtime, "device", None), "type", None)
        use_compiled_microbatch = (
            runtime_dev_type == "cuda"
            and bool(getattr(self.runtime, "_compile_enabled", False))
            and not self.runtime.is_distributed
            and callable(getattr(self.task, "prepare_batch_on_host", None))
            and callable(getattr(self.task, "training_step_prepared", None))
            and callable(getattr(self.runtime, "compile_training_step", None))
        )

        for _ in range(microbatch_limit):
            if use_compiled_microbatch:
                raw_batch = self._next_batch()
                if self._can_prepare_on_host(raw_batch):
                    task_batch, counts = self.task.prepare_batch_on_host(raw_batch)
                    if counts.exact and counts.supervised_tokens > 0:
                        self.state.micro_step += 1
                        if self._train_batches_per_epoch:
                            self.state.epoch = (
                                self.state.micro_step // self._train_batches_per_epoch
                            )
                        micro_steps += 1
                        token_count = counts.supervised_tokens
                        stop_after_microbatch = False
                        max_tokens = self.config.trainer.max_tokens
                        if max_tokens is not None:
                            global_microbatch_tokens = self._global_microbatch_tokens(
                                local_tokens=token_count,
                                exact=counts.exact,
                            )
                            if self._global_tokens_seen is None:
                                raise RuntimeError("Global token budget was not initialized.")
                            self._global_tokens_seen += global_microbatch_tokens
                            stop_after_microbatch = self._global_tokens_seen >= max_tokens
                        microbatch_fn = self._resolve_microbatch_step_fn()
                        task_batch = self.runtime.prepare_batch(task_batch)
                        detached_weighted_loss = microbatch_fn(task_batch, counts)
                        total_tokens += token_count
                        total_sequences += counts.sequences
                        loss_numerator = (
                            detached_weighted_loss
                            if loss_numerator is None
                            else loss_numerator + detached_weighted_loss
                        )
                        if stop_after_microbatch:
                            break
                        continue
                self._train_iterator = self._prepend_batch(raw_batch)

            result = self.task.training_step(
                self.model,
                self._next_batch(),
                self.runtime,
            )
            self.state.micro_step += 1
            if self._train_batches_per_epoch:
                self.state.epoch = (
                    self.state.micro_step // self._train_batches_per_epoch
                )
            micro_steps += 1
            token_count = result.tokens.supervised_tokens
            stop_after_microbatch = False
            max_tokens = self.config.trainer.max_tokens
            if max_tokens is not None:
                global_microbatch_tokens = self._global_microbatch_tokens(
                    local_tokens=token_count,
                    exact=result.tokens.exact,
                )
                if self._global_tokens_seen is None:
                    raise RuntimeError("Global token budget was not initialized.")
                self._global_tokens_seen += global_microbatch_tokens
                stop_after_microbatch = self._global_tokens_seen >= max_tokens

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

            if stop_after_microbatch:
                break

        if loss_numerator is None:
            raise RuntimeError("Training update produced no loss.")
        global_loss_numerator = None
        global_token_tensor = None
        global_tokens = None
        global_sequences = None
        if (
            self.runtime.is_distributed
            and exact_tokens
            and getattr(
                self.runtime,
                "mean_reduces_gradients_before_clipping",
                False,
            )
        ):
            detached_loss_numerator = loss_numerator.detach()
            local_update_stats = torch.stack(
                (
                    detached_loss_numerator.to(dtype=torch.float32),
                    detached_loss_numerator.new_tensor(
                        float(total_tokens), dtype=torch.float32
                    ),
                    detached_loss_numerator.new_tensor(
                        float(total_sequences), dtype=torch.float32
                    ),
                )
            )
            global_update_stats = self.runtime.reduce_sum(local_update_stats)
            global_loss_numerator = global_update_stats[0]
            global_token_tensor = global_update_stats[1]
            global_tokens = int(global_update_stats[1].item())
            global_sequences = int(global_update_stats[2].item())

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
            if (
                self.runtime.is_distributed
                and getattr(
                    self.runtime,
                    "mean_reduces_gradients_before_clipping",
                    False,
                )
            ):
                # Convert each rank's accumulated loss sum into the global
                # supervised-token mean before a backend's pre-clip gradient
                # average: (world_size / global_tokens) * mean_r(local_sum).
                if global_token_tensor is None:
                    raise RuntimeError(
                        "Distributed token totals were not reduced before scaling."
                    )
                gradient_scale = self.runtime.world_size / global_token_tensor
            else:
                gradient_scale = 1.0 / total_tokens
            self.runtime.scale_gradients(self.model.parameters(), gradient_scale)

        grad_norm = None
        if self.config.trainer.max_grad_norm > 0:
            grad_norm = self.runtime.clip_gradients(
                self.model.parameters(),
                self.config.trainer.max_grad_norm,
            )

        self.runtime.optimizer_step(
            self.optimizer,
        )

        # XLA flushes the optimizer update in ``on_step_end`` below. Calling
        # synchronize here as well creates two mark_step boundaries per update
        # and fragments the lazy graph on TPU. Other runtimes keep their normal
        # optimizer semantics through their own on_step_end hook.

        self._update_accumulated_state(
            total_tokens=total_tokens,
            total_sequences=total_sequences,
            loss_numerator=loss_numerator,
            exact_tokens=exact_tokens,
            grad_norm=grad_norm,
            global_loss_numerator=global_loss_numerator,
            global_tokens=global_tokens,
            global_sequences=global_sequences,
        )
        self._advance_scheduler(total_tokens=total_tokens)
        self.state.learning_rate = self._current_learning_rate()

        self.runtime.on_step_end(self.state.step)

    @staticmethod
    def _can_prepare_on_host(raw_batch: Any) -> bool:
        from collections.abc import Mapping

        if not isinstance(raw_batch, Mapping):
            return False
        return all(
            not isinstance(value, torch.Tensor) or value.device.type == "cpu"
            for value in raw_batch.values()
        )

    def _prepend_batch(self, raw_batch: Any) -> Iterator:
        from itertools import chain

        assert self._train_iterator is not None
        return chain((raw_batch,), self._train_iterator)

    def _resolve_single_step_fn(self):
        if self._compiled_single_step_fn is not None:
            return self._compiled_single_step_fn

        def _single_step_tensors(task_batch, counts):
            step_model = (
                getattr(self.model, "_orig_mod", self.model)
                if self._compiled_single_step_active
                else self.model
            )
            result = self.task.training_step_prepared(
                step_model,
                task_batch,
                counts,
                self.runtime,
            )
            self.runtime.backward(result.loss)
            grad_norm = None
            if self.config.trainer.max_grad_norm > 0:
                grad_norm = self.runtime.clip_gradients(
                    step_model.parameters(),
                    self.config.trainer.max_grad_norm,
                )
            self.runtime.optimizer_step(self.optimizer)
            return result.loss.detach(), grad_norm

        compile_fn = getattr(self.runtime, "compile_training_step", None)
        if callable(compile_fn):
            compiled = compile_fn(_single_step_tensors)
            self._compiled_single_step_active = compiled is not _single_step_tensors
            self._compiled_single_step_fn = compiled
        else:
            self._compiled_single_step_active = False
            self._compiled_single_step_fn = _single_step_tensors
        if self._compiled_single_step_active:
            self._sync_optimizer_lr_tensors()
        return self._compiled_single_step_fn

    def _resolve_microbatch_step_fn(self):
        if self._compiled_microbatch_step_fn is not None:
            return self._compiled_microbatch_step_fn

        def _microbatch_step_tensors(task_batch, counts):
            step_model = (
                getattr(self.model, "_orig_mod", self.model)
                if self._compiled_microbatch_step_active
                else self.model
            )
            result = self.task.training_step_prepared(
                step_model,
                task_batch,
                counts,
                self.runtime,
            )
            weighted_loss = result.loss * counts.supervised_tokens
            self.runtime.backward(weighted_loss)
            return weighted_loss.detach()

        compile_fn = getattr(self.runtime, "compile_training_step", None)
        if callable(compile_fn):
            compiled = compile_fn(_microbatch_step_tensors)
            self._compiled_microbatch_step_active = compiled is not _microbatch_step_tensors
            self._compiled_microbatch_step_fn = compiled
        else:
            self._compiled_microbatch_step_active = False
            self._compiled_microbatch_step_fn = _microbatch_step_tensors
        return self._compiled_microbatch_step_fn

    def _sync_optimizer_lr_tensors(self) -> None:
        """Keep ``group['lr']`` as persistent 0-D tensors when compiled to prevent Dynamo recompiles."""
        if not self._compiled_single_step_active:
            return
        param_groups = getattr(self.optimizer, "param_groups", None)
        if not param_groups:
            return
        is_adamw = isinstance(self.optimizer, torch.optim.AdamW)
        if getattr(self, "_lr_tensors", None) is None:
            self._lr_tensors = []
        if getattr(self, "_lr_values", None) is None:
            self._lr_values = []
        default_device = getattr(self.runtime, "device", None) or torch.device("cpu")
        for idx, group in enumerate(param_groups):
            if is_adamw and "capturable" in group:
                group["capturable"] = True
            current_lr = group.get("lr")
            if current_lr is None:
                continue
            if idx < len(self._lr_tensors) and current_lr is self._lr_tensors[idx]:
                lr_value = self._lr_values[idx]
            else:
                lr_value = float(
                    current_lr.item()
                    if isinstance(current_lr, torch.Tensor)
                    else current_lr
                )
            if idx >= len(self._lr_tensors):
                target_device = default_device
                for param in group.get("params", ()):
                    if isinstance(param, torch.Tensor):
                        target_device = param.device
                        break
                self._lr_tensors.append(
                    torch.tensor(lr_value, dtype=torch.float32, device=target_device)
                )
                self._lr_values.append(lr_value)
            elif lr_value != self._lr_values[idx]:
                self._lr_tensors[idx].fill_(lr_value)
                self._lr_values[idx] = lr_value
            group["lr"] = self._lr_tensors[idx]

    def _advance_scheduler(self, *, total_tokens: int) -> None:
        """Advance token-indexed schedules by cumulative consumed tokens."""

        if (
            getattr(self, "_compiled_single_step_active", False)
            and getattr(self, "_lr_values", None)
            and getattr(self, "_lr_tensors", None)
        ):
            param_groups = getattr(self.optimizer, "param_groups", None) or ()
            for idx, group in enumerate(param_groups):
                if (
                    idx < len(self._lr_values)
                    and group.get("lr") is self._lr_tensors[idx]
                ):
                    group["lr"] = self._lr_values[idx]

        step_tokens = getattr(self.scheduler, "step_tokens", None)
        if callable(step_tokens):
            # Use exact global counts when the backend advertises mean-reduced
            # gradients or when global token-budget tracking is active.
            global_tokens_seen = getattr(self, "_global_tokens_seen", None)
            if global_tokens_seen is None:
                if getattr(
                    self.runtime,
                    "mean_reduces_gradients_before_clipping",
                    False,
                ):
                    global_tokens_seen = self._sum_tokens(self.state.tokens_seen)
                else:
                    global_tokens_seen = (
                        self.state.tokens_seen * self.runtime.world_size
                    )
            step_tokens(global_tokens_seen)
        else:
            del total_tokens
            self.scheduler.step()
        sync_lr = getattr(self, "_sync_optimizer_lr_tensors", None)
        if callable(sync_lr):
            sync_lr()

    def _evaluation_step(self, batch) -> TaskResult:
        """Dispatch one evaluation batch through the selected task."""

        return self.task.evaluation_step(
            self.model,
            batch,
            self.runtime,
        )

    def _evaluation_results(self):
        """Yield evaluation results without retaining the evaluation set."""

        evaluation = getattr(self.config, "evaluation", None)
        maximum = getattr(evaluation, "max_batches", None)
        for batch_index, batch in enumerate(self.eval_dataloader):
            if maximum is not None and batch_index >= maximum:
                break
            yield self._evaluation_step(batch)

    def evaluate(self) -> dict[str, float]:
        """Run evaluation, including after training has finalized."""

        if self.eval_dataloader is None:
            raise RuntimeError(
                "Evaluation requested but no evaluation dataloader is configured."
            )

        previous_phase = self.state.phase
        if previous_phase not in {
            TrainerPhase.CREATED,
            TrainerPhase.PREPARED,
            TrainerPhase.TRAINING,
            TrainerPhase.FINALIZED,
        }:
            raise RuntimeError(
                f"Cannot evaluate trainer in phase '{previous_phase.value}'."
            )
        self.state.transition(TrainerPhase.EVALUATING)
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                distributed_aggregator = getattr(
                    self.task,
                    "aggregate_distributed_evaluation_stream",
                    None,
                )
                stream_aggregator = getattr(
                    self.task,
                    "aggregate_evaluation_stream",
                    None,
                )
                if self.runtime.is_distributed and callable(distributed_aggregator):
                    metrics = distributed_aggregator(
                        self._evaluation_results(), self.runtime
                    )
                elif callable(stream_aggregator):
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
        saved_step = self.state.step
        self.state.transition(TrainerPhase.SAVING)
        success = False
        try:
            self.runtime.before_checkpoint(name)
            result = self._checkpoint_saver(self, destination)
            self.callback_handler.on_save_checkpoint(self.state, self.control)
            self._last_checkpoint_step = saved_step
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
            self._last_checkpoint_step = self.state.step
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
