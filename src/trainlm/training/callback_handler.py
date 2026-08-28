"""
Trainer callback dispatcher.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .callback import TrainerCallback
from .control import TrainerControl
from .metrics import MetricSnapshot
from .state import TrainerState


class CallbackHandler:
    """
    Dispatches trainer events to registered callbacks.
    """

    def __init__(
        self,
        callbacks: Iterable[TrainerCallback] | None = None,
    ) -> None:
        self._callbacks = list(callbacks or [])

    @property
    def callbacks(self) -> tuple[TrainerCallback, ...]:
        """
        Registered callbacks.
        """
        return tuple(self._callbacks)

    def add_callback(
        self,
        callback: TrainerCallback,
    ) -> None:
        """
        Register a callback.
        """
        self._callbacks.append(callback)

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_train_begin(state, control)

    def on_resume(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_resume(state, control)

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_train_end(state, control)

    def on_step_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_step_begin(state, control)

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_step_end(state, control)

    def on_metrics(
        self,
        state: TrainerState,
        control: TrainerControl,
        metrics: Mapping[str, float],
    ) -> None:
        snapshot = MetricSnapshot.from_mapping(metrics)
        for callback in self._callbacks:
            callback.on_metrics(state, control, snapshot)

    def on_evaluate(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_evaluate(state, control)

    def on_save_checkpoint(
        self,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        for callback in self._callbacks:
            callback.on_save_checkpoint(state, control)
