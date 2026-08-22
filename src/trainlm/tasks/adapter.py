"""Compatibility adapters for pre-task TrainLM callables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn

from trainlm.runtime import ExecutionBackend

from .base import TaskResult, TokenCounts


class LegacyLoss(Protocol):
    def __call__(
        self,
        model: nn.Module,
        batch: Any,
        runtime: ExecutionBackend,
    ) -> torch.Tensor: ...


class LossTaskAdapter:
    """Adapt the former loss callable without claiming exact token counts."""

    name = "legacy_loss_adapter"

    def __init__(self, loss: LegacyLoss) -> None:
        self.loss = loss

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult:
        return self._step(model, batch, backend)

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult:
        return self._step(model, batch, backend)

    def aggregate_evaluation(
        self,
        results: Sequence[TaskResult],
    ) -> dict[str, float]:
        if not results:
            raise ValueError("Cannot aggregate an empty evaluation.")
        return {
            "eval_loss": sum(
                result.loss.detach().item() for result in results
            ) / len(results)
        }

    def _step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult:
        loss = self.loss(model, batch, backend)
        return TaskResult(
            loss=loss,
            tokens=TokenCounts(
                sequences=0,
                input_tokens=0,
                target_tokens=0,
                supervised_tokens=0,
                ignored_tokens=0,
                exact=False,
            ),
        )
