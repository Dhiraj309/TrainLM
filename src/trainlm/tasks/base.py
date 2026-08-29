"""Backend- and model-family-neutral language-model task contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from trainlm.runtime import ExecutionBackend


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Host-side token accounting for one task step."""

    sequences: int
    input_tokens: int
    target_tokens: int
    supervised_tokens: int
    ignored_tokens: int
    exact: bool = True

    def __post_init__(self) -> None:
        for name in (
            "sequences",
            "input_tokens",
            "target_tokens",
            "supervised_tokens",
            "ignored_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Token count '{name}' must be non-negative.")
        if self.exact and (
            self.supervised_tokens + self.ignored_tokens != self.target_tokens
        ):
            raise ValueError(
                "Exact supervised and ignored counts must equal target tokens."
            )


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Loss and accounting produced by a language-model task step."""

    loss: torch.Tensor
    tokens: TokenCounts
    metrics: Mapping[str, torch.Tensor | float] = field(default_factory=dict)
    loss_source: str = "task"

    def __post_init__(self) -> None:
        if not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0:
            raise ValueError("Task loss must be a scalar tensor.")
        if not isinstance(self.loss_source, str) or not self.loss_source:
            raise ValueError("Task loss_source must be a non-empty string.")


@runtime_checkable
class LanguageModelTask(Protocol):
    """Task semantics consumed by the generic trainer."""

    @property
    def name(self) -> str: ...

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult: ...

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult: ...

    def aggregate_evaluation(
        self,
        results: Sequence[TaskResult],
    ) -> dict[str, float]: ...


@runtime_checkable
class StreamingEvaluationTask(Protocol):
    """Optional task capability for one-pass evaluation aggregation."""

    def aggregate_evaluation_stream(
        self,
        results: Iterable[TaskResult],
    ) -> dict[str, float]: ...
