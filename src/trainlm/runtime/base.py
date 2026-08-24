"""Backend-neutral execution contract used by the training engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import nn
from torch.optim import Optimizer

Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True, slots=True)
class LogicalMesh:
    """Backend-neutral logical device mesh requested by TrainLM."""

    axis_sizes: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.axis_sizes:
            raise ValueError("A logical mesh must define at least one axis.")
        for name, size in self.axis_sizes.items():
            if not name:
                raise ValueError("Logical mesh axis names cannot be empty.")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"Logical mesh axis '{name}' must be positive.")

    @property
    def size(self) -> int:
        result = 1
        for axis_size in self.axis_sizes.values():
            result *= axis_size
        return result


@dataclass(frozen=True, slots=True)
class BackendDiagnostics:
    """Portable runtime facts and backend-specific diagnostic values."""

    backend: str
    device_type: str
    precision: Precision
    world_size: int
    rank: int
    local_rank: int
    values: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("Backend diagnostic name cannot be empty.")
        if self.world_size < 1:
            raise ValueError("Backend world size must be positive.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("Backend rank must be within world size.")
        if self.local_rank < 0:
            raise ValueError("Backend local rank cannot be negative.")


@runtime_checkable
class ExecutionBackend(Protocol):
    """Structural protocol for a replaceable training execution backend."""

    @property
    def name(self) -> str: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def precision(self) -> Precision: ...

    @property
    def world_size(self) -> int: ...

    @property
    def rank(self) -> int: ...

    @property
    def local_rank(self) -> int: ...

    @property
    def is_distributed(self) -> bool: ...

    @property
    def is_primary_process(self) -> bool: ...

    def initialize(self) -> None: ...

    def finalize(self) -> None: ...

    def on_train_begin(self) -> None: ...

    def on_train_end(self) -> None: ...

    def on_step_begin(self, step: int) -> None: ...

    def on_step_end(self, step: int) -> None: ...

    def prepare_model(self, model: nn.Module) -> nn.Module: ...

    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer: ...

    def prepare_dataloader(self, dataloader: Any) -> Any: ...

    def prepare_batch(self, batch: Any) -> Any: ...

    def autocast(self) -> AbstractContextManager[Any]: ...

    def compile_model(self, model: nn.Module) -> nn.Module: ...

    def create_mesh(self, mesh: LogicalMesh) -> Any: ...

    def shard_model(self, model: nn.Module, mesh: Any) -> nn.Module: ...

    def shard_optimizer(self, optimizer: Optimizer, mesh: Any) -> Optimizer: ...

    def backward(self, loss: torch.Tensor) -> None: ...

    def clip_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> None: ...

    def scale_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        scale: float,
    ) -> None: ...

    def optimizer_step(self, optimizer: Optimizer) -> None: ...

    def zero_grad(self, optimizer: Optimizer) -> None: ...

    def synchronize(self) -> None: ...

    def barrier(self, name: str | None = None) -> None: ...

    def before_checkpoint(self, name: str) -> None: ...

    def after_checkpoint(self, name: str, *, success: bool) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

    def diagnostics(self) -> BackendDiagnostics: ...
