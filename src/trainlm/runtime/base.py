from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch


class Runtime(ABC):
    """
    Abstract execution backend.

    A Runtime encapsulates all device-specific execution details so that
    the Trainer remains backend-independent.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The primary execution device."""

    @property
    @abstractmethod
    def is_distributed(self) -> bool:
        """Whether distributed execution is active."""

    @abstractmethod
    def prepare_model(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Prepare a model for execution."""

    @abstractmethod
    def prepare_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """Prepare an optimizer for execution."""

    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
    ) -> None:
        """Execute the backward pass."""

    @abstractmethod
    def clip_grad_norm(
        self,
        parameters,
        max_norm: float,
    ) -> None:
        """Clip gradients."""

    @abstractmethod
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Perform an optimiser step."""

    @abstractmethod
    def zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Clear gradients."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronise all workers."""

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return runtime state."""

    @abstractmethod
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> None:
        """Restore runtime state."""
