from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


class Runtime:
    """Execution backend for model training."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare a model for execution."""
        return model

    def prepare_batch(self, batch: Any) -> Any:
        """Prepare a batch before execution."""
        return batch

    def autocast(self):
        """Return the runtime autocast context."""
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Compute gradients."""
        loss.backward()

    def clip_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> None:
        """Clip gradients."""
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Execute an optimizer step."""
        optimizer.step()

    def zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Clear gradients."""
        optimizer.zero_grad(set_to_none=True)

    def synchronize(self) -> None:
        """Synchronize pending runtime work."""
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return runtime state."""
        return {}

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        """Restore runtime state."""
        del state_dict
