from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn

from trainlm.runtime import ExecutionBackend


class Loss(Protocol):
    """Protocol for training loss computation."""

    def __call__(
        self,
        model: nn.Module,
        batch: Any,
        runtime: ExecutionBackend,
    ) -> torch.Tensor:
        ...


class LanguageModelLoss:
    """Legacy adapter target for model-owned Hugging Face loss.

    New training code should use ``trainlm.tasks.CausalLMTask`` so shifting,
    masking, normalization, and token accounting remain explicit.
    """

    def __call__(
        self,
        model: nn.Module,
        batch: Any,
        runtime: ExecutionBackend,
    ) -> torch.Tensor:
        batch = runtime.prepare_batch(batch)

        with runtime.autocast():
            outputs = model(**batch)

        try:
            return outputs.loss
        except AttributeError as exc:
            raise TypeError(
                "Model output must expose a 'loss' attribute."
            ) from exc
