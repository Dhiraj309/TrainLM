"""Explicit optimized training view for linear-head causal language models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from trainlm.optimization.capabilities import ModelCapabilities

from .chunked_loss import (
    RematerializationPolicy,
    chunked_linear_causal_cross_entropy,
)

HiddenStateProvider = Callable[[nn.Module, Mapping[str, Any]], torch.Tensor]


@dataclass(frozen=True, slots=True)
class LinearCausalLMTrainingView:
    """Non-mutating, capability-guarded access to hidden states and LM head.

    The hidden-state provider is an explicit adapter boundary: TrainLM never
    guesses a model family's internal body path. The output projection is read
    through Hugging Face's public ``get_output_embeddings`` contract.
    """

    model: nn.Module
    capabilities: ModelCapabilities
    hidden_state_provider: HiddenStateProvider
    provider_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")
        if not isinstance(self.capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities.")
        if not callable(self.hidden_state_provider):
            raise TypeError("hidden_state_provider must be callable.")
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id cannot be empty.")
        head = self.capabilities.lm_head
        if head.status not in {"known", "inferred"} or head.kind != "linear":
            raise ValueError(
                "Optimized training view requires an inspected linear LM head."
            )
        self.output_projection()

    def output_projection(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        getter = getattr(self.model, "get_output_embeddings", None)
        if not callable(getter):
            raise TypeError("Model does not expose get_output_embeddings().")
        head = getter()
        weight = getattr(head, "weight", None)
        bias = getattr(head, "bias", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError("Output embeddings must expose a rank-2 weight tensor.")
        if bias is not None and (
            not isinstance(bias, torch.Tensor)
            or bias.ndim != 1
            or bias.shape[0] != weight.shape[0]
        ):
            raise TypeError("Output embedding bias must match the vocabulary size.")
        return weight, bias

    def hidden_states(self, model_inputs: Mapping[str, Any]) -> torch.Tensor:
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping.")
        hidden = self.hidden_state_provider(self.model, model_inputs)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim < 3:
            raise TypeError(
                "Hidden-state provider must return [..., sequence, hidden] tensor."
            )
        weight, _ = self.output_projection()
        if hidden.shape[-1] != weight.shape[1]:
            raise ValueError("Hidden states do not match the output projection width.")
        return hidden

    def loss(
        self,
        model_inputs: Mapping[str, Any],
        labels: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
        chunk_size: int = 2048,
        ignore_index: int = -100,
        z_loss: float = 0.0,
        rematerialization: RematerializationPolicy = "disabled",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.hidden_states(model_inputs)
        weight, bias = self.output_projection()
        return chunked_linear_causal_cross_entropy(
            hidden,
            weight,
            labels,
            bias=bias,
            loss_mask=loss_mask,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
            z_loss=z_loss,
            rematerialization=rematerialization,
        )


__all__ = ["HiddenStateProvider", "LinearCausalLMTrainingView"]
