"""Family-neutral normalization of causal language-model outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """The output fields required by TrainLM's dense causal task."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor):
            raise TypeError("Causal LM logits must be a torch.Tensor.")
        if self.loss is not None and (
            not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0
        ):
            raise TypeError("A model-provided causal LM loss must be scalar.")


def normalize_causal_lm_output(output: Any) -> CausalLMOutput:
    """Normalize HF attribute, mapping, and tuple output conventions."""

    if hasattr(output, "logits"):
        logits = output.logits
        loss = getattr(output, "loss", None)
    elif isinstance(output, Mapping):
        if "logits" not in output:
            raise TypeError("Model output mapping must contain 'logits'.")
        logits = output["logits"]
        loss = output.get("loss")
    elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        if not output:
            raise TypeError("Model output tuple cannot be empty.")
        first = output[0]
        if isinstance(first, torch.Tensor) and first.ndim == 0:
            if len(output) < 2:
                raise TypeError("Tuple output with loss must also contain logits.")
            loss = first
            logits = output[1]
        else:
            loss = None
            logits = first
    else:
        raise TypeError(
            "Model output must expose logits by attribute, mapping, or tuple."
        )

    return CausalLMOutput(logits=logits, loss=loss)
