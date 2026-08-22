"""Canonical dense autoregressive causal-language-model task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from trainlm.runtime import ExecutionBackend

from .base import TaskResult, TokenCounts


class CausalLMTask:
    """Compute consistent next-token loss for HF-style causal LMs."""

    name = "causal_lm"

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        normalization: Literal["supervised_tokens", "batch"] = (
            "supervised_tokens"
        ),
        z_loss: float = 0.0,
    ) -> None:
        if normalization not in {"supervised_tokens", "batch"}:
            raise ValueError(f"Unsupported loss normalization: {normalization}")
        if z_loss < 0:
            raise ValueError("z_loss must be non-negative.")
        self.ignore_index = ignore_index
        self.normalization = normalization
        self.z_loss = z_loss

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

        weight_name = (
            "supervised_tokens"
            if self.normalization == "supervised_tokens"
            else "sequences"
        )
        weighted_loss = 0.0
        total_weight = 0
        for result in results:
            weight = getattr(result.tokens, weight_name)
            weighted_loss += result.loss.detach().item() * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Evaluation contains no normalization units.")
        return {"eval_loss": weighted_loss / total_weight}

    def _step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult:
        task_batch, counts = self._prepare_task_batch(batch)
        task_batch = backend.prepare_batch(task_batch)

        labels = task_batch["labels"]
        loss_mask = task_batch["loss_mask"]
        model_inputs = {
            key: value
            for key, value in task_batch.items()
            if key not in {"labels", "loss_mask"}
        }

        with backend.autocast():
            outputs = model(**model_inputs)
            logits = _extract_logits(outputs)
            loss, z_loss_value = self._loss(logits, labels, loss_mask)

        metrics: dict[str, torch.Tensor | float] = {}
        if z_loss_value is not None:
            metrics["z_loss"] = z_loss_value.detach()

        return TaskResult(loss=loss, tokens=counts, metrics=metrics)

    def _prepare_task_batch(
        self,
        batch: Any,
    ) -> tuple[dict[str, Any], TokenCounts]:
        if not isinstance(batch, Mapping):
            raise TypeError("Causal LM batches must be mappings.")
        if "input_ids" not in batch:
            raise ValueError("Causal LM batches require 'input_ids'.")

        input_ids = batch["input_ids"]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
            raise ValueError("'input_ids' must be a rank-2-or-greater tensor.")
        if input_ids.shape[-1] < 2:
            raise ValueError("Causal LM sequences require at least two tokens.")

        labels = batch.get("labels", input_ids)
        if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
            raise ValueError("'labels' must be a tensor matching 'input_ids'.")

        effective_labels = labels[..., 1:].clone()
        target_mask = torch.ones_like(effective_labels, dtype=torch.bool)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError("'attention_mask' must be a tensor.")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("'attention_mask' must match 'input_ids'.")
            target_mask &= attention_mask[..., 1:].to(dtype=torch.bool)

        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            if not isinstance(loss_mask, torch.Tensor):
                raise ValueError("'loss_mask' must be a tensor.")
            if loss_mask.shape != input_ids.shape:
                raise ValueError("'loss_mask' must match 'input_ids'.")
            target_mask &= loss_mask[..., 1:].to(dtype=torch.bool)

        target_mask &= effective_labels.ne(self.ignore_index)
        effective_labels.masked_fill_(~target_mask, self.ignore_index)

        supervised = int(target_mask.sum().item())
        targets = effective_labels.numel()
        counts = TokenCounts(
            sequences=input_ids.numel() // input_ids.shape[-1],
            input_tokens=input_ids.numel(),
            target_tokens=targets,
            supervised_tokens=supervised,
            ignored_tokens=targets - supervised,
        )
        if supervised == 0:
            raise ValueError("Causal LM batch contains no supervised tokens.")

        task_batch = dict(batch)
        task_batch["labels"] = effective_labels
        task_batch["loss_mask"] = target_mask
        return task_batch, counts

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if logits.ndim != labels.ndim + 1:
            raise ValueError("Causal LM logits must add one vocabulary dimension.")
        if logits.shape[:-1] != (*labels.shape[:-1], labels.shape[-1] + 1):
            raise ValueError("Causal LM logits and shifted labels are misaligned.")

        shifted_logits = logits[..., :-1, :].float()
        flat_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1])
        flat_labels = labels.reshape(-1)
        loss_sum = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.ignore_index,
            reduction="sum",
        )

        if self.normalization == "supervised_tokens":
            denominator = loss_mask.sum()
        else:
            denominator = labels.numel() // labels.shape[-1]
        loss = loss_sum / denominator

        z_loss_value = None
        if self.z_loss:
            log_z = torch.logsumexp(shifted_logits, dim=-1)
            z_loss_value = (
                log_z.square().masked_select(loss_mask).sum()
                / loss_mask.sum()
            )
            loss = loss + self.z_loss * z_loss_value

        return loss, z_loss_value


def _extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, Mapping) and "logits" in outputs:
        logits = outputs["logits"]
    elif isinstance(outputs, tuple) and outputs:
        logits = outputs[0]
    else:
        raise TypeError("Model output must expose causal LM logits.")

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model logits must be a torch.Tensor.")
    return logits
