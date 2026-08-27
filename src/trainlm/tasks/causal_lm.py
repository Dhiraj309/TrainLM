"""Canonical dense autoregressive causal-language-model task."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from trainlm.model.dispatch import ForwardBatchDispatcher
from trainlm.model.outputs import normalize_causal_lm_output
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
        loss_implementation: Literal["auto", "causal_lm", "model"] = "auto",
    ) -> None:
        if normalization not in {"supervised_tokens", "batch"}:
            raise ValueError(f"Unsupported loss normalization: {normalization}")
        if z_loss < 0:
            raise ValueError("z_loss must be non-negative.")
        if loss_implementation not in {"auto", "causal_lm", "model"}:
            raise ValueError(
                f"Unsupported loss implementation: {loss_implementation}"
            )
        self.ignore_index = ignore_index
        self.normalization = normalization
        self.z_loss = z_loss
        self.loss_implementation = loss_implementation
        if loss_implementation == "model" and not self._model_loss_compatible:
            raise ValueError(
                "Model loss requires ignore_index=-100, supervised-token "
                "normalization, and z_loss=0."
            )
        self._dispatcher_model: nn.Module | None = None
        self._dispatcher: ForwardBatchDispatcher | None = None

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
        return self.aggregate_evaluation_stream(results)

    def aggregate_evaluation_stream(
        self,
        results: Iterable[TaskResult],
    ) -> dict[str, float]:

        weight_name = (
            "supervised_tokens"
            if self.normalization == "supervised_tokens"
            else "sequences"
        )
        weighted_loss: torch.Tensor | None = None
        total_weight = 0
        for result in results:
            weight = getattr(result.tokens, weight_name)
            contribution = result.loss.detach() * weight
            weighted_loss = (
                contribution
                if weighted_loss is None
                else weighted_loss + contribution
            )
            total_weight += weight

        if total_weight == 0 or weighted_loss is None:
            raise ValueError("Evaluation contains no normalization units.")
        eval_loss = (weighted_loss / total_weight).item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        return {
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity,
        }

    def _step(
        self,
        model: nn.Module,
        batch: Any,
        backend: ExecutionBackend,
    ) -> TaskResult:
        task_batch, counts = self._prepare_task_batch(batch)
        task_batch = backend.prepare_batch(task_batch)

        model_labels = task_batch["labels"]
        labels = model_labels[..., 1:]
        loss_mask = task_batch["loss_mask"]
        model_inputs = {
            key: value
            for key, value in task_batch.items()
            if key not in {"labels", "loss_mask"}
        }
        if self._dispatcher is None or self._dispatcher_model is not model:
            self._dispatcher = ForwardBatchDispatcher.from_model(model)
            self._dispatcher_model = model
        labels_supported = "labels" in self._dispatcher.signature.keyword_parameters
        request_model_loss = (
            self.loss_implementation in {"auto", "model"}
            and self._model_loss_compatible
        )
        if request_model_loss and labels_supported:
            model_inputs["labels"] = model_labels
        elif self.loss_implementation == "model":
            raise TypeError(
                "Model-owned loss was requested but forward does not declare labels."
            )
        dispatch = self._dispatcher.dispatch(model_inputs)

        with backend.autocast():
            outputs = model(**dispatch.inputs)
            normalized_output = normalize_causal_lm_output(outputs)
            if request_model_loss and labels_supported:
                if normalized_output.loss is None:
                    if self.loss_implementation == "model":
                        raise TypeError(
                            "Model-owned loss was requested but output has no loss."
                        )
                    loss, z_loss_value = self._loss(
                        normalized_output.logits, labels, loss_mask
                    )
                    loss_source = "trainlm_cross_entropy"
                else:
                    loss = normalized_output.loss
                    z_loss_value = None
                    loss_source = "model"
            else:
                loss, z_loss_value = self._loss(
                    normalized_output.logits, labels, loss_mask
                )
                loss_source = "trainlm_cross_entropy"

        metrics: dict[str, torch.Tensor | float] = {}
        if z_loss_value is not None:
            metrics["z_loss"] = z_loss_value.detach()

        return TaskResult(
            loss=loss,
            tokens=counts,
            metrics=metrics,
            loss_source=loss_source,
        )

    @property
    def _model_loss_compatible(self) -> bool:
        return (
            self.ignore_index == -100
            and self.normalization == "supervised_tokens"
            and self.z_loss == 0.0
        )

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

        model_labels = labels.clone()
        effective_labels = model_labels[..., 1:]
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
        task_batch["labels"] = model_labels
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
