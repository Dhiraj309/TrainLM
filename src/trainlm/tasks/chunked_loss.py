"""Reference memory-bounded linear causal cross-entropy."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def chunked_linear_causal_cross_entropy(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    chunk_size: int = 2048,
    ignore_index: int = -100,
    z_loss: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute shifted causal CE without materializing full-sequence logits.

    Projection and reductions are performed in FP32. Chunks partition flattened
    target tokens, so peak logits storage is bounded by ``chunk_size * vocab``.
    The returned z-loss is unscaled; ``z_loss`` is applied only to total loss.
    """

    if hidden_states.ndim < 3:
        raise ValueError("hidden_states must have shape [..., sequence, hidden].")
    if labels.shape != hidden_states.shape[:-1]:
        raise ValueError("labels must match the hidden-state batch/sequence shape.")
    if output_weight.ndim != 2 or output_weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError("output_weight must have shape [vocabulary, hidden].")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != output_weight.shape[0]):
        raise ValueError("bias must have shape [vocabulary].")
    if hidden_states.shape[-2] < 2:
        raise ValueError("Causal loss requires at least two sequence positions.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(z_loss, (int, float)) or not math.isfinite(z_loss) or z_loss < 0:
        raise ValueError("z_loss must be finite and non-negative.")

    shifted_hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[..., 1:].reshape(-1)
    if loss_mask is not None:
        if loss_mask.shape == labels.shape:
            loss_mask = loss_mask[..., 1:]
        elif loss_mask.shape != labels[..., 1:].shape:
            raise ValueError("loss_mask must match labels or shifted labels.")
        shifted_labels = torch.where(
            loss_mask.reshape(-1).to(dtype=torch.bool),
            shifted_labels,
            torch.full_like(shifted_labels, ignore_index),
        )
    supervised = shifted_labels.ne(ignore_index)
    denominator = supervised.sum()
    if not bool(denominator.item()):
        raise ValueError("Causal loss contains no supervised target tokens.")

    loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
    z_sum = hidden_states.new_zeros((), dtype=torch.float32)
    weight_fp32 = output_weight.float()
    bias_fp32 = bias.float() if bias is not None else None
    for start in range(0, shifted_hidden.shape[0], chunk_size):
        stop = min(start + chunk_size, shifted_hidden.shape[0])
        chunk_labels = shifted_labels[start:stop]
        logits = F.linear(shifted_hidden[start:stop].float(), weight_fp32, bias_fp32)
        loss_sum = loss_sum + F.cross_entropy(
            logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
        )
        if z_loss:
            active = chunk_labels.ne(ignore_index)
            z_sum = z_sum + torch.where(
                active, torch.logsumexp(logits, dim=-1).square(), 0.0
            ).sum()

    loss = loss_sum / denominator
    z_loss_value = z_sum / denominator if z_loss else None
    if z_loss_value is not None:
        loss = loss + z_loss * z_loss_value
    return loss, z_loss_value


__all__ = ["chunked_linear_causal_cross_entropy"]
