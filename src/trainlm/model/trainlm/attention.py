from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainlm.config import TrainLMConfig
from .rotary import TrainLMRotaryEmbedding


def _repeat_kv(
    hidden_states: torch.Tensor,
    num_repeats: int,
) -> torch.Tensor:
    """
    Repeat key/value heads for Grouped Query Attention (GQA).

    Parameters
    ----------
    hidden_states:
        Tensor of shape
        (batch_size, num_key_value_heads, sequence_length, head_dim)

    num_repeats:
        Number of attention head groups per key/value head.

    Returns
    -------
    torch.Tensor
        Tensor of shape
        (batch_size, num_attention_heads, sequence_length, head_dim)
    """

    batch_size, num_key_value_heads, sequence_length, head_dim = (
        hidden_states.shape
    )

    if num_repeats == 1:
        return hidden_states

    hidden_states = hidden_states[
        :,
        :,
        None,
        :,
        :,
    ].expand(
        batch_size,
        num_key_value_heads,
        num_repeats,
        sequence_length,
        head_dim,
    )

    return hidden_states.reshape(
        batch_size,
        num_key_value_heads * num_repeats,
        sequence_length,
        head_dim,
    )


def _rotate_half(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate the last dimension by half.

    [x1, x2] -> [-x2, x1]
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat(
        (-x2, x1),
        dim=-1,
    )


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Parameters
    ----------
    query:
        (batch_size, num_attention_heads, sequence_length, head_dim)

    key:
        (batch_size, num_key_value_heads, sequence_length, head_dim)

    cos:
        (batch_size, 1, sequence_length, head_dim)

    sin:
        (batch_size, 1, sequence_length, head_dim)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotary-embedded query and key tensors.
    """

    query = (
        query * cos
        + _rotate_half(query) * sin
    )

    key = (
        key * cos
        + _rotate_half(key) * sin
    )

    return query, key


class TrainLMAttention(nn.Module):
    """
    Multi-head self-attention using Grouped Query Attention (GQA).

    Features
    --------
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - PyTorch Scaled Dot Product Attention (SDPA)
    """

    rotary_emb: TrainLMRotaryEmbedding

    def __init__(
        self,
        config: TrainLMConfig,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.attention_dropout = config.attention_dropout

        q_proj_size = (
            self.num_attention_heads
            * self.head_dim
        )

        kv_proj_size = (
            self.num_key_value_heads
            * self.head_dim
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            q_proj_size,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            kv_proj_size,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            kv_proj_size,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            q_proj_size,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.rotary_emb = TrainLMRotaryEmbedding(
            config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply self-attention.

        Parameters
        ----------
        hidden_states:
            Shape:
                (batch_size, sequence_length, hidden_size)

        position_ids:
            Shape:
                (batch_size, sequence_length)

        attention_mask:
            Optional attention mask accepted by
            ``torch.nn.functional.scaled_dot_product_attention``.

            Supported shapes include broadcastable boolean masks
            and additive floating-point masks.
        """

        batch_size, sequence_length, _ = (
            hidden_states.shape
        )

        query_states = self.q_proj(
            hidden_states,
        )

        key_states = self.k_proj(
            hidden_states,
        )

        value_states = self.v_proj(
            hidden_states,
        )

        query_states = query_states.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(
            query_states,
            position_ids,
        )

        query_states, key_states = _apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        key_states = _repeat_kv(
            key_states,
            self.num_key_value_groups,
        )

        value_states = _repeat_kv(
            value_states,
            self.num_key_value_groups,
        )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=(
                self.attention_dropout
                if self.training
                else 0.0
            ),
            is_causal=attention_mask is None,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(
                batch_size,
                sequence_length,
                self.hidden_size,
            )
        )

        attn_output = self.o_proj(
            attn_output,
        )

        return attn_output
