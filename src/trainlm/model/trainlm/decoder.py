from __future__ import annotations

import torch
import torch.nn as nn

from trainlm.config import TrainLMConfig

from .attention import TrainLMAttention
from .mlp import TrainLMSwiGLU


class TrainLMDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    The layer applies the following sequence of operations:

        hidden_states = hidden_states + Attention(RMSNorm(hidden_states))
        hidden_states = hidden_states + SwiGLU(RMSNorm(hidden_states))
    """

    input_layernorm: nn.RMSNorm
    self_attn: TrainLMAttention
    post_attention_layernorm: nn.RMSNorm
    mlp: TrainLMSwiGLU

    def __init__(
        self,
        config: TrainLMConfig,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.self_attn = TrainLMAttention(
            config,
        )

        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.mlp = TrainLMSwiGLU(
            config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply a decoder layer.

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
            torch.nn.functional.scaled_dot_product_attention().

        Returns
        -------
        torch.Tensor
            Shape:
                (batch_size, sequence_length, hidden_size)
        """

        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask,
        )

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states),
        )

        return hidden_states
