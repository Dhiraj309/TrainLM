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
        *,
        is_first_layer: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.self_attn = TrainLMAttention(
            config,
            is_first_layer=is_first_layer,
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
        segment_ids: torch.Tensor | None = None,
        v_anchor: torch.Tensor | None = None,
        return_v_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        segment_ids:
            Optional packed document boundary IDs of shape
            ``(batch_size, sequence_length)``.
        v_anchor:
            Optional first-layer value tensor for ``V_1`` residual mixing.
        return_v_state:
            If True, return ``(hidden_states, raw_value_states)``.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Updated hidden states (and optional raw ``v_proj`` states).
        """

        attn_res = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask,
            segment_ids=segment_ids,
            v_anchor=v_anchor,
            return_v_state=return_v_state,
        )
        if return_v_state:
            attn_out, v_state = attn_res
        else:
            attn_out = attn_res
            v_state = None

        hidden_states = hidden_states + attn_out

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states),
        )

        if return_v_state:
            assert v_state is not None
            return hidden_states, v_state
        return hidden_states
