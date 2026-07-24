from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainlm.config import TrainLMConfig


class TrainLMSwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Applies the following transformation:

        down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(
        self,
        config: TrainLMConfig,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
        )

        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
        )

        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the SwiGLU feed-forward network.

        Parameters
        ----------
        hidden_states:
            Shape:
                (batch_size, sequence_length, hidden_size)

        Returns
        -------
        torch.Tensor
            Shape:
                (batch_size, sequence_length, hidden_size)
        """

        gate = F.silu(
            self.gate_proj(hidden_states)
        )

        up = self.up_proj(
            hidden_states
        )

        hidden_states = gate * up

        hidden_states = self.down_proj(
            hidden_states
        )

        return hidden_states
