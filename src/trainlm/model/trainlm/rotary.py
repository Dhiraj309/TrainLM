from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from trainlm.config import TrainLMConfig


class TrainLMRotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    This module computes the cosine and sine tensors required to apply
    Rotary Position Embeddings to query and key tensors.

    Notes
    -----
    - Implements the original RoPE formulation.
    - Computes trigonometric functions in FP32 for numerical stability.
    - Returns outputs cast back to the input dtype.
    - Stores only the inverse frequencies as a non-persistent buffer.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        config: TrainLMConfig,
    ) -> None:
        super().__init__()

        head_dim = config.head_dim

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cosine and sine tensors.

        Parameters
        ----------
        x:
            Reference tensor used only for device and dtype.

            Shape:
                (..., head_dim)

        position_ids:
            Position indices.

            Shape:
                (batch_size, sequence_length)

        Returns
        -------
        cos:
            Shape:
                (batch_size, 1, sequence_length, head_dim)

        sin:
            Shape:
                (batch_size, 1, sequence_length, head_dim)
        """

        inv_freq = self.inv_freq.float().unsqueeze(0).unsqueeze(-1)

        position_ids = position_ids.float().unsqueeze(1)

        freqs = torch.matmul(inv_freq, position_ids)

        freqs = freqs.transpose(1, 2)

        emb = torch.cat(
            (freqs, freqs),
            dim=-1,
        ).unsqueeze(1)

        cos, sin = emb.cos(), emb.sin()

        return (
            cos.to(dtype=x.dtype),
            sin.to(dtype=x.dtype),
        )
