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
    - Recomputes inverse frequencies from the configuration each forward
      instead of storing them as a buffer.
    """

    def __init__(
        self,
        config: TrainLMConfig,
    ) -> None:
        super().__init__()

        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

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

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=x.device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )

        inv_freq = inv_freq.unsqueeze(0).unsqueeze(-1)

        position_ids = position_ids.to(
            device=x.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        freqs = torch.matmul(inv_freq, position_ids)
        freqs = freqs.transpose(1, 2)

        emb = torch.cat(
            (freqs, freqs),
            dim=-1,
        ).unsqueeze(1)

        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)

        return cos, sin
