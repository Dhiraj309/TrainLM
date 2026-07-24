from __future__ import annotations

import torch
import torch.nn as nn

from trainlm.config import TrainLMConfig

from .decoder import TrainLMDecoderLayer


class TrainLMModel(nn.Module):
    """
    Decoder-only Transformer backbone.

    The model consists of:

        Token Embedding
            ↓
        N × Transformer Decoder Layers
            ↓
        Final RMSNorm

    Returns the final hidden states.
    """

    embed_tokens: nn.Embedding
    layers: nn.ModuleList
    norm: nn.RMSNorm

    def __init__(
        self,
        config: TrainLMConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.layers = nn.ModuleList(
            TrainLMDecoderLayer(config)
            for _ in range(self.num_hidden_layers)
        )

        self.norm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the decoder-only Transformer.

        Parameters
        ----------
        input_ids:
            Shape:
                (batch_size, sequence_length)

        attention_mask:
            Optional attention mask accepted by
            torch.nn.functional.scaled_dot_product_attention().

        Returns
        -------
        torch.Tensor
            Final hidden states of shape

                (batch_size, sequence_length, hidden_size)
        """

        batch_size, sequence_length = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids,
                attention_mask,
            )

        hidden_states = self.norm(
            hidden_states,
        )

        return hidden_states
