from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from trainlm.config import TrainLMConfig


class TrainLMPreTrainedModel(PreTrainedModel):
    """
    Base class for all TrainLM models.

    Provides Hugging Face integration including:

    - configuration handling
    - serialization
    - weight initialization
    - future gradient checkpointing support

    Architecture-specific implementations belong in subclasses.
    """

    config_class = TrainLMConfig
    base_model_prefix = "model"

    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_flash_attn_2 = False

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize module weights.

        Follows the standard Transformer initialization scheme.
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(
        self,
        module: nn.Module,
        enable: bool = False,
    ) -> None:
        """
        Enable/disable gradient checkpointing.

        Actual modules supporting checkpointing will be introduced
        in later milestones.
        """

        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = enable

    def get_input_embeddings(self) -> Optional[nn.Module]:
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def set_input_embeddings(
        self,
        value: nn.Module,
    ) -> None:
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_output_embeddings(self) -> Optional[nn.Module]:
        """
        Decoder-only base model has no output embeddings.

        TrainLMForCausalLM overrides this.
        """
        return None

    def set_output_embeddings(
        self,
        new_embeddings: nn.Module,
    ) -> None:
        """
        TrainLMForCausalLM overrides this.
        """
        raise NotImplementedError
