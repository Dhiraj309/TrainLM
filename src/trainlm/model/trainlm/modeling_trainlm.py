from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from trainlm.config import TrainLMConfig

from .decoder import TrainLMDecoderLayer


def _assert_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Raise an error if a floating-point tensor contains non-finite values.
    """
    if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name}.")


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
    _no_split_modules: list[str] = ["TrainLMDecoderLayer"]
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


class TrainLMModel(TrainLMPreTrainedModel):
    """
    Bare TrainLM decoder.

    This class contains the decoder backbone without a language modeling
    head. It is analogous to `LlamaModel` in the Transformers library.

    Returns hidden states only.
    """

    embed_tokens: nn.Embedding
    layers: nn.ModuleList
    norm: nn.RMSNorm

    def __init__(self, config: TrainLMConfig):
        super().__init__(config)

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            TrainLMDecoderLayer(config)
            for _ in range(self.num_hidden_layers)
        )

        self.norm = nn.RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False

        self.post_init()

    #
    # Embedding API
    #

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(
        self,
        value: nn.Module,
    ) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass.

        Transformer layers will be implemented in M3.
        """

        del (
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            cache_position,
            kwargs,
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "Only one of 'input_ids' or 'inputs_embeds' may be provided."
            )

        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "Either 'input_ids' or 'inputs_embeds' must be provided."
            )

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        _assert_finite(hidden_states, "model.embed_tokens")

        batch_size, sequence_length = hidden_states.shape[:2]

        if position_ids is None:
            position_ids = (
                torch.arange(
                    sequence_length,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, sequence_length)
            )

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids,
                attention_mask,
            )
            _assert_finite(hidden_states, f"model.layer_{layer_idx}")

        hidden_states = self.norm(hidden_states)
        _assert_finite(hidden_states, "model.norm")

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class TrainLMForCausalLM(TrainLMPreTrainedModel):
    """
    TrainLM Model with a causal language modeling head.

    This class wraps TrainLMModel with an output projection suitable for
    autoregressive language modeling.
    """

    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight",
    }

    model: TrainLMModel
    lm_head: nn.Linear

    def __init__(self, config: TrainLMConfig):
        super().__init__(config)

        self.model = TrainLMModel(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()

    #
    # Backbone
    #

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    #
    # Output head
    #

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        _assert_finite(hidden_states, "causal_lm.hidden_states")

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        _assert_finite(logits, "causal_lm.logits")

        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]

            if loss is not None:
                return (loss,) + output

            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
