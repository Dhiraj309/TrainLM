"""LaughLM-Loop: Looped-Depth Transformer (Prelude + Shared Core + Coda)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from trainlm.config.configuration_trainlm import LaughLMLoopConfig
from trainlm.kernels.pallas.recurrent_norm import fused_recurrent_inject_rmsnorm

from .decoder import TrainLMDecoderLayer
from .modeling_trainlm import TrainLMPreTrainedModel, _assert_finite


class LaughLMLoopModel(TrainLMPreTrainedModel):
    """Looped-depth transformer backbone with anchor injection and truncated BPTT."""

    config_class = LaughLMLoopConfig

    def __init__(self, config: LaughLMLoopConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.prelude_layers = nn.ModuleList(
            [TrainLMDecoderLayer(config) for _ in range(config.num_prelude_layers)]
        )
        self.core_layers = nn.ModuleList(
            [TrainLMDecoderLayer(config) for _ in range(config.num_core_layers)]
        )
        self.coda_layers = nn.ModuleList(
            [TrainLMDecoderLayer(config) for _ in range(config.num_coda_layers)]
        )
        self.recurrent_gate = nn.Parameter(torch.zeros(config.hidden_size))
        self.recurrent_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _run_decoder_layer(
        self,
        layer: TrainLMDecoderLayer,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return self._gradient_checkpointing_func(
                layer.__call__,
                hidden_states,
                position_ids,
                attention_mask,
            )
        return layer(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    def _resolve_active_loops(self, num_loops: int | None) -> int:
        if num_loops is not None:
            loops = int(num_loops)
            if loops <= 0:
                raise ValueError(f"num_loops must be > 0, got {loops}.")
            return loops
        is_compiling_fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
        compiling = bool(callable(is_compiling_fn) and is_compiling_fn())
        if (
            not compiling
            and self.training
            and self.config.train_min_loops < self.config.train_max_loops
        ):
            # In eager mode draw uniformly from [train_min_loops, train_max_loops].
            return int(
                torch.randint(
                    self.config.train_min_loops,
                    self.config.train_max_loops + 1,
                    (1,),
                ).item()
            )
        # Strictly static loop bound inside compiled graphs (no graph breaks).
        return int(self.config.num_loop_iterations)

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
        cache_position=None,
        num_loops: Optional[int] = None,
        **kwargs,
    ):
        del past_key_values, use_cache, cache_position, kwargs
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if output_attentions:
            raise NotImplementedError("LaughLMLoopModel does not support output_attentions=True.")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds."
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        _assert_finite(inputs_embeds, "loop.embeddings")

        batch_size, sequence_length, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = (
                torch.arange(
                    sequence_length,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                .unsqueeze(0)
                .expand(batch_size, sequence_length)
            )

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # 1. Prelude layers establish initial latent representation.
        for layer in self.prelude_layers:
            hidden_states = self._run_decoder_layer(
                layer, hidden_states, position_ids, attention_mask
            )
            _assert_finite(hidden_states, "loop.prelude")
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = (*all_hidden_states, hidden_states)

        # Anchor residual injected before each recurrent iteration.
        h_0 = hidden_states

        # 2. Shared recurrent core with truncated BPTT bound.
        active_loops = self._resolve_active_loops(num_loops)
        k_bptt = int(self.config.truncated_bptt_steps)
        detach_cutoff = max(0, active_loops - k_bptt)
        use_pallas_rec_norm = bool(
            getattr(self.config, "use_pallas_recurrent_norm", True)
        )

        for step_idx in range(active_loops):
            if self.training and 0 < step_idx <= detach_cutoff:
                hidden_states = hidden_states.detach()

            hidden_states = fused_recurrent_inject_rmsnorm(
                hidden_states,
                h_0,
                self.recurrent_gate,
                self.recurrent_norm.weight,
                eps=float(self.recurrent_norm.eps),
                use_pallas=use_pallas_rec_norm,
            )
            for layer in self.core_layers:
                hidden_states = self._run_decoder_layer(
                    layer, hidden_states, position_ids, attention_mask
                )
                _assert_finite(hidden_states, "loop.core")
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = (*all_hidden_states, hidden_states)

        # 3. Coda layers decode latent trajectory to vocabulary space.
        for layer in self.coda_layers:
            hidden_states = self._run_decoder_layer(
                layer, hidden_states, position_ids, attention_mask
            )
            _assert_finite(hidden_states, "loop.coda")
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = (*all_hidden_states, hidden_states)

        hidden_states = self.norm(hidden_states)
        _assert_finite(hidden_states, "loop.norm")

        if not return_dict:
            return tuple(
                v for v in (hidden_states, None, all_hidden_states, None) if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class LaughLMLoopForCausalLM(TrainLMPreTrainedModel):
    """Causal LM wrapper around ``LaughLMLoopModel`` compatible with ``TrainLMTrainer``."""

    config_class = LaughLMLoopConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LaughLMLoopConfig):
        super().__init__(config)
        self.model = LaughLMLoopModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        num_loops: Optional[int] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
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
            return_dict=True,
            cache_position=cache_position,
            num_loops=num_loops,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        _assert_finite(hidden_states, "loop_causal_lm.hidden_states")
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + tuple(
                v for v in (outputs.hidden_states, outputs.attentions) if v is not None
            )
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _register_laughlm_loop_model() -> None:
    try:
        from transformers import AutoModelForCausalLM

        AutoModelForCausalLM.register(
            LaughLMLoopConfig, LaughLMLoopForCausalLM, exist_ok=True
        )
    except Exception:
        pass


_register_laughlm_loop_model()

__all__ = [
    "LaughLMLoopForCausalLM",
    "LaughLMLoopModel",
]
