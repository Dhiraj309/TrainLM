"""Tiny official Transformers configurations for the dense-AR V1 matrix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from transformers import (
    BloomConfig,
    FalconConfig,
    GemmaConfig,
    GPT2Config,
    GPTNeoXConfig,
    LlamaConfig,
    MistralConfig,
    OPTConfig,
    PhiConfig,
    PretrainedConfig,
    Qwen2Config,
)

from trainlm.config import ModelSourceConfig


@dataclass(frozen=True, slots=True)
class DenseARFixture:
    name: str
    model_type: str
    config_class: type[PretrainedConfig]
    overrides: Mapping[str, Any]

    def config(self, tied: bool) -> PretrainedConfig:
        return self.config_class(
            **self.overrides,
            use_cache=False,
            tie_word_embeddings=tied,
        )

    def source(self, tied: bool) -> ModelSourceConfig:
        return ModelSourceConfig(
            provider="huggingface",
            initialization="config",
            model_type=self.model_type,
            dtype="float32",
            config_overrides={
                **self.overrides,
                "use_cache": False,
                "tie_word_embeddings": tied,
            },
        )


DENSE_AR_FIXTURES = (
    DenseARFixture(
        "gpt2", "gpt2", GPT2Config,
        {
            "vocab_size": 32, "n_positions": 8, "n_embd": 8,
            "n_layer": 1, "n_head": 2, "resid_pdrop": 0.0,
            "embd_pdrop": 0.0, "attn_pdrop": 0.0,
        },
    ),
    DenseARFixture(
        "opt", "opt", OPTConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "word_embed_proj_dim": 8,
            "ffn_dim": 16, "num_hidden_layers": 1,
            "num_attention_heads": 2, "max_position_embeddings": 8,
            "dropout": 0.0, "attention_dropout": 0.0,
        },
    ),
    DenseARFixture(
        "gpt_neox", "gpt_neox", GPTNeoXConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "max_position_embeddings": 8, "hidden_dropout": 0.0,
            "attention_dropout": 0.0, "rotary_pct": 0.5,
        },
    ),
    DenseARFixture(
        "bloom", "bloom", BloomConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "n_layer": 1,
            "n_head": 2, "hidden_dropout": 0.0, "attention_dropout": 0.0,
        },
    ),
    DenseARFixture(
        "falcon", "falcon", FalconConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_kv_heads": 1,
            "ffn_hidden_size": 16, "max_position_embeddings": 8,
            "hidden_dropout": 0.0, "attention_dropout": 0.0,
            "alibi": True, "multi_query": True,
        },
    ),
    DenseARFixture(
        "phi", "phi", PhiConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 8,
            "resid_pdrop": 0.0, "embd_pdrop": 0.0,
            "attention_dropout": 0.0, "partial_rotary_factor": 0.5,
        },
    ),
    DenseARFixture(
        "llama", "llama", LlamaConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 8,
            "attention_dropout": 0.0,
        },
    ),
    DenseARFixture(
        "mistral", "mistral", MistralConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 8,
            "sliding_window": 8, "attention_dropout": 0.0,
        },
    ),
    DenseARFixture(
        "qwen2", "qwen2", Qwen2Config,
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 8,
            "attention_dropout": 0.0,
        },
    ),
    DenseARFixture(
        "gemma", "gemma", GemmaConfig,
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 8, "attention_dropout": 0.0,
        },
    ),
)
