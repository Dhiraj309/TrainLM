"""Tiny official Transformers configurations for the dense-AR V1 matrix."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class DenseARFixture:
    name: str
    config_factory: Callable[[bool], PretrainedConfig]


def _gpt2(tied: bool) -> PretrainedConfig:
    return GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _opt(tied: bool) -> PretrainedConfig:
    return OPTConfig(
        vocab_size=32,
        hidden_size=8,
        word_embed_proj_dim=8,
        ffn_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _gpt_neox(tied: bool) -> PretrainedConfig:
    return GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        rotary_pct=0.5,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _bloom(tied: bool) -> PretrainedConfig:
    return BloomConfig(
        vocab_size=32,
        hidden_size=8,
        n_layer=1,
        n_head=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _falcon(tied: bool) -> PretrainedConfig:
    return FalconConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        ffn_hidden_size=16,
        max_position_embeddings=8,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        alibi=True,
        multi_query=True,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _phi(tied: bool) -> PretrainedConfig:
    return PhiConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.5,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _llama(tied: bool) -> PretrainedConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _mistral(tied: bool) -> PretrainedConfig:
    return MistralConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        sliding_window=8,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _qwen2(tied: bool) -> PretrainedConfig:
    return Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


def _gemma(tied: bool) -> PretrainedConfig:
    return GemmaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=8,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=tied,
    )


DENSE_AR_FIXTURES = (
    DenseARFixture("gpt2", _gpt2),
    DenseARFixture("opt", _opt),
    DenseARFixture("gpt_neox", _gpt_neox),
    DenseARFixture("bloom", _bloom),
    DenseARFixture("falcon", _falcon),
    DenseARFixture("phi", _phi),
    DenseARFixture("llama", _llama),
    DenseARFixture("mistral", _mistral),
    DenseARFixture("qwen2", _qwen2),
    DenseARFixture("gemma", _gemma),
)
