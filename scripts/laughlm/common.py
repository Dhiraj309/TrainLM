"""Shared configuration and checkpoint helpers for LaughLM 135M stages."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

DEFAULT_SHARD_REPO = "Dhiraj309/laughlm-135m-shards-v2"
DEFAULT_OUTPUT_REPO = "Dhiraj309/laughlm-135m-v2"
DEFAULT_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
VOCAB_SIZE = 49152
EOS_TOKEN_ID = 0
BOS_TOKEN_ID = 0


def build_135m_config(
    *,
    max_position_embeddings: int = 4096,
    rope_theta: float = 100000.0,
) -> Any:
    """Return the locked 30-layer deep-narrow 135M SmolLM2-compatible config."""
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.041666666666666664,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=True,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=EOS_TOKEN_ID,
        rope_theta=rope_theta,
    )


def load_or_init_135m_model(
    *,
    init_checkpoint: str | Path | None = None,
    max_position_embeddings: int = 4096,
    rope_theta: float = 100000.0,
) -> Any:
    """Initialize a fresh 135M model or load weights from a prior checkpoint."""
    from transformers import AutoModelForCausalLM

    if init_checkpoint is not None:
        ckpt_path = Path(init_checkpoint)
        model_dir = ckpt_path / "model" if (ckpt_path / "model").is_dir() else ckpt_path
        model = AutoModelForCausalLM.from_pretrained(str(model_dir))
        model.config.max_position_embeddings = max(
            int(getattr(model.config, "max_position_embeddings", 0) or 0),
            max_position_embeddings,
        )
        model.config.rope_theta = float(rope_theta)
        model.config.vocab_size = VOCAB_SIZE
        model.config.eos_token_id = EOS_TOKEN_ID
        model.config.use_cache = False
        return model

    cfg = build_135m_config(
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
    )
    return AutoModelForCausalLM.from_config(cfg)


def load_tokenizer(tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> Any:
    """Load the SmolLM2 tokenizer with verified vocab_size=49152 and eos_token_id=0."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if getattr(tok, "eos_token_id", None) is None:
        tok.eos_token_id = EOS_TOKEN_ID
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    if len(tok) != VOCAB_SIZE:
        raise ValueError(f"Expected vocab_size={VOCAB_SIZE}, got {len(tok)}.")
    if int(tok.eos_token_id) != EOS_TOKEN_ID:
        raise ValueError(f"Expected eos_token_id={EOS_TOKEN_ID}, got {tok.eos_token_id}.")
    return tok


def atomic_save_model_snapshot(model: Any, tokenizer: Any, target_dir: Path) -> Path:
    """Save model+tokenizer to a `.tmp` directory and atomically rename into place."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target_dir.parent / f".{target_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        if target_dir.exists():
            backup = target_dir.parent / f".{target_dir.name}.bak.{os.getpid()}"
            os.replace(target_dir, backup)
            os.replace(tmp_dir, target_dir)
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return target_dir
