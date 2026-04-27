from typing import Tuple, Optional, Dict, Any

import jax.numpy as jnp

from transformers import (
    LlamaConfig,
    GPT2Config,
    OPTConfig,
)

from src.registry.model_registry import get_model_class


# ------------------------------------------------------------
# Normalized Field Mapping
# ------------------------------------------------------------

def _common_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vocab_size": cfg["vocab_size"],
    }


# ------------------------------------------------------------
# HF Config Builders
# ------------------------------------------------------------

def _build_llama_config(cfg: Dict[str, Any]) -> LlamaConfig:
    return LlamaConfig(
        **_common_fields(cfg),

        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads"),
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],

        # modern fields
        hidden_act="silu",
        rms_norm_eps=1e-5,

        # rope
        rope_theta=cfg.get("rope_theta", 10000.0),
        rope_scaling=cfg.get("rope_scaling"),

        # embeddings
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),

        # dropout (safe defaults)
        attention_dropout=0.0,
    )


def _build_gpt2_config(cfg: Dict[str, Any]) -> GPT2Config:
    return GPT2Config(
        **_common_fields(cfg),

        n_embd=cfg["hidden_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],

        n_positions=cfg["max_position_embeddings"],
        n_ctx=cfg["max_position_embeddings"],

        # defaults
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )


def _build_opt_config(cfg: Dict[str, Any]) -> OPTConfig:
    return OPTConfig(
        **_common_fields(cfg),

        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],

        ffn_dim=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],

        # defaults
        activation_function="relu",
        dropout=0.1,
        attention_dropout=0.0,
    )


# ------------------------------------------------------------
# Config Dispatcher
# ------------------------------------------------------------

def build_hf_config(model_type: str, cfg: Dict[str, Any]):
    if model_type == "llama":
        return _build_llama_config(cfg)
    elif model_type == "gpt2":
        return _build_gpt2_config(cfg)
    elif model_type == "opt":
        return _build_opt_config(cfg)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ------------------------------------------------------------
# DType Utility
# ------------------------------------------------------------

def get_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return jnp.bfloat16
    elif dtype_str == "float16":
        return jnp.float16
    else:
        return jnp.float32


# ------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------

def build_model(
    model_cfg,
    parallel_cfg,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[object, Dict]:

    model_type = model_cfg.model_type

    cfg_dict = model_cfg.model_dump()

    # --------------------------------------------------------
    # HF Config
    # --------------------------------------------------------

    hf_config = build_hf_config(model_type, cfg_dict)

    # --------------------------------------------------------
    # DType
    # --------------------------------------------------------

    dtype = get_dtype(parallel_cfg.compute_dtype)

    # --------------------------------------------------------
    # Model class
    # --------------------------------------------------------

    model_cls = get_model_class(model_type)

    # --------------------------------------------------------
    # Load / Init
    # --------------------------------------------------------

    if checkpoint_dir is not None:
        model = model_cls.from_pretrained(
            checkpoint_dir,
            dtype=dtype,
        )
        params = model.params

    else:
        model = model_cls(
            hf_config,
            dtype=dtype,
        )
        params = model.params

    return model, params
