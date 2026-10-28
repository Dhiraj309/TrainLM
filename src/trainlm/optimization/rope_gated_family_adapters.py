"""Explicit mappings for RoPE gated-MLP dense causal model families."""

from __future__ import annotations

from .adapters import AdapterSpec, ModelAdapterRegistry, PackageVersionGuard
from .dense_family_adapters import DenseCausalFamilyMapping


_TRANSFORMERS_GUARD = PackageVersionGuard("transformers", ("5.15.0",))


def _mapping(
    adapter_id: str,
    model_class: str,
    config_class: str,
    *,
    position: str,
    mlp: str,
    operations: tuple[str, ...],
) -> DenseCausalFamilyMapping:
    return DenseCausalFamilyMapping(
        adapter=AdapterSpec(
            adapter_id=adapter_id,
            model_classes=(model_class,),
            config_classes=(config_class,),
            capability_kinds=(
                ("attention", "gqa"),
                ("position", position),
                ("normalization", "rms_norm"),
                ("mlp", mlp),
                ("residual", "serial"),
                ("projections", "separate_qkv"),
                ("lm_head", "linear"),
            ),
            version_guards=(_TRANSFORMERS_GUARD,),
        ),
        operations=operations,
    )


LLAMA_MAPPING = _mapping(
    "huggingface.llama.rope-swiglu-dense",
    "LlamaForCausalLM",
    "LlamaConfig",
    position="rope",
    mlp="swiglu",
    operations=(
        "grouped_causal_attention",
        "rope_position",
        "qkv_layout",
        "swiglu_mlp",
        "linear_causal_loss",
    ),
)

MISTRAL_MAPPING = _mapping(
    "huggingface.mistral.sliding-rope-swiglu-dense",
    "MistralForCausalLM",
    "MistralConfig",
    position="sliding_window_rope",
    mlp="swiglu",
    operations=(
        "grouped_causal_attention",
        "sliding_window",
        "rope_position",
        "qkv_layout",
        "swiglu_mlp",
        "linear_causal_loss",
    ),
)

QWEN2_MAPPING = _mapping(
    "huggingface.qwen2.rope-swiglu-qkv-bias-dense",
    "Qwen2ForCausalLM",
    "Qwen2Config",
    position="rope",
    mlp="swiglu",
    operations=(
        "grouped_causal_attention",
        "rope_position",
        "biased_qkv_layout",
        "swiglu_mlp",
        "linear_causal_loss",
    ),
)

GEMMA_MAPPING = _mapping(
    "huggingface.gemma.rope-geglu-scaled-embedding-dense",
    "GemmaForCausalLM",
    "GemmaConfig",
    position="rope",
    mlp="geglu_tanh",
    operations=(
        "grouped_causal_attention",
        "rope_position",
        "qkv_layout",
        "geglu_mlp",
        "scaled_embedding",
        "linear_causal_loss",
    ),
)

GEMMA2_MAPPING = _mapping(
    "huggingface.gemma2.alternating-window-softcap-dense",
    "Gemma2ForCausalLM",
    "Gemma2Config",
    position="alternating_sliding_window_rope",
    mlp="geglu_tanh",
    operations=(
        "grouped_causal_attention",
        "alternating_sliding_window",
        "rope_position",
        "attention_softcap",
        "qkv_layout",
        "geglu_mlp",
        "scaled_embedding",
        "linear_causal_loss",
    ),
)


ROPE_GATED_DENSE_MAPPINGS = (
    LLAMA_MAPPING,
    MISTRAL_MAPPING,
    QWEN2_MAPPING,
    GEMMA_MAPPING,
    GEMMA2_MAPPING,
)


def register_rope_gated_dense_adapters(registry: ModelAdapterRegistry) -> None:
    """Register guarded Llama, Mistral, Qwen2, Gemma, and Gemma2 mappings."""

    if not isinstance(registry, ModelAdapterRegistry):
        raise TypeError("registry must be a ModelAdapterRegistry.")
    for mapping in ROPE_GATED_DENSE_MAPPINGS:
        registry.register(mapping.adapter)


__all__ = [
    "GEMMA2_MAPPING",
    "GEMMA_MAPPING",
    "LLAMA_MAPPING",
    "MISTRAL_MAPPING",
    "QWEN2_MAPPING",
    "ROPE_GATED_DENSE_MAPPINGS",
    "register_rope_gated_dense_adapters",
]
