"""Explicit mappings for RoPE/ALiBi dense causal model families."""

from __future__ import annotations

from .adapters import AdapterSpec, ModelAdapterRegistry, PackageVersionGuard
from .dense_family_adapters import DenseCausalFamilyMapping


_TRANSFORMERS_GUARD = PackageVersionGuard("transformers", ("5.15.0",))


GPT_NEOX_MAPPING = DenseCausalFamilyMapping(
    adapter=AdapterSpec(
        adapter_id="huggingface.gpt-neox.rope-parallel-dense",
        model_classes=("GPTNeoXForCausalLM",),
        config_classes=("GPTNeoXConfig",),
        capability_kinds=(
            ("attention", "mha"),
            ("position", "rope"),
            ("normalization", "layer_norm"),
            ("mlp", "gelu_feed_forward"),
            ("residual", "parallel"),
            ("projections", "fused_qkv"),
            ("lm_head", "linear"),
        ),
        version_guards=(_TRANSFORMERS_GUARD,),
    ),
    operations=(
        "causal_attention",
        "rope_position",
        "qkv_layout",
        "gelu_mlp",
        "parallel_residual",
        "linear_causal_loss",
    ),
)


BLOOM_MAPPING = DenseCausalFamilyMapping(
    adapter=AdapterSpec(
        adapter_id="huggingface.bloom.alibi-dense",
        model_classes=("BloomForCausalLM",),
        config_classes=("BloomConfig",),
        capability_kinds=(
            ("attention", "mha"),
            ("position", "alibi"),
            ("normalization", "layer_norm"),
            ("mlp", "gelu_feed_forward"),
            ("residual", "serial"),
            ("projections", "fused_qkv"),
            ("lm_head", "linear"),
        ),
        version_guards=(_TRANSFORMERS_GUARD,),
    ),
    operations=(
        "causal_attention",
        "alibi_position",
        "qkv_layout",
        "gelu_mlp",
        "linear_causal_loss",
    ),
)


ROPE_ALIBI_DENSE_MAPPINGS = (GPT_NEOX_MAPPING, BLOOM_MAPPING)


def register_rope_alibi_dense_adapters(registry: ModelAdapterRegistry) -> None:
    """Register the tested GPT-NeoX and BLOOM mappings in stable order."""

    if not isinstance(registry, ModelAdapterRegistry):
        raise TypeError("registry must be a ModelAdapterRegistry.")
    for mapping in ROPE_ALIBI_DENSE_MAPPINGS:
        registry.register(mapping.adapter)


__all__ = [
    "BLOOM_MAPPING",
    "GPT_NEOX_MAPPING",
    "ROPE_ALIBI_DENSE_MAPPINGS",
    "register_rope_alibi_dense_adapters",
]
