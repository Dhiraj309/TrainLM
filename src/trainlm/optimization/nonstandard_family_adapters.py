"""Explicit mappings for Falcon and Phi dense causal model variants."""

from __future__ import annotations

from .adapters import AdapterSpec, ModelAdapterRegistry, PackageVersionGuard
from .dense_family_adapters import DenseCausalFamilyMapping


_TRANSFORMERS_GUARD = PackageVersionGuard("transformers", ("5.15.0",))


def _falcon_mapping(attention_kind: str) -> DenseCausalFamilyMapping:
    return DenseCausalFamilyMapping(
        adapter=AdapterSpec(
            adapter_id=f"huggingface.falcon.{attention_kind}-parallel-dense",
            model_classes=("FalconForCausalLM",),
            config_classes=("FalconConfig",),
            capability_kinds=(
                ("attention", attention_kind),
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
            "grouped_causal_attention",
            "rope_position",
            "qkv_layout",
            "gelu_mlp",
            "parallel_residual",
            "linear_causal_loss",
        ),
    )


FALCON_MQA_MAPPING = _falcon_mapping("mqa")
FALCON_GQA_MAPPING = _falcon_mapping("gqa")


PHI_MAPPING = DenseCausalFamilyMapping(
    adapter=AdapterSpec(
        adapter_id="huggingface.phi.partial-rope-parallel-dense",
        model_classes=("PhiForCausalLM",),
        config_classes=("PhiConfig",),
        capability_kinds=(
            ("attention", "mha"),
            ("position", "partial_rope"),
            ("normalization", "layer_norm"),
            ("mlp", "gelu_feed_forward"),
            ("residual", "parallel"),
            ("projections", "separate_qkv"),
            ("lm_head", "linear"),
        ),
        version_guards=(_TRANSFORMERS_GUARD,),
    ),
    operations=(
        "causal_attention",
        "partial_rope_position",
        "qkv_layout",
        "gelu_mlp",
        "parallel_residual",
        "linear_causal_loss",
    ),
)


NONSTANDARD_DENSE_MAPPINGS = (FALCON_MQA_MAPPING, FALCON_GQA_MAPPING, PHI_MAPPING)


def register_nonstandard_dense_adapters(registry: ModelAdapterRegistry) -> None:
    """Register Falcon head-layout variants and Phi in stable order."""

    if not isinstance(registry, ModelAdapterRegistry):
        raise TypeError("registry must be a ModelAdapterRegistry.")
    for mapping in NONSTANDARD_DENSE_MAPPINGS:
        registry.register(mapping.adapter)


__all__ = [
    "FALCON_GQA_MAPPING",
    "FALCON_MQA_MAPPING",
    "NONSTANDARD_DENSE_MAPPINGS",
    "PHI_MAPPING",
    "register_nonstandard_dense_adapters",
]
