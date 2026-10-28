"""Explicit mappings for learned-position dense causal language models."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import AdapterSpec, ModelAdapterRegistry, PackageVersionGuard


_TRANSFORMERS_GUARD = PackageVersionGuard("transformers", ("5.15.0",))


@dataclass(frozen=True, slots=True)
class DenseCausalFamilyMapping:
    """An adapter plus the reusable optimization operations it may request.

    The mapping is deliberately declarative.  Eligibility still depends on an
    inspected capability report and an exact tested Transformers version; this
    module neither imports Transformers nor mutates a model.
    """

    adapter: AdapterSpec
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.operations or any(
            not isinstance(operation, str) or not operation.strip()
            for operation in self.operations
        ):
            raise ValueError("Family mappings require non-empty operation IDs.")
        if len(self.operations) != len(set(self.operations)):
            raise ValueError("Family mapping operation IDs must be unique.")


GPT2_MAPPING = DenseCausalFamilyMapping(
    adapter=AdapterSpec(
        adapter_id="huggingface.gpt2.learned-position-dense",
        model_classes=("GPT2LMHeadModel",),
        config_classes=("GPT2Config",),
        capability_kinds=(
            ("attention", "mha"),
            ("position", "learned_absolute"),
            ("normalization", "layer_norm"),
            ("mlp", "gelu_feed_forward"),
            ("residual", "serial"),
            ("projections", "fused_qkv"),
            ("lm_head", "linear"),
        ),
        version_guards=(_TRANSFORMERS_GUARD,),
    ),
    operations=("causal_attention", "qkv_layout", "gelu_mlp", "linear_causal_loss"),
)


OPT_MAPPING = DenseCausalFamilyMapping(
    adapter=AdapterSpec(
        adapter_id="huggingface.opt.learned-position-dense",
        model_classes=("OPTForCausalLM",),
        config_classes=("OPTConfig",),
        capability_kinds=(
            ("attention", "mha"),
            ("position", "learned_absolute_offset"),
            ("normalization", "layer_norm"),
            ("mlp", "gelu_feed_forward"),
            ("residual", "serial"),
            ("projections", "separate_qkv"),
            ("lm_head", "linear"),
        ),
        version_guards=(_TRANSFORMERS_GUARD,),
    ),
    operations=("causal_attention", "qkv_layout", "gelu_mlp", "linear_causal_loss"),
)


LEARNED_POSITION_DENSE_MAPPINGS = (GPT2_MAPPING, OPT_MAPPING)


def register_learned_position_dense_adapters(registry: ModelAdapterRegistry) -> None:
    """Register the tested GPT-2 and OPT mappings in stable catalog order."""

    if not isinstance(registry, ModelAdapterRegistry):
        raise TypeError("registry must be a ModelAdapterRegistry.")
    for mapping in LEARNED_POSITION_DENSE_MAPPINGS:
        registry.register(mapping.adapter)


__all__ = [
    "DenseCausalFamilyMapping",
    "GPT2_MAPPING",
    "LEARNED_POSITION_DENSE_MAPPINGS",
    "OPT_MAPPING",
    "register_learned_position_dense_adapters",
]
