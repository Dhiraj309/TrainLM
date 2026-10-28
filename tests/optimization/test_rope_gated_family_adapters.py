from dataclasses import replace

from trainlm.optimization import (
    GEMMA2_MAPPING,
    GEMMA_MAPPING,
    LLAMA_MAPPING,
    MISTRAL_MAPPING,
    QWEN2_MAPPING,
    ROPE_GATED_DENSE_MAPPINGS,
    ComponentCapability,
    ModelAdapterRegistry,
    register_rope_gated_dense_adapters,
)

from .test_capabilities import capabilities


def _known(kind):
    return ComponentCapability(status="known", kind=kind, evidence=("test fixture",))


def _report(mapping):
    values = {
        component: _known(kind)
        for component, kind in mapping.adapter.capability_kinds
    }
    return replace(
        capabilities(),
        model_class=mapping.adapter.model_classes[0],
        config_class=mapping.adapter.config_classes[0],
        **values,
    )


def test_catalog_preserves_family_specific_operations():
    assert ROPE_GATED_DENSE_MAPPINGS == (
        LLAMA_MAPPING,
        MISTRAL_MAPPING,
        QWEN2_MAPPING,
        GEMMA_MAPPING,
        GEMMA2_MAPPING,
    )
    assert "sliding_window" in MISTRAL_MAPPING.operations
    assert "biased_qkv_layout" in QWEN2_MAPPING.operations
    assert "scaled_embedding" in GEMMA_MAPPING.operations
    assert "attention_softcap" in GEMMA2_MAPPING.operations
    assert "alternating_sliding_window" in GEMMA2_MAPPING.operations


def test_each_rope_gated_mapping_resolves_at_tested_version():
    registry = ModelAdapterRegistry()
    register_rope_gated_dense_adapters(registry)
    for mapping in ROPE_GATED_DENSE_MAPPINGS:
        assert registry.resolve(
            _report(mapping), package_versions={"transformers": "5.15.0"}
        ).selected == mapping.adapter


def test_mistral_cannot_resolve_without_sliding_window_semantics():
    registry = ModelAdapterRegistry()
    register_rope_gated_dense_adapters(registry)
    report = replace(_report(MISTRAL_MAPPING), position=_known("rope"))

    assert registry.resolve(
        report, package_versions={"transformers": "5.15.0"}
    ).selected is None


def test_gemma2_cannot_fall_back_to_gemma_without_class_and_softcap_semantics():
    registry = ModelAdapterRegistry()
    register_rope_gated_dense_adapters(registry)
    report = replace(
        _report(GEMMA2_MAPPING),
        position=_known("rope"),
    )
    resolution = registry.resolve(
        report, package_versions={"transformers": "5.15.0"}
    )

    assert resolution.selected is None
    gemma = next(
        item
        for item in resolution.candidates
        if item.adapter_id == GEMMA_MAPPING.adapter.adapter_id
    )
    assert any("model class" in reason for reason in gemma.reasons)
