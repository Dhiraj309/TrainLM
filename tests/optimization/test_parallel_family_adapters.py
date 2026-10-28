from dataclasses import replace

from trainlm.optimization import (
    BLOOM_MAPPING,
    GPT_NEOX_MAPPING,
    ROPE_ALIBI_DENSE_MAPPINGS,
    ComponentCapability,
    ModelAdapterRegistry,
    register_rope_alibi_dense_adapters,
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


def test_catalog_preserves_family_specific_position_and_residual_semantics():
    assert ROPE_ALIBI_DENSE_MAPPINGS == (GPT_NEOX_MAPPING, BLOOM_MAPPING)
    assert "rope_position" in GPT_NEOX_MAPPING.operations
    assert "parallel_residual" in GPT_NEOX_MAPPING.operations
    assert "alibi_position" in BLOOM_MAPPING.operations
    assert "parallel_residual" not in BLOOM_MAPPING.operations


def test_catalog_resolves_both_explicit_mappings_at_tested_version():
    registry = ModelAdapterRegistry()
    register_rope_alibi_dense_adapters(registry)

    for mapping in ROPE_ALIBI_DENSE_MAPPINGS:
        resolution = registry.resolve(
            _report(mapping), package_versions={"transformers": "5.15.0"}
        )
        assert resolution.selected == mapping.adapter


def test_neox_mapping_rejects_serial_residual_and_missing_version_evidence():
    registry = ModelAdapterRegistry()
    register_rope_alibi_dense_adapters(registry)
    report = replace(_report(GPT_NEOX_MAPPING), residual=_known("serial"))

    resolution = registry.resolve(report, package_versions={})
    candidate = next(
        item
        for item in resolution.candidates
        if item.adapter_id == GPT_NEOX_MAPPING.adapter.adapter_id
    )

    assert resolution.selected is None
    assert any("parallel" in reason for reason in candidate.reasons)
    assert any("unavailable" in reason for reason in candidate.reasons)
