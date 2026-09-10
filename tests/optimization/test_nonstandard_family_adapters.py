from dataclasses import replace

from trainlm.optimization import (
    FALCON_GQA_MAPPING,
    FALCON_MQA_MAPPING,
    NONSTANDARD_DENSE_MAPPINGS,
    PHI_MAPPING,
    ComponentCapability,
    ModelAdapterRegistry,
    register_nonstandard_dense_adapters,
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


def test_catalog_keeps_falcon_head_layouts_and_phi_semantics_explicit():
    assert NONSTANDARD_DENSE_MAPPINGS == (
        FALCON_MQA_MAPPING,
        FALCON_GQA_MAPPING,
        PHI_MAPPING,
    )
    assert FALCON_MQA_MAPPING.adapter.capability_kinds[0] == ("attention", "mqa")
    assert FALCON_GQA_MAPPING.adapter.capability_kinds[0] == ("attention", "gqa")
    assert "partial_rope_position" in PHI_MAPPING.operations
    assert "parallel_residual" in PHI_MAPPING.operations


def test_each_nonstandard_mapping_resolves_only_its_semantic_variant():
    registry = ModelAdapterRegistry()
    register_nonstandard_dense_adapters(registry)

    for mapping in NONSTANDARD_DENSE_MAPPINGS:
        resolution = registry.resolve(
            _report(mapping), package_versions={"transformers": "5.15.0"}
        )
        assert resolution.selected == mapping.adapter


def test_falcon_mqa_report_cannot_select_gqa_mapping():
    registry = ModelAdapterRegistry()
    register_nonstandard_dense_adapters(registry)
    resolution = registry.resolve(
        _report(FALCON_MQA_MAPPING), package_versions={"transformers": "5.15.0"}
    )
    candidates = {item.adapter_id: item for item in resolution.candidates}

    assert resolution.selected == FALCON_MQA_MAPPING.adapter
    gqa = candidates[FALCON_GQA_MAPPING.adapter.adapter_id]
    assert not gqa.eligible
    assert any("requires 'gqa'" in reason for reason in gqa.reasons)


def test_phi_mapping_rejects_full_rope_or_serial_residual():
    registry = ModelAdapterRegistry()
    register_nonstandard_dense_adapters(registry)
    report = replace(
        _report(PHI_MAPPING),
        position=_known("rope"),
        residual=_known("serial"),
    )

    assert registry.resolve(
        report, package_versions={"transformers": "5.15.0"}
    ).selected is None
