from dataclasses import replace

from trainlm.optimization import (
    GPT2_MAPPING,
    LEARNED_POSITION_DENSE_MAPPINGS,
    OPT_MAPPING,
    ComponentCapability,
    ModelAdapterRegistry,
    register_learned_position_dense_adapters,
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


def test_catalog_maps_gpt2_and_opt_to_shared_operations():
    assert LEARNED_POSITION_DENSE_MAPPINGS == (GPT2_MAPPING, OPT_MAPPING)
    assert GPT2_MAPPING.operations == OPT_MAPPING.operations
    assert GPT2_MAPPING.adapter.capability_kinds != OPT_MAPPING.adapter.capability_kinds


def test_catalog_adapters_resolve_only_with_exact_semantics_and_version():
    registry = ModelAdapterRegistry()
    register_learned_position_dense_adapters(registry)

    for mapping in LEARNED_POSITION_DENSE_MAPPINGS:
        resolution = registry.resolve(
            _report(mapping), package_versions={"transformers": "5.15.0"}
        )
        assert resolution.selected == mapping.adapter


def test_gpt2_mapping_rejects_opt_projection_layout_and_untested_version():
    registry = ModelAdapterRegistry()
    register_learned_position_dense_adapters(registry)
    report = replace(_report(GPT2_MAPPING), projections=_known("separate_qkv"))

    resolution = registry.resolve(
        report, package_versions={"transformers": "5.16.0"}
    )

    assert resolution.selected is None
    reasons = resolution.candidates[0].reasons
    assert any("fused_qkv" in reason for reason in reasons)
    assert any("untested" in reason for reason in reasons)
