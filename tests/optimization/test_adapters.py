from dataclasses import replace

import pytest

from trainlm.optimization import (
    AdapterSpec,
    ModelAdapterRegistry,
    PackageVersionGuard,
)

from .test_capabilities import capabilities


def _adapter(adapter_id="llama-qkv", *, priority=0):
    return AdapterSpec(
        adapter_id=adapter_id,
        model_classes=("LlamaForCausalLM",),
        config_classes=("LlamaConfig",),
        capability_kinds=(("attention", "gqa"), ("projections", "separate_qkv")),
        version_guards=(PackageVersionGuard("transformers", ("5.15.0",)),),
        priority=priority,
    )


def test_registry_selects_only_explicit_class_semantics_and_version_match():
    registry = ModelAdapterRegistry()
    registry.register(_adapter())

    resolution = registry.resolve(
        capabilities(), package_versions={"transformers": "5.15.0"}
    )

    assert resolution.selected is not None
    assert resolution.selected.adapter_id == "llama-qkv"
    assert resolution.candidates[0].eligible
    assert resolution.candidates[0].reasons == ()


def test_registry_explains_unknown_semantics_and_untested_versions():
    registry = ModelAdapterRegistry()
    registry.register(_adapter())
    report = replace(
        capabilities(),
        projections=capabilities().projections.unknown("not inspected"),
    )

    resolution = registry.resolve(
        report, package_versions={"transformers": "5.16.0"}
    )

    assert resolution.selected is None
    reasons = resolution.candidates[0].reasons
    assert any("projections requires" in reason for reason in reasons)
    assert any("untested" in reason for reason in reasons)


def test_registry_resolution_is_stable_by_priority_then_id():
    registry = ModelAdapterRegistry()
    registry.register(_adapter("lower", priority=1))
    registry.register(_adapter("z-high", priority=2))
    registry.register(_adapter("a-high", priority=2))

    resolution = registry.resolve(
        capabilities(), package_versions={"transformers": "5.15.0"}
    )

    assert resolution.selected.adapter_id == "a-high"
    assert [candidate.adapter_id for candidate in resolution.candidates] == [
        "a-high", "z-high", "lower"
    ]


def test_registry_rejects_duplicates_and_invalid_guards():
    registry = ModelAdapterRegistry()
    registry.register(_adapter())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_adapter())
    with pytest.raises(ValueError, match="at least one tested version"):
        PackageVersionGuard("transformers", ())
    with pytest.raises(ValueError, match="unique by component"):
        AdapterSpec(
            adapter_id="bad",
            model_classes=("SomeModel",),
            capability_kinds=(("attention", "mha"), ("attention", "gqa")),
        )
