"""Adapter selection is optional, exact-class, versioned, and read-only."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from trainlm.optimization import (
    AdapterDefinition, AdapterRegistry, CapabilityFact, CapabilityRequirement,
    ComponentCapability, VersionRequirement, inspect_dense_causal_lm,
)


class FixtureConfig:
    model_type = "llama"  # Metadata alone must never select an adapter.
    num_attention_heads = 2
    num_key_value_heads = 1


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FixtureConfig()
        self.embedding = nn.Embedding(16, 4)
        self.head = nn.Linear(4, 16, bias=False)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        return self.head(hidden + torch.sin(hidden))


def adapter():
    return AdapterDefinition(
        adapter_id="fixture.serial-v1",
        model_class=f"{FixtureModel.__module__}.{FixtureModel.__qualname__}",
        config_class=f"{FixtureConfig.__module__}.{FixtureConfig.__qualname__}",
        versions=(VersionRequirement("torch", ">=2.9,<2.10"),),
        requirements=(CapabilityRequirement("attention", ("mqa",), (CapabilityFact("heads", 2),)),),
        overrides=(("residual", ComponentCapability(
            status="known", kind="serial", facts=(CapabilityFact("scale", 1.0, "fixture forward audit"),),
            evidence=("audited exact fixture class",),
        )),),
    )


def test_adapter_enriches_report_without_changing_model_or_generic_execution():
    model = FixtureModel().train()
    inputs = torch.tensor([[1, 2, 3]])
    baseline = model(inputs).detach().clone()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    registry = AdapterRegistry((adapter(),))
    resolution = registry.resolve(model, versions={"torch": "2.9.0+cpu"})
    assert resolution.adapter_id == "fixture.serial-v1"
    assert resolution.capabilities.residual.kind == "serial"
    assert "adapter=fixture.serial-v1" in str(resolution.capabilities.residual.evidence)
    assert resolution.capabilities.fingerprint != inspect_dense_causal_lm(model).fingerprint
    assert model.training and model.head.weight is model.embedding.weight
    assert all(torch.equal(weights[name], value) for name, value in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
    removed = AdapterRegistry().resolve(model, versions={"torch": "2.9.0"})
    assert removed.capabilities == inspect_dense_causal_lm(model)
    assert removed.adapter_id is None and removed.capabilities.residual.status == "unknown"
    torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)


@pytest.mark.parametrize("versions, reason", [({}, "missing version"),
    ({"torch": "2.10.0"}, "does not satisfy"), ({"torch": "invalid"}, "invalid version")])
def test_missing_or_unsupported_versions_preserve_generic_capabilities(versions, reason):
    model = FixtureModel()
    result = AdapterRegistry((adapter(),)).resolve(model, versions=versions)
    assert result.adapter_id is None
    assert result.capabilities == inspect_dense_causal_lm(model)
    assert reason in result.rejections[0]


def test_family_name_and_inheritance_do_not_select_exact_class_adapters():
    class DifferentModel(FixtureModel):
        pass
    result = AdapterRegistry((adapter(),)).resolve(DifferentModel(), versions={"torch": "2.9.0"})
    assert result.adapter_id is None
    assert result.capabilities.model_type == "llama"


def test_semantic_guards_and_config_identity_are_checked():
    model = FixtureModel()
    wrong_semantics = replace(adapter(), requirements=(
        CapabilityRequirement("attention", ("mqa",), (CapabilityFact("heads", 4),)),
    ))
    result = AdapterRegistry((wrong_semantics,)).resolve(model, versions={"torch": "2.9.0"})
    assert result.adapter_id is None and "heads must equal 4" in result.rejections[0]
    model.config = type("OtherConfig", (FixtureConfig,), {})()
    result = AdapterRegistry((adapter(),)).resolve(model, versions={"torch": "2.9.0"})
    assert result.adapter_id is None and "config class" in result.rejections[0]


def test_ambiguous_or_contradictory_adapters_fail_without_mutation():
    model = FixtureModel()
    first = adapter()
    second = replace(first, adapter_id="fixture.other")
    for definitions in ((first, second), (second, first)):
        with pytest.raises(ValueError, match="Ambiguous adapters: fixture.other, fixture.serial-v1"):
            AdapterRegistry(definitions).resolve(model, versions={"torch": "2.9.0"})
    contradicting = replace(first, overrides=(("attention", ComponentCapability(
        status="known", kind="mha", evidence=("conflicting fixture declaration",),
    )),))
    with pytest.raises(ValueError, match="contradicts inspected attention kind"):
        AdapterRegistry((contradicting,)).resolve(model, versions={"torch": "2.9.0"})
    assert model.head.weight is model.embedding.weight


def test_registry_rejects_duplicate_ids_and_unguarded_or_unevidenced_semantics():
    with pytest.raises(ValueError, match="IDs must be unique"):
        AdapterRegistry((adapter(), adapter()))
    with pytest.raises(ValueError, match="version guards"):
        replace(adapter(), versions=())
    with pytest.raises(ValueError, match="semantics and evidence"):
        replace(adapter(), overrides=(("residual", ComponentCapability(status="known", kind="serial")),))
