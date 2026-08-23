"""Compatibility explanations for the generic dense-AR path."""

from __future__ import annotations

import pytest
import torch

from trainlm.model import (
    ModelCompatibilityExplanation,
    explain_huggingface_compatibility,
    load_huggingface_causal_lm,
)

from .dense_ar_fixtures import DENSE_AR_FIXTURES, DenseARFixture


@pytest.mark.parametrize("fixture", DENSE_AR_FIXTURES, ids=lambda item: item.name)
def test_generic_compatibility_explanation_is_stable_and_non_mutating(
    fixture: DenseARFixture,
):
    torch.manual_seed(0)
    loaded = load_huggingface_causal_lm(fixture.source(tied=True))
    model = loaded.model
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    training = model.training

    report = explain_huggingface_compatibility(loaded)
    restored = ModelCompatibilityExplanation.from_json(report.to_json())

    assert restored == report
    assert restored.to_json() == report.to_json()
    assert restored.explain() == report.explain()
    assert report.support_level == "compatible"
    assert report.selected_path == "huggingface.generic_causal_lm"
    assert report.adapter is None
    assert report.capabilities.model_type == fixture.model_type
    assert report.execution_plan.capability_fingerprint == (
        report.capabilities.fingerprint
    )
    assert all(
        report.capabilities.component(name).status == "unknown"
        for name in report.capabilities.component_names
    )

    assert len(report.fallbacks) == 1
    fallback = report.fallbacks[0]
    assert fallback.decision_id == "architecture-optimization"
    assert fallback.requested_provider == "trainlm.architecture_optimized"
    assert fallback.selected_provider == "huggingface.generic"

    explanation = report.explain()
    assert "Support level: Compatible" in explanation
    assert f"Model: {fixture.model_type}" in explanation
    assert "Adapter: none" in explanation
    assert "requested=trainlm.architecture_optimized" in explanation
    assert "Compatible does not mean TPU Optimized" in explanation

    assert model.training is training
    after = model.state_dict()
    assert tuple(after) == tuple(before)
    for name, expected in before.items():
        assert torch.equal(after[name], expected)


def test_compatibility_explanation_rejects_non_loaded_model():
    with pytest.raises(TypeError, match="LoadedCausalLM"):
        explain_huggingface_compatibility(object())
