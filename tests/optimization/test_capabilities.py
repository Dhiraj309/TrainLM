import json

import pytest

from trainlm.optimization import (
    CapabilityFact,
    ComponentCapability,
    ModelCapabilities,
)


def _known(kind, **facts):
    return ComponentCapability(
        status="known",
        kind=kind,
        facts=tuple(
            CapabilityFact(name=name, value=value, source="config")
            for name, value in facts.items()
        ),
        evidence=("hf_config",),
    )


def capabilities():
    return ModelCapabilities(
        schema_version=1,
        model_type="llama",
        model_class="LlamaForCausalLM",
        config_class="LlamaConfig",
        source_provider="huggingface",
        architectures=("LlamaForCausalLM",),
        attention=_known("gqa", heads=8, kv_heads=4, head_dim=128),
        position=_known("rope", theta=10000.0),
        normalization=_known("rms_norm", placement="pre", epsilon=1e-5),
        mlp=_known("swiglu", intermediate_size=4096, bias=False),
        residual=_known("serial", scale=1.0),
        projections=_known("separate_qkv", bias=False),
        embedding=_known("standard", vocab_size=32064, hidden_size=1024),
        lm_head=_known("linear", tied=True, bias=False),
        checkpointing=_known("decoder_block", supported=True),
    )


def test_capability_report_round_trips_and_has_stable_fingerprint():
    report = capabilities()
    restored = ModelCapabilities.from_json(report.to_json())

    assert restored == report
    assert restored.fingerprint == report.fingerprint
    assert len(report.fingerprint) == 64
    assert json.loads(report.to_json())["attention"]["kind"] == "gqa"


def test_unknown_capability_is_explicit_and_cannot_claim_kind():
    unknown = ComponentCapability.unknown("No structural evidence.")

    assert unknown.status == "unknown"
    assert unknown.kind is None

    with pytest.raises(ValueError, match="cannot claim"):
        ComponentCapability(status="unknown", kind="rope")


def test_capability_facts_are_unique_and_scalar():
    fact = CapabilityFact("heads", 8, "config.num_attention_heads")

    with pytest.raises(ValueError, match="unique"):
        ComponentCapability(
            status="known",
            kind="mha",
            facts=(fact, fact),
        )

    with pytest.raises(ValueError, match="finite"):
        CapabilityFact("epsilon", float("nan"))

