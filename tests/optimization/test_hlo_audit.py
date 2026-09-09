import json

import pytest

from trainlm.optimization import HLOFusionObservation, audit_hlo_fusions


def observation(component, **values):
    defaults = dict(
        component=component,
        hlo_fingerprint=f"sha256:{component}",
        native_fused=True,
        copy_count=0,
        transpose_count=0,
        materialization_count=0,
        custom_call_count=0,
    )
    return HLOFusionObservation(**{**defaults, **values})


def test_clean_native_fusion_is_retained():
    audit = audit_hlo_fusions((observation("normalization"),))
    assert audit.ready
    assert audit.decisions[0].decision == "native"


@pytest.mark.parametrize(
    "values",
    [
        {"native_fused": False},
        {"copy_count": 1},
        {"transpose_count": 1},
        {"materialization_count": 1},
        {"custom_call_count": 1},
    ],
)
def test_proven_native_overhead_selects_custom_candidate(values):
    audit = audit_hlo_fusions((observation("mlp", **values),))
    assert audit.ready
    assert audit.decisions[0].decision == "custom"


def test_unknown_fusion_and_fallbacks_block_selection():
    audit = audit_hlo_fusions(
        (
            observation("rope", native_fused=None),
            observation("residual", fallback_count=1),
        )
    )
    assert not audit.ready
    assert [item.decision for item in audit.decisions] == ["blocked", "blocked"]


def test_decision_order_and_json_are_deterministic():
    reverse = (
        observation("rope"),
        observation("mlp", native_fused=False),
        observation("normalization"),
    )
    forward = tuple(reversed(reverse))
    first = audit_hlo_fusions(reverse)
    second = audit_hlo_fusions(forward)
    assert first == second
    assert json.loads(first.to_json())["schema_version"] == 1


def test_duplicate_component_evidence_is_rejected():
    with pytest.raises(ValueError, match="components must be unique"):
        audit_hlo_fusions((observation("mlp"), observation("mlp")))
