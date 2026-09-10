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


def test_hlo_text_capture_counts_structural_operations_and_fingerprints():
    hlo = """
HloModule fixture
ENTRY main {
  %copy.1 = f32[2,2] copy(%parameter.0)
  %transpose.1 = f32[2,2] transpose(%copy.1), dimensions={1,0}
  ROOT %call = f32[2,2] custom-call(%transpose.1), custom_call_target="x"
}
"""
    captured = HLOFusionObservation.from_hlo_text(
        component="mlp",
        hlo_text=hlo,
        native_fused=False,
        materialization_patterns=(r"%copy\.1\s*=",),
    )

    assert captured.hlo_fingerprint.startswith("sha256:")
    assert len(captured.hlo_fingerprint) == 71
    assert captured.copy_count == 1
    assert captured.transpose_count == 1
    assert captured.custom_call_count == 1
    assert captured.materialization_count == 1


def test_hlo_text_fingerprint_normalizes_line_endings_and_outer_whitespace():
    unix = HLOFusionObservation.from_hlo_text(
        component="rope", hlo_text="HloModule x\n", native_fused=None
    )
    windows = HLOFusionObservation.from_hlo_text(
        component="rope", hlo_text="  HloModule x\r\n\r\n", native_fused=None
    )
    assert unix.hlo_fingerprint == windows.hlo_fingerprint


def test_hlo_text_capture_rejects_invalid_or_ambiguous_patterns():
    with pytest.raises(ValueError, match="hlo_text cannot be empty"):
        HLOFusionObservation.from_hlo_text(
            component="mlp", hlo_text="", native_fused=None
        )
    with pytest.raises(TypeError, match="tuple or list"):
        HLOFusionObservation.from_hlo_text(
            component="mlp",
            hlo_text="HloModule x",
            native_fused=None,
            materialization_patterns="copy",
        )
    with pytest.raises(ValueError, match="Invalid materialization pattern"):
        HLOFusionObservation.from_hlo_text(
            component="mlp",
            hlo_text="HloModule x",
            native_fused=None,
            materialization_patterns=("[",),
        )
