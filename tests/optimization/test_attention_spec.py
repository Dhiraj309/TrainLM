from dataclasses import replace

import pytest

from trainlm.optimization import (
    AttentionMaskSpec,
    CanonicalAttentionSpec,
    ComponentCapability,
)

from .test_capabilities import capabilities


def test_canonical_attention_maps_capabilities_and_round_trips():
    original = CanonicalAttentionSpec.from_capabilities(
        capabilities(),
        head_dim=128,
        mask=AttentionMaskSpec(
            layout="causal_sliding_window", sliding_window=4096, segment_ids=True
        ),
        dropout=0.1,
        soft_cap=50.0,
        qk_normalization=True,
    )
    restored = CanonicalAttentionSpec.from_json(original.to_json())

    assert restored == original
    assert restored.layout == "gqa"
    assert restored.position_encoding == "rope"
    assert restored.effective_scale == pytest.approx(128 ** -0.5)
    assert restored.mask.sliding_window == 4096
    assert restored.mask.segment_ids


@pytest.mark.parametrize(
    "values",
    [
        {"layout": "mha", "query_heads": 8, "key_value_heads": 4},
        {"layout": "gqa", "query_heads": 7, "key_value_heads": 2},
        {"layout": "mqa", "query_heads": 8, "key_value_heads": 2},
    ],
)
def test_canonical_attention_rejects_inconsistent_head_layout(values):
    with pytest.raises(ValueError, match="heads|layout"):
        CanonicalAttentionSpec(
            schema_version=1,
            head_dim=64,
            position_encoding="rope",
            mask=AttentionMaskSpec(),
            **values,
        )


def test_attention_mapping_rejects_unknown_semantics():
    report = replace(
        capabilities(),
        attention=ComponentCapability.unknown("Head geometry unavailable."),
    )
    with pytest.raises(ValueError, match="not proven"):
        CanonicalAttentionSpec.from_capabilities(report, head_dim=128)


def test_attention_mask_requires_consistent_sliding_window():
    with pytest.raises(ValueError, match="cannot declare"):
        AttentionMaskSpec(layout="causal", sliding_window=128)
    with pytest.raises(ValueError, match="positive window"):
        AttentionMaskSpec(layout="causal_sliding_window")
