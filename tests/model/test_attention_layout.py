from types import SimpleNamespace

import pytest

from trainlm.model import detect_attention_layout


@pytest.mark.parametrize(
    ("config", "kind", "query", "kv", "projection"),
    [
        (
            SimpleNamespace(
                num_attention_heads=8,
                num_key_value_heads=8,
                fused_qkv=True,
            ),
            "mha",
            8,
            8,
            "packed",
        ),
        (
            SimpleNamespace(
                num_attention_heads=8,
                num_key_value_heads=2,
                fused_qkv=False,
            ),
            "gqa",
            8,
            2,
            "separate",
        ),
        (
            SimpleNamespace(
                num_attention_heads=8,
                multi_query=True,
            ),
            "mqa",
            8,
            1,
            "unknown",
        ),
    ],
)
def test_head_layouts_and_projection_packing_are_classified(
    config, kind, query, kv, projection
):
    result = detect_attention_layout(config)

    assert result.kind == kind
    assert result.projection == projection
    assert result.query_heads == query
    assert result.kv_heads == kv
    assert result.evidence


def test_incomplete_head_counts_are_unknown():
    result = detect_attention_layout(SimpleNamespace(fused_qkv=True))

    assert result.kind == "unknown"
    assert result.projection == "packed"
    assert "incomplete" in result.evidence[-1]


def test_incompatible_head_counts_are_not_silently_rewritten():
    result = detect_attention_layout(
        SimpleNamespace(num_attention_heads=7, num_key_value_heads=2)
    )

    assert result.kind == "unknown"
    assert "incompatible" in result.evidence[-1]


def test_module_structure_can_describe_projection_layout():
    class Q_Proj:
        pass

    class K_Proj:
        pass

    class V_Proj:
        pass

    class Model:
        def modules(self):
            return iter((Q_Proj(), K_Proj(), V_Proj()))

    result = detect_attention_layout(
        SimpleNamespace(num_attention_heads=2), Model()
    )

    assert result.kind == "mha"
    assert result.projection == "separate"


def test_config_is_required():
    with pytest.raises(TypeError, match="config is required"):
        detect_attention_layout(None)
