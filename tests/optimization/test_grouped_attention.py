import pytest

from trainlm.optimization import (
    AttentionMaskSpec,
    CanonicalAttentionSpec,
    KVHeadMapping,
    PallasAttentionRuntime,
    pallas_grouped_attention_provider,
)


def spec(kv_heads):
    layout = "mha" if kv_heads == 8 else "mqa" if kv_heads == 1 else "gqa"
    return CanonicalAttentionSpec(
        schema_version=1,
        layout=layout,
        query_heads=8,
        key_value_heads=kv_heads,
        head_dim=64,
        position_encoding="rope",
        mask=AttentionMaskSpec(),
    )


def runtime(kernel, *, grouped=True):
    return PallasAttentionRuntime(
        torch_xla_version="2.8.0",
        tested_torch_xla_versions=("2.8.0",),
        kernel=kernel,
        backward_verified=True,
        grouped_query_verified=grouped,
    )


@pytest.mark.parametrize(
    ("kv_heads", "owners"),
    [
        (8, (0, 1, 2, 3, 4, 5, 6, 7)),
        (4, (0, 0, 1, 1, 2, 2, 3, 3)),
        (1, (0, 0, 0, 0, 0, 0, 0, 0)),
    ],
)
def test_logical_head_mapping_does_not_expand_kv_storage(kv_heads, owners):
    mapping = KVHeadMapping(8, kv_heads)
    assert mapping.as_tuple() == owners
    assert mapping.key_value_heads == kv_heads


@pytest.mark.parametrize("kv_heads", [8, 4, 1])
def test_provider_passes_compact_head_geometry_to_kernel(kv_heads):
    calls = []

    def kernel(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return "output"

    provider = pallas_grouped_attention_provider(runtime(kernel), spec(kv_heads))
    assert provider.attention_forward(None, "q", "compact-k", "compact-v") == (
        "output",
        None,
    )
    assert calls[0][1:3] == ("compact-k", "compact-v")
    assert calls[0][3]["query_heads"] == 8
    assert calls[0][3]["key_value_heads"] == kv_heads


def test_gqa_and_mqa_require_explicit_runtime_evidence():
    with pytest.raises(RuntimeError, match="explicit GQA/MQA evidence"):
        pallas_grouped_attention_provider(
            runtime(lambda *args, **kwargs: None, grouped=False), spec(4)
        )


def test_invalid_non_divisible_head_mapping_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        KVHeadMapping(8, 3)
