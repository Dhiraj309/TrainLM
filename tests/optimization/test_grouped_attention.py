import pytest

from trainlm.optimization import (
    AttentionMaskSpec,
    CanonicalAttentionSpec,
    KVHeadMapping,
    PallasAttentionRuntime,
    pallas_grouped_attention_provider,
)


def spec(kv_heads, **values):
    layout = "mha" if kv_heads == 8 else "mqa" if kv_heads == 1 else "gqa"
    defaults = dict(
        schema_version=1,
        layout=layout,
        query_heads=8,
        key_value_heads=kv_heads,
        head_dim=64,
        position_encoding="rope",
        mask=AttentionMaskSpec(),
    )
    return CanonicalAttentionSpec(**{**defaults, **values})


def runtime(kernel, *, grouped=True, **values):
    defaults = dict(
        torch_xla_version="2.8.0",
        tested_torch_xla_versions=("2.8.0",),
        kernel=kernel,
        backward_verified=True,
        grouped_query_verified=grouped,
    )
    return PallasAttentionRuntime(**{**defaults, **values})


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


def test_sliding_window_is_forwarded_without_dense_mask_materialization():
    calls = []
    provider = pallas_grouped_attention_provider(
        runtime(
            lambda *args, **kwargs: calls.append(kwargs) or "output",
            sliding_window_verified=True,
        ),
        spec(
            4,
            mask=AttentionMaskSpec(
                layout="causal_sliding_window", sliding_window=4096
            ),
        ),
    )
    assert provider.mask_factory(sequence_length=8192) is None
    provider.attention_forward(None, "q", "k", "v")
    assert calls[0]["sliding_window"] == 4096


def test_alibi_requires_explicit_evidence_and_model_supplied_slopes():
    alibi = spec(8, position_encoding="alibi")
    kernel = lambda *args, **kwargs: "output"
    with pytest.raises(RuntimeError, match="ALiBi.*runtime evidence"):
        pallas_grouped_attention_provider(runtime(kernel), alibi)
    with pytest.raises(ValueError, match="one explicit numeric slope"):
        pallas_grouped_attention_provider(
            runtime(kernel, alibi_verified=True), alibi, alibi_slopes=(0.5,)
        )


def test_model_supplied_alibi_slopes_are_forwarded_unchanged():
    calls = []
    slopes = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
    provider = pallas_grouped_attention_provider(
        runtime(
            lambda *args, **kwargs: calls.append(kwargs) or "output",
            alibi_verified=True,
        ),
        spec(8, position_encoding="alibi"),
        alibi_slopes=slopes,
    )
    provider.attention_forward(None, "q", "k", "v")
    assert calls[0]["alibi_slopes"] is slopes


def test_sliding_window_requires_explicit_runtime_evidence():
    with pytest.raises(RuntimeError, match="Sliding-window.*runtime evidence"):
        pallas_grouped_attention_provider(
            runtime(lambda *args, **kwargs: None),
            spec(
                8,
                mask=AttentionMaskSpec(
                    layout="causal_sliding_window", sliding_window=128
                ),
            ),
        )
