import pytest

from trainlm.optimization import (
    AttentionTuningCache,
    AttentionTuningCandidate,
    AttentionTuningKey,
)


def key(**values):
    defaults = dict(
        hardware="v5e-8",
        provider_id="trainlm.pallas_grouped_attention",
        provider_version="1",
        dtype="bf16",
        batch_size=8,
        query_heads=8,
        key_value_heads=4,
        sequence_length=4096,
        head_dim=64,
        mask_layout="causal",
    )
    return AttentionTuningKey(**{**defaults, **values})


def candidate(name, block_q):
    return AttentionTuningCandidate(name, block_q, 128, 4)


def test_tuning_is_independent_of_candidate_input_order():
    first, second = candidate("a", 128), candidate("b", 256)
    scores = {"a": 10.0, "b": 20.0}
    forward = AttentionTuningCache().tune(
        key(), (first, second), lambda item: scores[item.candidate_id]
    )
    reverse = AttentionTuningCache().tune(
        key(), (second, first), lambda item: scores[item.candidate_id]
    )
    assert forward == reverse
    assert forward.candidate == second


def test_equal_scores_use_stable_candidate_id_tie_breaker():
    result = AttentionTuningCache().tune(
        key(), (candidate("z", 256), candidate("a", 128)), lambda item: 1.0
    )
    assert result.candidate.candidate_id == "a"


def test_exact_cache_hit_does_not_benchmark_again():
    cache = AttentionTuningCache()
    selected = cache.tune(key(), (candidate("a", 128),), lambda item: 5.0)
    cached = cache.tune(
        key(),
        (candidate("b", 256),),
        lambda item: pytest.fail("cache hit re-ran benchmark"),
    )
    assert cached is selected


def test_shape_dtype_and_provider_version_are_distinct_cache_keys():
    base = key()
    variants = (
        key(sequence_length=8192),
        key(dtype="fp32"),
        key(provider_version="2"),
    )
    assert len({base.cache_id, *(item.cache_id for item in variants)}) == 4


def test_cache_json_round_trip_is_stable():
    cache = AttentionTuningCache()
    cache.tune(key(), (candidate("a", 128),), lambda item: 42.0)
    encoded = cache.to_json()
    restored = AttentionTuningCache.from_json(encoded)
    assert restored.to_json() == encoded
    assert restored.get(key()) == cache.get(key())


def test_non_finite_benchmark_score_is_rejected():
    with pytest.raises(ValueError, match="score must be finite"):
        AttentionTuningCache().tune(
            key(), (candidate("a", 128),), lambda item: float("nan")
        )
