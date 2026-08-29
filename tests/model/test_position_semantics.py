from types import SimpleNamespace

import pytest

from trainlm.model import detect_position_semantics


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(position_embedding_type="learned"), "learned"),
        (SimpleNamespace(position_embedding_type="absolute"), "learned"),
        (SimpleNamespace(rope_theta=10_000.0), "rope"),
        (SimpleNamespace(position_embedding_type="alibi"), "alibi"),
    ],
)
def test_explicit_position_semantics_are_classified(config, expected):
    result = detect_position_semantics(config)

    assert result.kind == expected
    assert result.evidence


def test_max_position_embeddings_alone_is_not_enough():
    result = detect_position_semantics(
        SimpleNamespace(max_position_embeddings=2048)
    )

    assert result.kind == "unknown"
    assert "no explicit" in result.evidence[0]


def test_conflicting_position_indicators_remain_unknown():
    result = detect_position_semantics(
        SimpleNamespace(position_embedding_type="alibi", rope_theta=10_000.0)
    )

    assert result.kind == "unknown"
    assert "contradictory" in result.evidence[-1]


def test_module_fallback_is_optional_and_safe():
    class RotaryEmbedding:
        pass

    class Model:
        def modules(self):
            return iter((RotaryEmbedding(),))

    assert detect_position_semantics(SimpleNamespace(), Model()).kind == "rope"
    assert detect_position_semantics(SimpleNamespace(), object()).kind == "unknown"


def test_config_is_required():
    with pytest.raises(TypeError, match="config is required"):
        detect_position_semantics(None)
