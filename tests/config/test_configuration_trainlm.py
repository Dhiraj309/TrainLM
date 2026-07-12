import pytest
from transformers import PretrainedConfig

from trainlm.config import TrainLMConfig


def test_inherits_pretrained_config():
    config = TrainLMConfig()

    assert isinstance(config, PretrainedConfig)


def test_model_type():
    config = TrainLMConfig()

    assert config.model_type == "trainlm"


def test_reference_configuration():
    config = TrainLMConfig()

    assert config.vocab_size == 32000
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.num_key_value_heads == 4
    assert config.intermediate_size == 3072
    assert config.hidden_act == "silu"
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 10000.0
    assert config.initializer_range == 0.02
    assert config.max_position_embeddings == 2048
    assert config.tie_word_embeddings is True
    assert config.attention_bias is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"hidden_size": 0},
        {"num_hidden_layers": 0},
        {"num_attention_heads": 0},
        {"num_key_value_heads": 0},
        {"intermediate_size": 0},
        {"max_position_embeddings": 0},
    ],
)
def test_positive_integer_validation(kwargs):
    with pytest.raises(ValueError):
        TrainLMConfig(**kwargs)


def test_hidden_size_divisibility():
    with pytest.raises(ValueError):
        TrainLMConfig(
            hidden_size=770,
            num_attention_heads=12,
        )


def test_gqa_validation():
    with pytest.raises(ValueError):
        TrainLMConfig(
            num_attention_heads=10,
            num_key_value_heads=3,
        )


def test_requires_tied_embeddings():
    with pytest.raises(ValueError):
        TrainLMConfig(
            tie_word_embeddings=False,
        )


def test_requires_bias_free_attention():
    with pytest.raises(ValueError):
        TrainLMConfig(
            attention_bias=True,
        )


def test_configuration_serialization():
    config = TrainLMConfig()

    config_dict = config.to_dict()

    restored = TrainLMConfig(**config_dict)

    assert restored.to_dict() == config_dict
