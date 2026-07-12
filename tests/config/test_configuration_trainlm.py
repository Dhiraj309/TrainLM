from transformers import PretrainedConfig

from trainlm.config import TrainLMConfig


def test_default_configuration():
    config = TrainLMConfig()

    assert config.model_type == "trainlm"


def test_inherits_pretrained_config():
    config = TrainLMConfig()

    assert isinstance(config, PretrainedConfig)


def test_default_embedding_configuration():
    config = TrainLMConfig()

    assert config.vocab_size == 32000
    assert config.tie_word_embeddings is True


def test_invalid_vocab_size():
    import pytest

    with pytest.raises(ValueError):
        TrainLMConfig(vocab_size=0)


def test_untied_embeddings_not_supported():
    import pytest

    with pytest.raises(ValueError):
        TrainLMConfig(tie_word_embeddings=False)
