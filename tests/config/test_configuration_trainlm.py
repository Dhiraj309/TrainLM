from transformers import PretrainedConfig

from trainlm.config import TrainLMConfig


def test_default_configuration():
    config = TrainLMConfig()

    assert config.model_type == "trainlm"


def test_inherits_pretrained_config():
    config = TrainLMConfig()

    assert isinstance(config, PretrainedConfig)
