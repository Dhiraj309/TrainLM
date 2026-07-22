import torch.nn as nn

from transformers import PreTrainedModel

from trainlm.config import TrainLMConfig
from trainlm.model import TrainLMPreTrainedModel


class DummyModel(TrainLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass


def test_pretrained_model_inheritance():
    model = DummyModel(TrainLMConfig())

    assert isinstance(model, PreTrainedModel)


def test_config_class():
    assert TrainLMPreTrainedModel.config_class is TrainLMConfig


def test_base_model_prefix():
    assert TrainLMPreTrainedModel.base_model_prefix == "model"


def test_linear_initialization():
    model = DummyModel(TrainLMConfig())

    linear = nn.Linear(16, 16)

    model._init_weights(linear)

    assert linear.bias is not None
