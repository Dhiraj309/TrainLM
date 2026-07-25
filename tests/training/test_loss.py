import torch
from torch import nn

from trainlm.runtime import Runtime
from trainlm.training.loss import LanguageModelLoss


class Output:

    def __init__(self, loss):
        self.loss = loss


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, input_ids):
        return Output(self.linear(input_ids).mean())


def test_language_model_loss():
    model = Model()

    runtime = Runtime()

    loss_fn = LanguageModelLoss()

    loss = loss_fn(
        model,
        {
            "input_ids": torch.randn(2, 4),
        },
        runtime,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
