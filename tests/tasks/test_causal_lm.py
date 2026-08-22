from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trainlm.runtime import Runtime
from trainlm.tasks import CausalLMTask, TaskResult, TokenCounts


class FixedLogitsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits)
        self.received = None

    def forward(self, **inputs):
        self.received = inputs
        return SimpleNamespace(logits=self.logits)


class ExplicitInputsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits)
        self.received = None

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        self.received = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return SimpleNamespace(logits=self.logits)


def test_causal_task_owns_shift_masks_normalization_and_counts():
    logits = torch.tensor(
        [[
            [0.0, 2.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, -1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]],
        requires_grad=True,
    )
    model = FixedLogitsModel(logits)
    task = CausalLMTask(ignore_index=-100)

    result = task.training_step(
        model,
        {
            "input_ids": torch.tensor([[0, 1, 2, 2]]),
            "labels": torch.tensor([[0, 1, -100, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        },
        Runtime(),
    )

    expected = F.cross_entropy(logits[:, 0, :], torch.tensor([1]))
    assert torch.allclose(result.loss, expected)
    assert result.tokens == TokenCounts(
        sequences=1,
        input_tokens=4,
        target_tokens=3,
        supervised_tokens=1,
        ignored_tokens=2,
    )
    assert "labels" not in model.received
    assert "loss_mask" not in model.received


def test_causal_task_uses_input_ids_as_labels_when_labels_are_absent():
    model = FixedLogitsModel(torch.randn(2, 4, 8))

    result = CausalLMTask().evaluation_step(
        model,
        {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])},
        Runtime(),
    )

    assert isinstance(result, TaskResult)
    assert result.tokens.supervised_tokens == 6
    assert result.tokens.ignored_tokens == 0


def test_causal_task_filters_dataset_metadata_before_model_forward():
    model = ExplicitInputsModel(torch.randn(1, 4, 8))

    CausalLMTask().training_step(
        model,
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4),
            "position_ids": torch.arange(4).unsqueeze(0),
            "dataset_document_id": torch.tensor(12),
        },
        Runtime(),
    )

    assert set(model.received) == {"input_ids", "attention_mask", "position_ids"}


def test_causal_task_rejects_batch_without_supervised_targets():
    model = FixedLogitsModel(torch.randn(1, 3, 8))

    with pytest.raises(ValueError, match="no supervised tokens"):
        CausalLMTask().training_step(
            model,
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.zeros(1, 3),
            },
            Runtime(),
        )


def test_causal_task_weights_evaluation_by_supervised_tokens():
    task = CausalLMTask()
    results = [
        TaskResult(
            loss=torch.tensor(2.0),
            tokens=TokenCounts(1, 4, 3, 3, 0),
        ),
        TaskResult(
            loss=torch.tensor(8.0),
            tokens=TokenCounts(1, 4, 3, 1, 2),
        ),
    ]

    metrics = task.aggregate_evaluation(results)

    assert metrics["eval_loss"] == pytest.approx(3.5)
