"""HF-like facade lifecycle, cadence, metrics, and local resume."""

from types import SimpleNamespace

import torch

from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.training import TrainerCallback


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = torch.nn.Linear(4, 8)

    def forward(self, input_ids, labels=None, **kwargs):
        del labels, kwargs
        logits = self.projection(self.embedding(input_ids))
        return SimpleNamespace(logits=logits)


class MetricsRecorder(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_metrics(self, state, control, metrics):
        del state, control
        self.metrics.append(dict(metrics))


def _dataset():
    return [
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "labels": torch.tensor([0, 1, 2, 3]),
        }
        for _ in range(4)
    ]


def test_facade_runs_eval_save_and_forwards_metrics(tmp_path):
    recorder = MetricsRecorder()
    trainer = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
        callbacks=[recorder],
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=1,
            sequence_length=4,
            logging_steps=1,
            eval_steps=1,
            save_steps=1,
        ),
    )

    state = trainer.train()

    assert state.step == 1
    assert any("loss" in metrics for metrics in recorder.metrics)
    assert any("eval_loss" in metrics for metrics in recorder.metrics)
    assert (tmp_path / "checkpoint-1" / "trainer_state.pt").is_file()


def test_facade_resumes_local_training_state(tmp_path):
    first = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=1,
            sequence_length=4,
            save_steps=1,
        ),
    )
    first.train()
    checkpoint = tmp_path / "checkpoint-1"

    resumed = TrainLMTrainer(
        model=TinyModel(),
        train_dataset=_dataset(),
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=2,
            sequence_length=4,
        ),
    )
    state = resumed.train(resume_from_checkpoint=checkpoint)

    assert state.step == 2
    assert state.micro_step == 2
