"""CPU tests for built-in TrainLMTrainer console step logging."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

from trainlm.api import TrainLMTrainer, TrainLMTrainingArguments


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.proj(self.embed(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        return SimpleNamespace(loss=loss, logits=logits)


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def test_trainer_prints_step_logs_by_default_and_respects_quiet(tmp_path, capsys) -> None:
    trainer = TrainLMTrainer(
        model=_TinyModel(),
        train_dataset=_TinyDataset(),
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "normal"),
            max_steps=4,
            logging_steps=2,
            sequence_length=4,
            per_device_train_batch_size=2,
            accelerator="cpu",
            logging_verbosity="normal",
        ),
    )
    trainer.train()
    out = capsys.readouterr().out
    assert "step 2/4 | loss=" in out
    assert "step 4/4 | loss=" in out
    assert "tok/s" in out

    quiet_trainer = TrainLMTrainer(
        model=_TinyModel(),
        train_dataset=_TinyDataset(),
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "quiet"),
            max_steps=4,
            logging_steps=2,
            sequence_length=4,
            per_device_train_batch_size=2,
            accelerator="cpu",
            logging_verbosity="quiet",
        ),
    )
    quiet_trainer.train()
    quiet_out = capsys.readouterr().out
    assert quiet_out == ""
