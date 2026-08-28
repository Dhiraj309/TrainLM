"""Generic trainer overfit checks across representative dense-AR HF families."""

from __future__ import annotations

import math

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from trainlm.config import LoggingConfig, TrainConfig, TrainerConfig
from trainlm.model import load_huggingface_causal_lm
from trainlm.runtime import Runtime
from trainlm.tasks import CausalLMTask
from trainlm.training import Trainer, TrainerCallback

from ..model.dense_ar_fixtures import DENSE_AR_FIXTURES, DenseARFixture


class RepeatedTokenDataset(Dataset):

    def __init__(self):
        self.input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        del index
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": torch.ones_like(self.input_ids),
        }


class LossAndGradientRecorder(TrainerCallback):

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.gradients_are_finite = True

    def on_step_end(self, state, control):
        del control
        self.losses.append(state.loss)
        self.gradients_are_finite &= all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )


@pytest.mark.parametrize("fixture", DENSE_AR_FIXTURES, ids=lambda item: item.name)
def test_generic_trainer_overfits_and_exports_each_dense_ar_family(
    fixture: DenseARFixture,
    tmp_path,
):
    torch.manual_seed(0)
    loaded = load_huggingface_causal_lm(fixture.source(tied=True))
    model = loaded.model
    optimizer = AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    recorder = LossAndGradientRecorder(model)
    trainer = Trainer(
        config=TrainConfig(
            trainer=TrainerConfig(max_steps=30),
            logging=LoggingConfig(log_every_steps=1000),
        ),
        model=model,
        runtime=Runtime(),
        optimizer=optimizer,
        scheduler=scheduler,
        task=CausalLMTask(loss_implementation="causal_lm"),
        train_dataloader=DataLoader(RepeatedTokenDataset(), batch_size=2),
        callbacks=[recorder],
    )

    state = trainer.train()

    assert state.step == 30
    assert len(recorder.losses) == 30
    assert all(
        loss is not None and math.isfinite(loss) for loss in recorder.losses
    )
    assert recorder.gradients_are_finite
    assert min(recorder.losses[1:]) < recorder.losses[0] * 0.8

    export_dir = tmp_path / fixture.name
    model.save_pretrained(export_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(
        export_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    assert type(reloaded).__module__.startswith("transformers.")
