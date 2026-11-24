"""CPU tests for Step 4: 8-bit optimizers and PEFT/LoRA integration."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from trainlm.api import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.config import OptimizerConfig
from trainlm.optimization.optimizers import create_optimizer


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.proj(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return SimpleNamespace(loss=loss, logits=logits)


def test_training_arguments_validate_optim(tmp_path) -> None:
    for optim in (
        "adamw",
        "adamw_torch",
        "adamw_8bit",
        "paged_adamw_8bit",
        "torchao_adamw_8bit",
    ):
        args = TrainLMTrainingArguments(output_dir=str(tmp_path / optim), optim=optim)
        assert args.optim == optim

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad"),
            optim="sgd_8bit",  # type: ignore[arg-type]
        )


def test_optimizer_factory_filters_frozen_parameters() -> None:
    frozen = nn.Parameter(torch.ones(4), requires_grad=False)
    trainable = nn.Parameter(torch.ones(4), requires_grad=True)
    opt = create_optimizer(
        [frozen, trainable],
        OptimizerConfig(name="adamw", fused=False),
    )
    params_in_opt = [p for group in opt.param_groups for p in group["params"]]
    assert params_in_opt == [trainable]


def test_optimizer_factory_8bit_backends_with_mocks(monkeypatch) -> None:
    class Fake8BitOptim(torch.optim.SGD):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
            super().__init__(params, lr=lr)
            self.kwargs = {"betas": betas, "eps": eps, "weight_decay": weight_decay}

    fake_bnb = ModuleType("bitsandbytes")
    fake_bnb.optim = SimpleNamespace(
        AdamW8bit=Fake8BitOptim,
        PagedAdamW8bit=Fake8BitOptim,
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

    p = nn.Parameter(torch.ones(4), requires_grad=True)
    opt1 = create_optimizer([p], OptimizerConfig(name="adamw_8bit", fused=False))
    opt2 = create_optimizer([p], OptimizerConfig(name="paged_adamw_8bit", fused=False))
    assert isinstance(opt1, Fake8BitOptim)
    assert isinstance(opt2, Fake8BitOptim)

    fake_torchao = ModuleType("torchao")
    fake_torchao_optim = ModuleType("torchao.optim")
    fake_torchao_optim.AdamW8bit = Fake8BitOptim  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.optim", fake_torchao_optim)

    opt3 = create_optimizer([p], OptimizerConfig(name="torchao_adamw_8bit", fused=False))
    assert isinstance(opt3, Fake8BitOptim)


def test_optimizer_factory_8bit_missing_dependency_message(monkeypatch) -> None:
    real_import = builtins.__import__

    def block_bnb(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ModuleNotFoundError("No module named 'bitsandbytes'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_bnb)
    p = nn.Parameter(torch.ones(4), requires_grad=True)
    with pytest.raises(RuntimeError, match="pip install bitsandbytes"):
        create_optimizer([p], OptimizerConfig(name="adamw_8bit", fused=False))


def test_trainer_peft_config_wraps_model_and_trains_only_adapters(tmp_path, monkeypatch) -> None:
    def fake_get_peft_model(model, peft_config):
        for param in model.parameters():
            param.requires_grad_(False)
        model.lora_adapter = nn.Parameter(torch.ones(8, 8), requires_grad=True)
        model.peft_config = {"default": peft_config}
        return model

    fake_peft = ModuleType("peft")
    fake_peft.get_peft_model = fake_get_peft_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    model = _TinyModel()
    trainer = TrainLMTrainer(
        model=model,
        train_dataset=_TinyDataset(),
        peft_config={"r": 4},
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "peft"),
            max_steps=1,
            sequence_length=4,
            per_device_train_batch_size=2,
            accelerator="cpu",
            optim="adamw_torch",
        ),
    )
    assert hasattr(trainer.model, "peft_config")
    opt_params = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    assert len(opt_params) == 1
    assert opt_params[0] is model.lora_adapter
