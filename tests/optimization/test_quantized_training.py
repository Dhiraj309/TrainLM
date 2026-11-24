"""CPU tests for Step 5: opt-in FP8 and experimental INT8 pretraining."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from trainlm.api import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.optimization.quantized_training import (
    QuantizedTrainingResult,
    apply_quantized_training,
)


class _TinyModel(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.unaligned = nn.Linear(hidden, 10)
        self.lm_head = nn.Linear(hidden, 32, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.proj(input_ids.float())
        logits = self.lm_head(x)
        return SimpleNamespace(loss=logits.sum(), logits=logits)


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.ones(16, dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def test_training_arguments_validate_quantized_training(tmp_path) -> None:
    for mode in ("none", "fp8", "int8"):
        args = TrainLMTrainingArguments(
            output_dir=str(tmp_path / mode),
            quantized_training=mode,
        )
        assert args.quantized_training == mode

    with pytest.raises(ValueError, match="quantized_training"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad"),
            quantized_training="int4",  # type: ignore[arg-type]
        )


def test_apply_quantized_training_fp8_filters_lm_head_and_unaligned(monkeypatch) -> None:
    converted: list[str] = []

    def fake_convert(model, *, module_filter_fn):
        for name, mod in model.named_modules():
            if module_filter_fn(mod, name):
                converted.append(name)

    fake_torchao = ModuleType("torchao")
    fake_float8 = ModuleType("torchao.float8")
    fake_float8.convert_to_float8_training = fake_convert  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", fake_float8)

    model = _TinyModel(hidden=16)
    res = apply_quantized_training(model, mode="fp8")
    assert res == QuantizedTrainingResult(
        mode="fp8",
        backend="torchao.float8",
        excluded_modules=("lm_head", "embed_tokens"),
    )
    assert converted == ["proj"]


def test_apply_quantized_training_int8_warns_on_small_hidden_and_converts(monkeypatch) -> None:
    seen_configs: list[object] = []
    converted: list[str] = []

    class FakeInt8Config:
        def __init__(self, *, output: bool, grad_input: bool, grad_weight: bool) -> None:
            self.output = output
            self.grad_input = grad_input
            self.grad_weight = grad_weight

    def fake_quantize(model, config, *, filter_fn):
        seen_configs.append(config)
        for name, mod in model.named_modules():
            if filter_fn(mod, name):
                converted.append(name)

    fake_torchao = ModuleType("torchao")
    fake_proto = ModuleType("torchao.prototype")
    fake_qt = ModuleType("torchao.prototype.quantized_training")
    fake_qt.Int8MixedPrecisionTrainingConfig = FakeInt8Config  # type: ignore[attr-defined]
    fake_quant = ModuleType("torchao.quantization")
    fake_quant.quantize_ = fake_quantize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.prototype", fake_proto)
    monkeypatch.setitem(sys.modules, "torchao.prototype.quantized_training", fake_qt)
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_quant)

    model = _TinyModel(hidden=512)
    with pytest.warns(UserWarning, match="hidden_size=512"):
        res = apply_quantized_training(model, mode="int8")

    assert res.mode == "int8"
    assert converted == ["proj"]
    assert len(seen_configs) == 1


def test_apply_quantized_training_missing_torchao_raises(monkeypatch) -> None:
    real_import = builtins.__import__

    def block_torchao(name, *args, **kwargs):
        if name.startswith("torchao"):
            raise ModuleNotFoundError("No module named 'torchao'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_torchao)
    with pytest.raises(RuntimeError, match="pip install torchao"):
        apply_quantized_training(_TinyModel(), mode="fp8")


def test_trainer_wires_quantized_training(tmp_path, monkeypatch) -> None:
    called: list[str] = []
    fake_torchao = ModuleType("torchao")
    fake_float8 = ModuleType("torchao.float8")
    fake_float8.convert_to_float8_training = lambda m, module_filter_fn: called.append("fp8")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", fake_float8)

    trainer = TrainLMTrainer(
        model=_TinyModel(),
        train_dataset=_TinyDataset(),
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "fp8"),
            max_steps=1,
            sequence_length=16,
            per_device_train_batch_size=1,
            accelerator="cpu",
            quantized_training="fp8",
        ),
    )
    assert called == ["fp8"]
    assert trainer._quantized_training_result.mode == "fp8"
