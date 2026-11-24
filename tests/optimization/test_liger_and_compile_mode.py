"""CPU/mocked tests for Step 6: Liger-Kernel auto-wiring and torch_compile_mode."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from trainlm.api import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.optimization.liger import LigerKernelResult, apply_liger_kernels
from trainlm.runtime.runtime import TorchRuntime


class _TinyLlama(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama")
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
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def test_training_arguments_validate_liger_and_compile_mode(tmp_path) -> None:
    args = TrainLMTrainingArguments(
        output_dir=str(tmp_path / "ok"),
        use_liger_kernel=True,
        torch_compile_mode="max-autotune",
    )
    assert args.use_liger_kernel is True
    assert args.torch_compile_mode == "max-autotune"

    with pytest.raises(ValueError, match="use_liger_kernel"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad-liger"),
            use_liger_kernel="yes",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="torch_compile_mode"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad-mode"),
            torch_compile_mode="fast",  # type: ignore[arg-type]
        )


def test_torch_runtime_passes_compile_mode(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        "trainlm.runtime.runtime.torch.compile",
        lambda fn, **kwargs: calls.append((fn, kwargs)) or fn,
    )
    rt = TorchRuntime(
        device="cuda",
        precision="bf16",
        compile_enabled=True,
        compile_mode="max-autotune",
    )
    model = nn.Linear(4, 4)
    rt.compile_model(model)
    step = lambda b, c: (b, c)
    rt.compile_training_step(step)
    assert calls == [
        (model, {"mode": "max-autotune"}),
        (step, {"mode": "max-autotune"}),
    ]


def test_apply_liger_kernels_auto_noop_and_strict_error(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert apply_liger_kernels(_TinyLlama(), strict=False) is None
    with pytest.raises(RuntimeError, match="pip install liger-kernel"):
        apply_liger_kernels(_TinyLlama(), strict=True)


def test_apply_liger_kernels_wires_model_type_fn(monkeypatch) -> None:
    recorded: list[dict[str, object]] = []

    def apply_llama(
        *,
        rope: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        model=None,
    ):
        recorded.append(
            {
                "rope": rope,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "model": model,
            }
        )

    fake_liger = ModuleType("liger_kernel")
    fake_tx = ModuleType("liger_kernel.transformers")
    fake_mp = ModuleType("liger_kernel.transformers.monkey_patch")
    fake_mp.MODEL_TYPE_TO_APPLY_LIGER_FN = {"llama": apply_llama}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liger_kernel", fake_liger)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", fake_tx)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.monkey_patch", fake_mp)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    model = _TinyLlama()
    res = apply_liger_kernels(model, strict=True)
    assert res == LigerKernelResult(
        model_type="llama",
        fused_rms_norm=True,
        fused_rope=True,
        fused_swiglu=True,
        fused_linear_cross_entropy=True,
    )
    assert len(recorded) == 1
    assert recorded[0]["model"] is model
    assert recorded[0]["fused_linear_cross_entropy"] is True


def test_trainer_wires_use_liger_kernel(tmp_path, monkeypatch) -> None:
    called: list[str] = []

    def apply_llama(*, model=None, swiglu=True, fused_linear_cross_entropy=True):
        called.append("llama")

    fake_liger = ModuleType("liger_kernel")
    fake_tx = ModuleType("liger_kernel.transformers")
    fake_mp = ModuleType("liger_kernel.transformers.monkey_patch")
    fake_mp.MODEL_TYPE_TO_APPLY_LIGER_FN = {"llama": apply_llama}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liger_kernel", fake_liger)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", fake_tx)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.monkey_patch", fake_mp)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    trainer = TrainLMTrainer(
        model=_TinyLlama(),
        train_dataset=_TinyDataset(),
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "liger"),
            max_steps=1,
            sequence_length=4,
            per_device_train_batch_size=1,
            accelerator="cpu",
            use_liger_kernel=True,
        ),
    )
    assert called == ["llama"]
    assert trainer._liger_result is not None
    assert trainer._liger_result.model_type == "llama"
