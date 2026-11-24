from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from trainlm.api import TrainLMTrainingArguments
from trainlm.optimization import (
    apply_fused_rms_norm,
    restore_rms_norm,
)
from trainlm.training.trainer import Trainer


class _MockLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _MockGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * (1.0 + self.weight)


def test_apply_fused_rms_norm_patches_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(F, "rms_norm"):
        pytest.skip("torch.nn.functional.rms_norm unavailable")

    model = nn.Sequential(_MockLlamaRMSNorm(16), _MockGemmaRMSNorm(16))
    x = torch.randn(2, 4, 16)
    expected = model(x).detach()

    result = apply_fused_rms_norm(model)
    assert result.applied is True
    assert result.patched_modules == 1
    assert getattr(model[0], "_trainlm_rms_norm_patched", False) is True
    assert getattr(model[1], "_trainlm_rms_norm_patched", False) is False

    actual = model(x).detach()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    restored = restore_rms_norm(model)
    assert restored == 1
    assert getattr(model[0], "_trainlm_rms_norm_patched", False) is False


def test_use_fused_rms_norm_argument_validation() -> None:
    args = TrainLMTrainingArguments(output_dir="out")
    assert args.use_fused_rms_norm is True
    args_off = TrainLMTrainingArguments(output_dir="out", use_fused_rms_norm=False)
    assert args_off.use_fused_rms_norm is False
    with pytest.raises(ValueError, match="use_fused_rms_norm"):
        TrainLMTrainingArguments(output_dir="out", use_fused_rms_norm="auto")  # type: ignore[arg-type]


def test_trainer_sync_optimizer_lr_tensors_keeps_same_tensor_object() -> None:
    param = nn.Parameter(torch.randn(4, 4))
    optimizer = torch.optim.AdamW([param], lr=1e-3, capturable=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5 ** step)

    trainer = object.__new__(Trainer)
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.runtime = SimpleNamespace(device=torch.device("cpu"), world_size=1)
    trainer._compiled_single_step_active = True
    trainer._lr_tensors = None

    trainer._sync_optimizer_lr_tensors()
    lr_tensor = optimizer.param_groups[0]["lr"]
    assert isinstance(lr_tensor, torch.Tensor)
    assert lr_tensor.ndim == 0
    assert optimizer.param_groups[0]["capturable"] is True
    assert float(lr_tensor.item()) == pytest.approx(1e-3)

    trainer._advance_scheduler(total_tokens=16)
    lr_tensor_after = optimizer.param_groups[0]["lr"]
    assert lr_tensor_after is lr_tensor
    assert float(lr_tensor_after.item()) == pytest.approx(5e-4)
