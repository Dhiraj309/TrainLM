from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from trainlm.api import TrainLMTrainingArguments
from trainlm.config import OptimizerConfig
from trainlm.kernels.pallas import is_pallas_ce_available
from trainlm.optimization import OptimizerFactory
from trainlm.runtime import AsyncBatchPrefetcher
from trainlm.runtime.torchtpu import TorchTPURuntime


def test_adamw_bf16_zero_cast_fast_path() -> None:
    param = nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16))
    opt = OptimizerFactory.create(
        [param],
        OptimizerConfig(name="adamw_bf16", fused=False),
    )
    assert opt.state_policy.mu_dtype is torch.bfloat16
    assert opt.state_policy.nu_dtype is torch.bfloat16

    param.grad = torch.ones_like(param)
    opt.step()
    assert opt.state[param]["exp_avg"].dtype is torch.bfloat16
    assert opt.state[param]["exp_avg_sq"].dtype is torch.bfloat16
    assert opt._moments_match_grads_and_policy() is True


def test_single_chip_tpu_training_args_validation() -> None:
    args = TrainLMTrainingArguments(output_dir="out", optim="adamw_bf16")
    assert args.optim == "adamw_bf16"
    assert args.tpu_compile_cache_dir == "auto"
    assert args.prefetch_batches is True

    with pytest.raises(ValueError, match="tpu_compile_cache_dir"):
        TrainLMTrainingArguments(output_dir="out", tpu_compile_cache_dir="   ")
    with pytest.raises(ValueError, match="prefetch_batches"):
        TrainLMTrainingArguments(output_dir="out", prefetch_batches="yes")  # type: ignore[arg-type]


def test_async_batch_prefetcher_preserves_order_and_closes() -> None:
    batches = [{"id": i} for i in range(6)]
    prefetcher = AsyncBatchPrefetcher(iter(batches), transform=lambda b: {"id": b["id"] * 10})
    out = list(prefetcher)
    assert [x["id"] for x in out] == [0, 10, 20, 30, 40, 50]


def test_torchtpu_compile_cache_configures_directory_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TORCH_TPU_CACHE_DIR", raising=False)
    monkeypatch.delenv("XLA_COMPILATION_CACHE_PATH", raising=False)
    cache_dir = tmp_path / "tpu_cache"

    rt = object.__new__(TorchTPURuntime)
    rt._compile_enabled = True
    rt._compile_cache_dir = str(cache_dir)
    rt._configure_compile_cache()

    assert cache_dir.is_dir()
    assert os.environ.get("TORCH_TPU_CACHE_DIR") == str(cache_dir)
    assert os.environ.get("XLA_COMPILATION_CACHE_PATH") == str(cache_dir)


def test_is_pallas_ce_available_returns_bool() -> None:
    assert isinstance(is_pallas_ce_available(), bool)
