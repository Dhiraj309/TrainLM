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
    assert opt._fast_path_verified is False

    param.grad = torch.ones_like(param)
    opt.step()
    assert opt._fast_path_verified is True
    opt.load_state_dict(opt.state_dict())
    assert opt._fast_path_verified is False


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
    monkeypatch.delenv("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", raising=False)
    monkeypatch.delenv("XLA_COMPILATION_CACHE_PATH", raising=False)
    cache_dir = tmp_path / "tpu_cache"

    rt = object.__new__(TorchTPURuntime)
    rt._compile_enabled = True
    rt._compile_cache_dir = str(cache_dir)
    rt._configure_compile_cache()

    assert cache_dir.is_dir()
    assert os.environ.get("TORCH_TPU_CACHE_DIR") == str(cache_dir)
    assert os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT") == str(cache_dir)
    assert os.environ.get("XLA_COMPILATION_CACHE_PATH") == str(cache_dir)


def test_is_pallas_ce_available_returns_bool() -> None:
    assert isinstance(is_pallas_ce_available(), bool)


def test_chunked_loss_pads_xla_tpu_tail_chunk_to_uniform_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    import torch.nn.functional as F
    from trainlm.tasks.chunked_loss import chunked_linear_causal_cross_entropy

    class _FakeXLATensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="xla")

    seen_chunk_rows: list[int] = []
    orig_linear = F.linear

    def _rec_linear(inp, weight, bias=None):
        seen_chunk_rows.append(int(inp.shape[0]))
        return orig_linear(
            inp.as_subclass(torch.Tensor),
            weight.as_subclass(torch.Tensor),
            None if bias is None else bias.as_subclass(torch.Tensor),
        )

    monkeypatch.setattr(F, "linear", _rec_linear)

    torch.manual_seed(9)
    # 2 * (6 - 1) = 10 shifted tokens -> with chunk_size=4, padded to 12 = [4, 4, 4].
    h_cpu = torch.randn(2, 6, 8, dtype=torch.float32, requires_grad=True)
    w_cpu = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 16, (2, 6))

    ref_loss, _ = chunked_linear_causal_cross_entropy(
        h_cpu, w_cpu, labels, chunk_size=4
    )
    ref_loss.backward()
    seen_chunk_rows.clear()

    h_xla = h_cpu.detach().clone().as_subclass(_FakeXLATensor).requires_grad_(True)
    w_xla = w_cpu.detach().clone().as_subclass(_FakeXLATensor).requires_grad_(True)
    xla_loss, _ = chunked_linear_causal_cross_entropy(
        h_xla, w_xla, labels, chunk_size=4
    )
    xla_loss.backward()

    assert seen_chunk_rows == [4, 4, 4]
    torch.testing.assert_close(xla_loss.as_subclass(torch.Tensor), ref_loss)
    torch.testing.assert_close(h_xla.grad.as_subclass(torch.Tensor), h_cpu.grad)
    torch.testing.assert_close(w_xla.grad.as_subclass(torch.Tensor), w_cpu.grad)


def test_chunked_linear_omits_loss_mask_transfer_when_all_supervised() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel
    from trainlm.api import _auto_training_view
    from trainlm.tasks import CausalLMTask

    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2)
    )
    view = _auto_training_view(model)
    assert view is not None
    task = CausalLMTask(loss_implementation="chunked_linear", training_view=view)
    prepared, counts = task.prepare_batch_on_host(
        {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])}
    )
    assert counts.ignored_tokens == 0
    assert "loss_mask" not in prepared


def test_compiled_lr_sync_skips_idle_fills_and_multiplicative_scheduler_item_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.optim.lr_scheduler import LinearLR
    from trainlm.runtime import Runtime
    from trainlm.training.trainer import Trainer

    param = nn.Parameter(torch.ones(4, 4))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = LinearLR(opt, start_factor=1.0, end_factor=0.5, total_iters=4)
    param.grad = torch.zeros_like(param)
    opt.step()

    tr = object.__new__(Trainer)
    tr.optimizer = opt
    tr.scheduler = sched
    tr.runtime = Runtime()
    tr._compiled_single_step_active = True
    tr._lr_tensors = None
    tr._lr_values = None

    tr._sync_optimizer_lr_tensors()
    assert tr._lr_tensors is not None and len(tr._lr_tensors) == 1
    lr_tensor = tr._lr_tensors[0]

    fill_calls: list[float] = []
    orig_fill = lr_tensor.fill_

    def _track_fill(val):
        fill_calls.append(float(val))
        return orig_fill(val)

    monkeypatch.setattr(lr_tensor, "fill_", _track_fill)
    monkeypatch.setattr(
        lr_tensor,
        "item",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected device .item() sync")),
    )

    # Unchanged LR must not call .fill_() or .item()
    tr._sync_optimizer_lr_tensors()
    assert fill_calls == []
    assert tr._current_learning_rate() == pytest.approx(1e-3)

    # Advancing LinearLR updates LR in Python floats without calling lr_tensor.item()
    tr._advance_scheduler(total_tokens=16)
    assert len(fill_calls) == 1
    assert fill_calls[0] == pytest.approx(0.875e-3)
    assert tr._current_learning_rate() == pytest.approx(0.875e-3)
    assert opt.param_groups[0]["lr"] is lr_tensor


def test_place_model_on_tpu_uses_torch_accelerator_not_torch_tpu_is_available(monkeypatch):
    """TorchTPU 0.2.0 forbids third-party torch.tpu.is_available() calls."""
    import types

    import pytest
    import torch

    from trainlm.api import TrainLMTrainer

    try:
        tpu_device = torch.device("tpu")
    except RuntimeError:
        pytest.skip("torch.device('tpu') unavailable without TorchTPU backend")

    def _forbidden():
        raise RuntimeError("is_available is internal to TorchTPU")

    monkeypatch.setattr(torch, "tpu", types.SimpleNamespace(is_available=_forbidden), raising=False)
    monkeypatch.setattr(
        torch, "accelerator", types.SimpleNamespace(is_available=lambda: True), raising=False
    )

    class _Model:
        config = types.SimpleNamespace(tie_word_embeddings=False)
        moved = None

        def to(self, **kwargs):
            self.moved = kwargs
            return self

    trainer = object.__new__(TrainLMTrainer)
    trainer.runtime = types.SimpleNamespace(device=tpu_device, precision="bf16")
    trainer.model = _Model()
    trainer._place_model_on_runtime_device()
    assert trainer.model.moved is not None
