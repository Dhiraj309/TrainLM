from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from trainlm.api import TrainLMTrainingArguments
from trainlm.benchmark.mfu import calculate_causal_lm_flops, calculate_mfu
from trainlm.runtime import prepare_distributed_model
from trainlm.training.performance import build_performance_summary


class _TinyDenseLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            tie_word_embeddings=True,
        )
        self.layers = nn.ModuleList([nn.Linear(8, 8)])
        self.weight = nn.Parameter(torch.ones(100))


def test_distributed_strategy_argument_validation() -> None:
    args = TrainLMTrainingArguments(output_dir="out")
    assert args.distributed_strategy == "auto"
    assert args.ddp_bucket_cap_mb == 50

    args_fsdp = TrainLMTrainingArguments(
        output_dir="out",
        distributed_strategy="fsdp2",
        ddp_bucket_cap_mb=100,
    )
    assert args_fsdp.distributed_strategy == "fsdp2"
    assert args_fsdp.ddp_bucket_cap_mb == 100

    with pytest.raises(ValueError, match="distributed_strategy"):
        TrainLMTrainingArguments(output_dir="out", distributed_strategy="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ddp_bucket_cap_mb"):
        TrainLMTrainingArguments(output_dir="out", ddp_bucket_cap_mb=0)


def test_prepare_distributed_model_single_process_behavior() -> None:
    model = _TinyDenseLM()
    wrapped, result = prepare_distributed_model(model, strategy="auto")
    assert wrapped is model
    assert result.wrapped is False
    assert result.strategy == "none"

    with pytest.raises(RuntimeError, match="world_size > 1"):
        prepare_distributed_model(model, strategy="fsdp2")
    with pytest.raises(RuntimeError, match="world_size > 1"):
        prepare_distributed_model(model, strategy="ddp")


def test_build_performance_summary_supports_world_size_gt_1_with_global_tps() -> None:
    model = _TinyDenseLM()
    runtime = SimpleNamespace(
        diagnostics=lambda: SimpleNamespace(
            backend="torchtpu",
            device_type="tpu",
            precision="bf16",
            world_size=8,
            values={"device_name": "TPU v5e"},
        )
    )
    summary = build_performance_summary(
        measurements={
            "measured_optimizer_steps": 4,
            "supervised_tokens": 100,
            "wall_seconds": 2.0,
            "tokens_per_second": 50.0,
            "global_tokens_per_second": 400.0,
            "recent_windows": [
                {"train_wall_seconds": 1.0, "supervised_tokens": 50, "dataloader_wait_seconds": 0.0},
                {"train_wall_seconds": 1.0, "supervised_tokens": 50, "dataloader_wait_seconds": 0.0},
            ],
        },
        model=model,
        runtime=runtime,
        sequence_length=8,
    )
    flops = calculate_causal_lm_flops(
        parameter_count=sum(p.numel() for p in model.parameters()),
        vocabulary_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        sequence_length=8,
        tie_word_embeddings=True,
    )
    expected_mfu = calculate_mfu(
        tokens_per_second=400.0,
        flops_per_token=flops.non_embedding_flops_per_token,
        peak_flops_per_device=197_000_000_000_000.0,
        device_count=8,
    )
    assert summary["mfu_status"] == "measured"
    assert summary["hardware"]["device_count"] == 8
    assert summary["non_embedding_mfu"] == pytest.approx(expected_mfu)
    assert summary["steady_state"]["tokens_per_second"] == pytest.approx(400.0)
    assert summary["steady_state"]["non_embedding_mfu"] == pytest.approx(expected_mfu)
