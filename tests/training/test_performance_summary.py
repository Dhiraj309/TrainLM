from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainlm.benchmark.mfu import calculate_causal_lm_flops, calculate_mfu
from trainlm.training.performance import build_performance_summary


class TinyDenseLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 4)
        self.block = nn.Linear(4, 4, bias=False)
        self.output = nn.Linear(4, 10, bias=False)
        self.config = SimpleNamespace(
            vocab_size=10,
            hidden_size=4,
            num_hidden_layers=1,
            tie_word_embeddings=False,
            is_encoder_decoder=False,
        )


class FakeRuntime:
    def __init__(self, *, device_name="TPU v5e", precision="bf16", world_size=1):
        self._diagnostics = SimpleNamespace(
            backend="torchtpu",
            device_type="tpu",
            precision=precision,
            world_size=world_size,
            values={"device_name": device_name},
        )

    def diagnostics(self):
        return self._diagnostics


def test_summary_uses_measured_tokens_and_documented_v5e_peak_for_mfu():
    model = TinyDenseLM()
    measurements = {
        "measured_optimizer_steps": 5,
        "supervised_tokens": 100,
        "wall_seconds": 2.0,
        "tokens_per_second": 50.0,
        "dataloader_wait_seconds": 0.2,
        "dataloader_wait_fraction": 0.1,
    }

    summary = build_performance_summary(
        measurements=measurements,
        model=model,
        runtime=FakeRuntime(),
        sequence_length=8,
    )

    flops = calculate_causal_lm_flops(
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        vocabulary_size=10,
        hidden_size=4,
        num_hidden_layers=1,
        sequence_length=8,
        tie_word_embeddings=False,
    )
    assert summary["status"] == "measured"
    assert summary["tokens_per_second"] == 50.0
    assert summary["dataloader_wait_fraction"] == 0.1
    assert summary["mfu_status"] == "measured"
    assert summary["non_embedding_mfu"] == pytest.approx(
        calculate_mfu(
            tokens_per_second=50.0,
            flops_per_token=flops.non_embedding_flops_per_token,
            peak_flops_per_device=197_000_000_000_000.0,
            device_count=1,
        )
    )
    assert summary["hardware"]["peak_source"].endswith("/v5e")


@pytest.mark.parametrize(
    "runtime,reason",
    [
        (FakeRuntime(precision="fp32"), "No documented peak"),
        (FakeRuntime(world_size=8), "single-device"),
        (FakeRuntime(device_name="unknown TPU"), "No documented peak"),
    ],
)
def test_summary_does_not_guess_unknown_peak_or_distributed_mfu(runtime, reason):
    summary = build_performance_summary(
        measurements={
            "measured_optimizer_steps": 1,
            "supervised_tokens": 10,
            "wall_seconds": 1.0,
            "tokens_per_second": 10.0,
        },
        model=TinyDenseLM(),
        runtime=runtime,
        sequence_length=8,
    )

    assert summary["mfu_status"] == "unavailable"
    assert reason in summary["mfu_unavailable_reason"]


def test_summary_reports_measurement_missing_without_fabricating_throughput():
    summary = build_performance_summary(
        measurements={},
        model=TinyDenseLM(),
        runtime=FakeRuntime(),
        sequence_length=8,
    )

    assert summary["status"] == "not_measured"
    assert summary["tokens_per_second"] == 0.0
    assert summary["mfu_status"] == "unavailable"


def test_summary_computes_steady_state_excluding_warmup_window():
    summary = build_performance_summary(
        measurements={
            "measured_optimizer_steps": 15,
            "supervised_tokens": 300,
            "wall_seconds": 12.0,
            "tokens_per_second": 25.0,
            "dataloader_wait_seconds": 0.3,
            "dataloader_wait_fraction": 0.025,
            "recent_windows": [
                {"step": 5.0, "supervised_tokens": 100.0, "train_wall_seconds": 10.0, "dataloader_wait_seconds": 0.1},
                {"step": 10.0, "supervised_tokens": 100.0, "train_wall_seconds": 1.0, "dataloader_wait_seconds": 0.1},
                {"step": 15.0, "supervised_tokens": 100.0, "train_wall_seconds": 1.0, "dataloader_wait_seconds": 0.1},
            ],
        },
        model=TinyDenseLM(),
        runtime=FakeRuntime(),
        sequence_length=8,
    )

    assert summary["tokens_per_second"] == pytest.approx(25.0)
    steady = summary["steady_state"]
    assert steady["measured_windows"] == 2
    assert steady["supervised_tokens"] == 200
    assert steady["wall_seconds"] == pytest.approx(2.0)
    assert steady["tokens_per_second"] == pytest.approx(100.0)
    assert steady["non_embedding_mfu"] == pytest.approx(4.0 * summary["non_embedding_mfu"])

