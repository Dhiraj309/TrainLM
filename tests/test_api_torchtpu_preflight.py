"""Trainer-facing non-allocating TorchTPU preflight behavior."""

import json

import pytest

import trainlm.api as api
from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.config import ModelSourceConfig


def _tpu_report():
    return {
        "status": "available_unverified",
        "device_type": "tpu",
        "allocation": "not_run",
        "device_name": None,
    }


def test_explicit_xla_route_reports_native_preflight_without_launch(monkeypatch, tmp_path):
    calls = []

    def discover(_torch, **kwargs):
        calls.append(kwargs)
        return _tpu_report()

    monkeypatch.setattr(api, "discover_tpu", discover)
    trainer = TrainLMTrainer(
        model=ModelSourceConfig(
            provider="huggingface", initialization="pretrained",
            name_or_path="org/model", revision="a" * 40,
        ),
        train_dataset=tmp_path / "train",
        args=TrainLMTrainingArguments(accelerator="tpu", max_steps=1),
    )
    assert calls == []
    assert trainer.explain()["backend"] == "xla"
    assert calls == []

    preflight = trainer.preflight()
    assert preflight["selected_backend"] == "xla"
    assert preflight["native_tpu_training"] == "not_integrated"
    assert preflight["native_tpu"]["allocation"] == "not_run"
    assert calls == [{"verify_allocation": False, "include_properties": False}]
    assert trainer.explain(include_hardware=True)["preflight"] == preflight
    assert json.loads(trainer.explain(format="json", include_hardware=True))[
        "preflight"
    ] == preflight
    assert "Native TPU preflight: available_unverified" in trainer.explain(
        format="text", include_hardware=True
    )


def test_auto_does_not_fall_back_to_cpu_when_native_tpu_detected(monkeypatch):
    monkeypatch.setattr(api, "discover_tpu", lambda _torch, **_kwargs: _tpu_report())
    trainer = object.__new__(TrainLMTrainer)
    trainer.args = TrainLMTrainingArguments(accelerator="auto")
    with pytest.raises(RuntimeError, match="Native TorchTPU was detected"):
        trainer._make_runtime()
