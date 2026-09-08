import json
from pathlib import Path
import subprocess

import pytest
import torch

from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm._tpu_coordinator import (
    TPUCoordinatorError,
    _TPUCoordinator,
    _TPURunRequest,
)
from trainlm.config import ModelSourceConfig


class RecordingCoordinator:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return {"status": "completed"}


def test_tpu_facade_defers_model_and_runtime_construction(tmp_path):
    coordinator = RecordingCoordinator()
    trainer = TrainLMTrainer(
        model="org/model",
        train_dataset=tmp_path / "manifests",
        args=TrainLMTrainingArguments(
            output_dir=tmp_path / "run",
            accelerator="tpu",
            bf16=True,
            max_steps=3,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=2,
            sequence_length=128,
        ),
    )
    trainer._tpu_coordinator = coordinator

    result = trainer.train()

    assert result == {"status": "completed"}
    assert trainer.model is None
    assert trainer.engine is None
    request = coordinator.requests[0]
    assert request.model.name_or_path == "org/model"
    assert request.manifest_dir == tmp_path / "manifests"
    assert request.output_dir == tmp_path / "run"
    assert request.max_steps == 3
    assert request.gradient_accumulation_steps == 4
    assert request.micro_batch_per_device == 2
    assert request.sequence_length == 128
    assert request.precision == "bf16"
    assert trainer.explain()["selected_path"] == "tpu_coordinator"


def test_tpu_facade_rejects_parent_owned_objects_and_unsupported_data(tmp_path):
    with pytest.raises(TypeError, match="each worker"):
        TrainLMTrainer(
            model=torch.nn.Linear(2, 2),
            train_dataset=tmp_path,
            args=TrainLMTrainingArguments(accelerator="tpu"),
        )

    trainer = TrainLMTrainer(
        model="org/model",
        train_dataset=[{"input_ids": [1, 2]}],
        args=TrainLMTrainingArguments(accelerator="tpu"),
    )
    trainer._tpu_coordinator = RecordingCoordinator()
    with pytest.raises(TypeError, match="local manifest directory"):
        trainer.train()


def _request(tmp_path):
    return _TPURunRequest(
        model=ModelSourceConfig(
            provider="huggingface",
            initialization="pretrained",
            name_or_path="org/model",
            revision="a" * 40,
        ),
        manifest_dir=tmp_path / "manifests",
        output_dir=tmp_path / "run",
        max_steps=2,
        gradient_accumulation_steps=4,
        micro_batch_per_device=2,
        sequence_length=128,
        seed=7,
        log_every_steps=1,
        learning_rate=1e-4,
        betas=(0.8, 0.9),
        eps=1e-7,
        weight_decay=0.01,
        scheduler="linear",
        warmup_steps=1,
        precision="bf16",
    )


def test_coordinator_owns_stages_logs_and_structured_summary(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("# test worker\n", encoding="utf-8")
    request = _request(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "--probe-only" not in command and "--model-preflight" not in command:
            request.output_dir.mkdir(parents=True, exist_ok=True)
            (request.output_dir / "summary.json").write_text(
                json.dumps({"phase": "finalized", "steps": 2}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="stage passed\n")

    monkeypatch.setattr(subprocess, "run", run)

    summary = _TPUCoordinator(worker).run(request)

    assert summary["status"] == "completed"
    assert summary["completed_stages"] == ["probe", "model_preflight", "train"]
    assert summary["worker_summary"]["steps"] == 2
    assert len(calls) == 3
    assert "--probe-only" in calls[0][0]
    assert "--model-preflight" in calls[1][0]
    assert "--learning-rate" in calls[2][0]
    assert all(call[1]["check"] is False for call in calls)
    assert (request.output_dir / "request.json").is_file()
    assert (request.output_dir / "coordinator_summary.json").is_file()
    assert (request.output_dir / "train.log").read_text() == "stage passed\n"


def test_coordinator_reports_actionable_stage_failure(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("# test worker\n", encoding="utf-8")
    request = _request(tmp_path)

    def run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 9, stdout="PJRT launch failed\n")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(TPUCoordinatorError, match="probe stage failed.*probe.log"):
        _TPUCoordinator(worker).run(request)

    summary = json.loads(
        (request.output_dir / "coordinator_summary.json").read_text()
    )
    assert summary["status"] == "failed"
    assert summary["completed_stages"] == []
    assert "PJRT launch failed" in summary["error"]
