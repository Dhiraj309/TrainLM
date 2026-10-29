import json
from pathlib import Path
import subprocess
from dataclasses import replace

import pytest
import torch

from trainlm import PackedBinDataset, TrainLMTrainer, TrainLMTrainingArguments
from trainlm._tpu_coordinator import (
    TPUCoordinatorError,
    _TPUCoordinator,
    _TPURunRequest,
)
from trainlm.config import ModelSourceConfig
from trainlm.training import TrainerCallback


class RecordingCoordinator:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return {
            "status": "completed",
            "worker_summary": {"steps": 3, "global_supervised_tokens": 96},
            "metrics": [{"step": 3.0, "loss": 1.5}],
        }


class MetricsRecorder(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_metrics(self, state, control, metrics):
        del state, control
        self.metrics.append(dict(metrics))


def test_tpu_facade_defers_model_and_runtime_construction(tmp_path):
    coordinator = RecordingCoordinator()
    recorder = MetricsRecorder()
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
        callbacks=[recorder],
    )
    trainer._tpu_coordinator = coordinator

    result = trainer.train()

    assert result["status"] == "completed"
    assert result["trainer_state"]["step"] == 3
    assert recorder.metrics == [{"step": 3.0, "loss": 1.5}]
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


def test_tpu_facade_accepts_public_packed_dataset(tmp_path, monkeypatch):
    dataset = object.__new__(PackedBinDataset)
    manifest_dir = tmp_path / "validated"
    monkeypatch.setattr(
        PackedBinDataset,
        "coordinator_manifest_dir",
        lambda self, output_dir: manifest_dir,
    )
    coordinator = RecordingCoordinator()
    trainer = TrainLMTrainer(
        model="org/model",
        train_dataset=dataset,
        args=TrainLMTrainingArguments(
            accelerator="tpu", output_dir=tmp_path / "run", max_steps=1
        ),
    )
    trainer._tpu_coordinator = coordinator

    trainer.train()

    assert coordinator.requests[0].manifest_dir == manifest_dir


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
            (request.output_dir / "metrics.jsonl").write_text(
                '{"loss": 2.5, "step": 2}\n', encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, stdout="stage passed\n")

    monkeypatch.setattr(subprocess, "run", run)

    summary = _TPUCoordinator(worker).run(request)

    assert summary["status"] == "completed"
    assert summary["completed_stages"] == ["probe", "model_preflight", "train"]
    assert summary["worker_summary"]["steps"] == 2
    assert summary["metrics"] == [{"loss": 2.5, "step": 2.0}]
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


def test_coordinator_rejects_malformed_metrics_artifact(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"loss": true}\n', encoding="utf-8")

    with pytest.raises(TPUCoordinatorError, match="Invalid TPU metric snapshot"):
        _TPUCoordinator._read_metrics(path)


def test_tpu_facade_passes_checkpoint_and_evaluation_options(tmp_path):
    recorder = RecordingCoordinator()
    trainer = TrainLMTrainer(
        model="org/model", train_dataset=tmp_path / "training",
        eval_dataset=tmp_path / "validation",
        args=TrainLMTrainingArguments(
            accelerator="tpu", output_dir=tmp_path / "run", max_steps=4,
            save_steps=2, eval_steps=1, per_device_eval_batch_size=3,
        ),
    )
    trainer._tpu_coordinator = recorder
    source = tmp_path / "old-run" / "checkpoint-2"
    trainer.train(resume_from_checkpoint=source)
    request = recorder.requests[0]
    assert request.resume_from_checkpoint == source
    assert request.eval_manifest_dir == tmp_path / "validation"
    assert (request.save_steps, request.eval_steps, request.eval_batch_size) == (2, 1, 3)
    command = _TPUCoordinator()._command(request)
    for flag, value in (("--resume-from-checkpoint", str(source)),
                        ("--eval-manifest-dir", str(tmp_path / "validation")),
                        ("--save-steps", "2"), ("--eval-steps", "1"),
                        ("--eval-batch-size", "3")):
        assert command[command.index(flag) + 1] == value
    json.dumps(request.to_dict())


def test_coordinator_rejects_stale_success_artifacts(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("# worker\n")
    request = _request(tmp_path)
    request.output_dir.mkdir()
    (request.output_dir / "summary.json").write_text('{"steps": 2}')
    (request.output_dir / "metrics.jsonl").write_text('{"loss": 1}\n')
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout=""))
    with pytest.raises(TPUCoordinatorError, match="did not write summary"):
        _TPUCoordinator(worker).run(request)


def test_public_tpu_methods_reuse_latest_checkpoint_and_isolate_operation_artifacts(tmp_path):
    class LifecycleCoordinator(RecordingCoordinator):
        def run(self, request):
            self.requests.append(request)
            summary = {"steps": 2, "operation": request.operation}
            if request.operation in {"train", "save_state"}:
                summary["final_checkpoint"] = str(request.final_checkpoint_dir)
            elif request.operation == "export":
                summary["export_dir"] = str(request.export_dir)
            elif request.operation == "evaluate":
                summary["evaluation_metrics"] = {"eval_loss": 1.25}
            return {"status": "completed", "worker_summary": summary,
                    "metrics": [{"eval_loss": 1.25}] if request.operation == "evaluate" else []}
    coordinator = LifecycleCoordinator()
    callback = MetricsRecorder()
    trainer = TrainLMTrainer(
        model="org/model", train_dataset=tmp_path / "train", eval_dataset=tmp_path / "validation",
        args=TrainLMTrainingArguments(accelerator="tpu", output_dir=tmp_path / "run",
                                     max_steps=None, max_tokens=100),
        callbacks=[callback],
    )
    trainer._tpu_coordinator = coordinator
    trainer.train()
    original_checkpoint = coordinator.requests[0].final_checkpoint_dir
    assert coordinator.requests[0].max_steps is None
    assert coordinator.requests[0].max_tokens == 100
    assert trainer.evaluate() == {"eval_loss": 1.25}
    assert trainer.save_model(tmp_path / "export") == tmp_path / "export"
    saved = trainer.save_state(tmp_path / "states")
    assert saved.parent == tmp_path / "states"
    assert saved == coordinator.requests[-1].final_checkpoint_dir
    assert trainer._tpu_checkpoint == saved
    for request in coordinator.requests[1:]:
        assert request.resume_from_checkpoint == original_checkpoint
        assert request.output_dir != tmp_path / "run"
    assert len({request.output_dir for request in coordinator.requests}) == 4
    assert callback.metrics == [{"eval_loss": 1.25}]
    trainer.evaluate()
    assert coordinator.requests[-1].resume_from_checkpoint == saved
    assert trainer.model is None and trainer.engine is None


def test_tpu_evaluation_requires_data_without_launching_workers(tmp_path):
    trainer = TrainLMTrainer(model="org/model", train_dataset=tmp_path,
                            args=TrainLMTrainingArguments(accelerator="tpu"))
    coordinator = RecordingCoordinator()
    trainer._tpu_coordinator = coordinator
    with pytest.raises(ValueError, match="eval_dataset"):
        trainer.evaluate()
    assert coordinator.requests == []


@pytest.mark.parametrize("operation", ["evaluate", "export", "save_state"])
def test_coordinator_routes_standalone_worker_modes(tmp_path, monkeypatch, operation):
    worker = tmp_path / "worker.py"
    worker.write_text("# worker\n")
    request = replace(_request(tmp_path), operation=operation,
                      final_checkpoint_dir=tmp_path / "state", export_dir=tmp_path / "export",
                      max_steps=None, max_tokens=100)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if "--probe-only" not in command and "--model-preflight" not in command:
            (request.output_dir / "summary.json").write_text(json.dumps({"operation": operation}))
        return subprocess.CompletedProcess(command, 0, stdout="passed")
    monkeypatch.setattr(subprocess, "run", run)
    summary = _TPUCoordinator(worker).run(request)
    assert summary["completed_stages"] == ["probe", "model_preflight", operation]
    assert commands[-1][commands[-1].index("--operation") + 1] == operation
    assert commands[-1][commands[-1].index("--max-steps") + 1] == "None"
    assert commands[-1][commands[-1].index("--max-tokens") + 1] == "100"
    assert (request.output_dir / f"{operation}.log").is_file()
    json.dumps(request.to_dict())


@pytest.mark.parametrize("preflight_fails", [False, True])
def test_optimization_preflight_precedes_every_accelerator_stage(tmp_path, monkeypatch, preflight_fails):
    worker = tmp_path / "worker.py"
    worker.write_text("# worker\n")
    request = replace(_request(tmp_path), optimization_factory="fixture:profile", optimization_policy="required")
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if "--optimization-preflight" in command:
            if preflight_fails:
                return subprocess.CompletedProcess(command, 1, stdout="unsupported required shape")
            (request.output_dir / "optimization_preflight.json").write_text(json.dumps({"execution_plan": {"plan_id": "verified"}}))
        elif "--probe-only" not in command and "--model-preflight" not in command:
            (request.output_dir / "summary.json").write_text('{"steps": 2}')
        return subprocess.CompletedProcess(command, 0, stdout="passed")
    monkeypatch.setattr(subprocess, "run", run)
    if preflight_fails:
        with pytest.raises(TPUCoordinatorError, match="optimization_preflight stage failed"):
            _TPUCoordinator(worker).run(request)
        assert len(commands) == 1
    else:
        result = _TPUCoordinator(worker).run(request)
        assert result["completed_stages"] == ["optimization_preflight", "probe", "model_preflight", "train"]
        assert commands[1][commands[1].index("--optimization-plan-id") + 1] == "verified"
    assert "--optimization-preflight" in commands[0]
