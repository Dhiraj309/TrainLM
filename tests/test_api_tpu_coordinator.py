import json
from pathlib import Path
import signal
import subprocess

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


def pinned_model():
    return ModelSourceConfig(
        provider="huggingface",
        initialization="pretrained",
        name_or_path="org/model",
        revision="a" * 40,
    )


def test_tpu_facade_defers_model_and_runtime_construction(tmp_path):
    coordinator = RecordingCoordinator()
    recorder = MetricsRecorder()
    trainer = TrainLMTrainer(
        model=pinned_model(),
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
        model=pinned_model(),
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
        model=pinned_model(),
        train_dataset=dataset,
        args=TrainLMTrainingArguments(
            accelerator="tpu", output_dir=tmp_path / "run", max_steps=1
        ),
    )
    trainer._tpu_coordinator = coordinator

    trainer.train()

    assert coordinator.requests[0].manifest_dir == manifest_dir


def test_tpu_facade_stages_evaluation_dataset_and_cadence(tmp_path):
    coordinator = RecordingCoordinator()
    trainer = TrainLMTrainer(
        model=pinned_model(),
        train_dataset=tmp_path / "train",
        eval_dataset=tmp_path / "eval",
        args=TrainLMTrainingArguments(
            accelerator="tpu",
            output_dir=tmp_path / "run",
            max_steps=4,
            eval_steps=2,
        ),
    )
    trainer._tpu_coordinator = coordinator

    trainer.train()

    request = coordinator.requests[0]
    assert request.eval_manifest_dir == tmp_path / "eval"
    assert request.eval_every_steps == 2


def test_tpu_facade_resolves_mutable_hugging_face_model_revision(
    tmp_path, monkeypatch
):
    trainer = TrainLMTrainer(
        model=ModelSourceConfig(
            provider="huggingface",
            initialization="pretrained",
            name_or_path="org/model",
            revision="main",
        ),
        train_dataset=tmp_path / "train",
        args=TrainLMTrainingArguments(accelerator="tpu", max_steps=1),
    )
    coordinator = RecordingCoordinator()
    trainer._tpu_coordinator = coordinator
    monkeypatch.setattr(
        "trainlm.api._resolve_hugging_face_model_revision",
        lambda repo_id, revision: "b" * 40,
    )

    trainer.train()

    assert coordinator.requests[0].model.revision == "b" * 40


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


def test_tpu_checkpoint_request_is_serialized_and_forwarded(tmp_path):
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    request = _TPURunRequest(
        **{
            **_request(tmp_path).to_dict(),
            "model": _request(tmp_path).model,
            "manifest_dir": tmp_path / "manifests",
            "output_dir": tmp_path / "run",
            "save_every_steps": 2,
            "resume_from_checkpoint": checkpoint,
        }
    )

    command = _TPUCoordinator(tmp_path / "worker.py")._command(request)

    assert command[command.index("--save-every-steps") + 1] == "2"
    assert command[command.index("--resume-from-checkpoint") + 1] == str(
        checkpoint.resolve()
    )
    assert request.to_dict()["resume_from_checkpoint"] == str(checkpoint)


def test_tpu_request_does_not_expose_or_assume_world_size(tmp_path):
    request = _request(tmp_path)

    assert "expected_world_size" not in request.to_dict()
    assert "--expected-world-size" not in _TPUCoordinator(
        tmp_path / "worker.py"
    )._command(request)


def test_tpu_config_model_is_serialized_for_worker_reconstruction(tmp_path):
    base = _request(tmp_path)
    model = ModelSourceConfig(
        provider="huggingface",
        initialization="config",
        model_type="llama",
        dtype="float32",
        config_overrides={"hidden_size": 1024, "num_hidden_layers": 8},
    )
    request = _TPURunRequest(
        **{
            **base.to_dict(),
            "model": model,
            "manifest_dir": tmp_path / "manifests",
            "output_dir": tmp_path / "run",
        }
    )

    command = _TPUCoordinator(tmp_path / "worker.py")._command(request)
    payload = json.loads(command[command.index("--model-source-json") + 1])

    assert payload["initialization"] == "config"
    assert payload["model_type"] == "llama"
    assert payload["config_overrides"] == {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
    }
    assert "--model-id" not in command


def test_tpu_checkpoint_request_rejects_missing_resume_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        _TPURunRequest(
            model=_request(tmp_path).model,
            manifest_dir=tmp_path / "manifests",
            output_dir=tmp_path / "run",
            max_steps=2,
            gradient_accumulation_steps=1,
            micro_batch_per_device=1,
            sequence_length=8,
            seed=1,
            log_every_steps=1,
            learning_rate=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            scheduler="constant",
            warmup_steps=0,
            precision="bf16",
            resume_from_checkpoint=tmp_path / "missing",
        )


def test_tpu_evaluation_request_requires_dataset_and_cadence_together(tmp_path):
    base = _request(tmp_path)
    values = {
        **base.to_dict(),
        "model": base.model,
        "manifest_dir": tmp_path / "manifests",
        "output_dir": tmp_path / "run",
        "eval_every_steps": 2,
    }
    with pytest.raises(ValueError, match="configured together"):
        _TPURunRequest(**values)


def test_tpu_evaluation_request_is_forwarded_to_worker(tmp_path):
    base = _request(tmp_path)
    request = _TPURunRequest(
        **{
            **base.to_dict(),
            "model": base.model,
            "manifest_dir": tmp_path / "manifests",
            "output_dir": tmp_path / "run",
            "eval_manifest_dir": tmp_path / "eval",
            "eval_every_steps": 2,
        }
    )

    command = _TPUCoordinator(tmp_path / "worker.py")._command(request)

    assert command[command.index("--eval-manifest-dir") + 1] == str(
        (tmp_path / "eval").resolve()
    )
    assert command[command.index("--eval-every-steps") + 1] == "2"


def test_coordinator_owns_stages_logs_and_structured_summary(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("# test worker\n", encoding="utf-8")
    request = _request(tmp_path)
    calls = []

    class Process:
        pid = 123

        def __init__(self, returncode=0):
            self.returncode = returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("stage passed\n")
        kwargs["stdout"].flush()
        if "--probe-only" not in command and "--model-preflight" not in command:
            request.output_dir.mkdir(parents=True, exist_ok=True)
            (request.output_dir / "summary.json").write_text(
                json.dumps({"phase": "finalized", "steps": 2}),
                encoding="utf-8",
            )
            (request.output_dir / "metrics.jsonl").write_text(
                '{"loss": 2.5, "step": 2}\n', encoding="utf-8"
            )
        return Process()

    monkeypatch.setattr(subprocess, "Popen", popen)

    summary = _TPUCoordinator(worker).run(request)

    assert summary["status"] == "completed"
    assert summary["completed_stages"] == ["probe", "model_preflight", "train"]
    assert summary["worker_summary"]["steps"] == 2
    assert summary["metrics"] == [{"loss": 2.5, "step": 2.0}]
    assert len(calls) == 3
    assert "--probe-only" in calls[0][0]
    assert "--model-preflight" in calls[1][0]
    assert "--learning-rate" in calls[2][0]
    assert all(call[1]["start_new_session"] is True for call in calls)
    assert (request.output_dir / "request.json").is_file()
    assert (request.output_dir / "coordinator_summary.json").is_file()
    assert (request.output_dir / "train.log").read_text() == "stage passed\n"


def test_coordinator_reports_actionable_stage_failure(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("# test worker\n", encoding="utf-8")
    request = _request(tmp_path)

    class FailedProcess:
        pid = 123

        def wait(self, timeout=None):
            del timeout
            return 9

        def poll(self):
            return 9

    def popen(command, **kwargs):
        del command
        kwargs["stdout"].write("PJRT launch failed\n")
        kwargs["stdout"].flush()
        return FailedProcess()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr("os.killpg", lambda pid, sig: None)

    with pytest.raises(TPUCoordinatorError, match="probe stage failed.*probe.log"):
        _TPUCoordinator(worker).run(request)

    summary = json.loads(
        (request.output_dir / "coordinator_summary.json").read_text()
    )
    assert summary["status"] == "failed"
    assert summary["completed_stages"] == []
    assert "PJRT launch failed" in summary["error"]


def test_coordinator_releases_worker_group_when_notebook_is_interrupted(
    tmp_path, monkeypatch
):
    worker = tmp_path / "worker.py"
    worker.write_text("# test worker\n", encoding="utf-8")
    process = type("InterruptProcess", (), {})()
    process.pid = 456
    process.running = True
    waits = 0

    def wait(timeout=None):
        nonlocal waits
        waits += 1
        if timeout is None and waits == 1:
            raise KeyboardInterrupt
        process.running = False
        return -15

    process.wait = wait
    process.poll = lambda: None if process.running else -15
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    signals = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(KeyboardInterrupt):
        _TPUCoordinator(worker).run(_request(tmp_path))

    assert signals == [(456, signal.SIGTERM)]
    assert process.running is False


def test_coordinator_rejects_malformed_metrics_artifact(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"loss": true}\n', encoding="utf-8")

    with pytest.raises(TPUCoordinatorError, match="Invalid TPU metric snapshot"):
        _TPUCoordinator._read_metrics(path)
