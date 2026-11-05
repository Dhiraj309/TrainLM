"""Execute the notebook's saved recovery example on CPU and inspect its ZIP."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest
import torch

import trainlm
from trainlm import TrainLMPretrainingConfig, TrainLMTrainer
from trainlm._notebook_resume import build_resume_report


def test_resume_example_round_trips_all_settings_and_quoted_paths(
    tmp_path, monkeypatch,
):
    config = TrainLMPretrainingConfig(
        output_dir=tmp_path / "Dhiraj's run",
        accelerator="cpu",
        model_config_overrides={
            "custom_metadata": {
                "text": "'); raise RuntimeError('not code'); #\nCafé",
                "enabled": True,
                "optional": None,
            },
        },
        save_total_limit=3,
    )
    checkpoint = config.output_dir / "checkpoints" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    for name in ("run_config.json", "preparation_cache.json"):
        (config.output_dir / name).write_text("{}")
    calls = []

    class RecordingTrainer:
        @staticmethod
        def from_scratch(restored_config, *, extend_training_budget):
            calls.append((restored_config, extend_training_budget))
            return RecordingTrainer()

        def train(self, *, resume_from_checkpoint):
            calls.append(resume_from_checkpoint)
            return SimpleNamespace(step=2, tokens_seen=8)

    monkeypatch.setattr(trainlm, "TrainLMTrainer", RecordingTrainer)
    report = build_resume_report(
        config, trainer=SimpleNamespace(), checkpoint={"latest_path": str(checkpoint)},
    )
    report = json.loads(json.dumps(report))
    namespace = {}
    exec(compile(report["code"], "resume.py", "exec"), namespace)

    expected_settings = asdict(config)
    expected_settings["output_dir"] = str(config.output_dir)
    assert report["pretraining_config"] == expected_settings
    assert report["status"] == "available"
    assert report["script"] == "resume.py"
    assert calls == [(config, False), checkpoint]


@pytest.mark.parametrize(
    ("trainer", "checkpoint", "reason"),
    [
        (None, {"latest_path": None}, "no_checkpoint"),
        (SimpleNamespace(), {"latest_path": None}, "no_checkpoint"),
        (
            SimpleNamespace(),
            {"latest_path": None, "lookup_error_type": "PermissionError"},
            "checkpoint_lookup_failed",
        ),
    ],
)
def test_resume_example_is_unavailable_without_a_checkpoint(
    tmp_path, trainer, checkpoint, reason,
):
    report = build_resume_report(
        TrainLMPretrainingConfig(output_dir=tmp_path),
        trainer=trainer,
        checkpoint=checkpoint,
    )
    assert report["status"] == "unavailable"
    assert report["reason"] == reason
    assert report["code"] is None
    assert report["script"] is None
    json.dumps(report)


@pytest.mark.parametrize(
    "missing", ["checkpoint", "run_config.json", "preparation_cache.json"],
)
def test_resume_example_rejects_missing_run_before_constructing_trainer(
    tmp_path, monkeypatch, missing,
):
    config = TrainLMPretrainingConfig(output_dir=tmp_path)
    checkpoint = tmp_path / "checkpoints" / "checkpoint-2"
    if missing != "checkpoint":
        checkpoint.mkdir(parents=True)
    for name in ("run_config.json", "preparation_cache.json"):
        if missing != name:
            (tmp_path / name).write_text("{}")

    def unexpected_construction(*args, **kwargs):
        pytest.fail("Missing recovery files must not start a fresh trainer.")

    monkeypatch.setattr(TrainLMTrainer, "from_scratch", unexpected_construction)
    report = build_resume_report(
        config, trainer=SimpleNamespace(), checkpoint={"latest_path": str(checkpoint)},
    )
    with pytest.raises(FileNotFoundError, match="Restore the full original run"):
        exec(report["code"], {})


@pytest.mark.parametrize("extension", [None, "pending", "committed"])
def test_saved_example_resumes_real_cpu_state_and_budget_extension(
    tmp_path, monkeypatch, extension,
):
    datasets = ModuleType("datasets")
    rows = [
        {"text": f"A deterministic training story number {i}. " * 5}
        for i in range(16)
    ]
    datasets.load_dataset = lambda *args, **kwargs: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    settings = dict(
        output_dir=tmp_path / "artifacts",
        dataset_id="org/tiny-stories",
        dataset_revision="a" * 40,
        max_tokens=8,
        sequence_length=8,
        max_train_examples=12,
        max_eval_examples=8,
        tokenizer_vocab_size=260,
        model_embedding_size=16,
        model_layers=1,
        model_attention_heads=2,
        accelerator="cpu",
        save_total_limit=1,
    )
    config = TrainLMPretrainingConfig(**settings)
    trainer = TrainLMTrainer.from_scratch(config)
    trainer.train()
    if extension is not None:
        settings["max_tokens"] = 22
        config = TrainLMPretrainingConfig(**settings)
        trainer = TrainLMTrainer.from_scratch(config, extend_training_budget=True)
        if extension == "pending":
            def fail_manifest_publish(*args, **kwargs):
                raise OSError("simulated manifest publication failure")

            monkeypatch.setattr(
                trainer, "_write_scratch_run_configuration", fail_manifest_publish,
            )
            with pytest.raises(OSError, match="simulated manifest publication"):
                trainer.train(resume_from_checkpoint="latest")
        else:
            trainer.train(resume_from_checkpoint="latest")

    checkpoint = trainer.latest_checkpoint
    report = build_resume_report(
        config, trainer=trainer, checkpoint={"latest_path": str(checkpoint)},
    )
    assert report["extend_training_budget"] is (extension == "pending")
    namespace = {}
    exec(report["code"], namespace)

    assert namespace["config"] == config
    assert namespace["checkpoint"] == checkpoint
    assert namespace["state"].tokens_seen >= config.max_tokens
    assert namespace["state"].step == (2 if extension is None else 4)
    if extension != "pending":
        for name, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(
                namespace["trainer"].model.state_dict()[name], tensor,
            )
    else:
        assert namespace["trainer"].scheduler.last_epoch == 2


@pytest.mark.parametrize("resume_code", [None, "print('resume example')\n"])
def test_download_cell_bundles_and_displays_resume_only_when_available(
    tmp_path, monkeypatch, resume_code,
):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb").read_text()
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"resume": {"code": resume_code}}))
    displayed = []
    display_module = ModuleType("IPython.display")
    display_module.Code = lambda code, language: (code, language)
    display_module.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython", ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(download=lambda path: None)
    monkeypatch.setitem(sys.modules, "google", ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    exec(
        "".join(notebook["cells"][7]["source"]),
        {"session_dir": tmp_path, "report_path": report_path},
    )

    with zipfile.ZipFile(tmp_path / "trainlm-scratch-artifact.zip") as bundle:
        assert "report.json" in bundle.namelist()
        assert ("resume.py" in bundle.namelist()) is (resume_code is not None)
        if resume_code:
            assert bundle.read("resume.py").decode() == resume_code
    assert displayed == ([(resume_code, "python")] if resume_code else [])
