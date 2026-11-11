"""End-to-end contracts for the simple raw-text pretraining API."""

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import pytest

from trainlm import (
    TrainLMPretrainingConfig,
    TrainLMPretrainingError,
    TrainLMTrainer,
    bundle_pretraining_artifacts,
    restore_pretraining_report,
    run_pretraining,
)


pytestmark = pytest.mark.usefixtures("compatible_pretraining_profile")


def _config(output_dir: Path) -> TrainLMPretrainingConfig:
    return TrainLMPretrainingConfig(
        output_dir=output_dir,
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
        save_total_limit=2,
    )


def test_run_pretraining_completes_report_export_bundle_and_restore(
    tmp_path, monkeypatch
):
    rows = [
        {"text": f"Tiny pretraining example with trees and stars {index}. " * 4}
        for index in range(16)
    ]
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    config = _config(tmp_path / "artifacts")
    report_path = tmp_path / "report.json"
    result = run_pretraining(
        config,
        source_revision="b" * 40,
        report_path=report_path,
    )

    assert result.status == "passed"
    assert result.report_path == report_path
    assert result.export_path == tmp_path / "artifacts" / "baseline-export"
    assert (result.export_path / "config.json").is_file()
    assert (result.export_path / "model.safetensors").is_file()
    assert result.report["training"]["supervised_tokens"] >= config.max_tokens
    assert result.report["source_revision"] == "b" * 40
    assert result.report["resume"]["status"] == "available"
    assert result.metrics["eval_loss"] == result.report["training"]["eval_loss"]
    assert result.checkpoint_path is not None and result.checkpoint_path.is_dir()
    assert json.loads(report_path.read_text())["schema_version"] == 6

    restored = restore_pretraining_report(report_path, run_dir=tmp_path)
    assert restored.config.max_tokens == config.max_tokens
    assert restored.checkpoint_path.is_dir()
    assert restored.extend_training_budget is False

    bundle = bundle_pretraining_artifacts(report_path)
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "report.json" in names
    assert "resume.py" in names
    assert "model/config.json" in names
    assert any(name.startswith("model/") and name.endswith(".safetensors") for name in names)


def test_run_pretraining_can_show_sparse_training_progress(tmp_path, monkeypatch, capsys):
    rows = [
        {"text": f"Progress sample number {index}. " * 5}
        for index in range(16)
    ]
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    config = replace(_config(tmp_path / "progress"), logging_steps=1)
    result = run_pretraining(config, show_progress=True)

    output = capsys.readouterr().out
    assert result.status == "passed"
    assert "train step 1/2" in output
    assert "tokens 7/8" in output
    assert "loss " in output
    assert "lr " in output


def test_run_pretraining_requires_boolean_progress_setting(tmp_path):
    with pytest.raises(TypeError, match="show_progress must be a boolean"):
        run_pretraining(_config(tmp_path / "progress"), show_progress=1)


def test_run_pretraining_writes_a_report_when_construction_fails(tmp_path, monkeypatch):
    def fail_from_scratch(cls, config, **kwargs):
        raise OSError("synthetic setup failure")

    monkeypatch.setattr(TrainLMTrainer, "from_scratch", classmethod(fail_from_scratch))
    report_path = tmp_path / "artifacts" / "report.json"

    with pytest.raises(TrainLMPretrainingError, match="report.json") as raised:
        run_pretraining(_config(tmp_path / "artifacts"))

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.report_path == report_path
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "OSError"


def test_run_pretraining_report_preserves_package_preflight_failure(
    tmp_path, monkeypatch,
):
    profile = {
        "schema_version": 1,
        "scope": "installed_metadata_only",
        "status": "incompatible",
        "accelerator": "cpu",
        "requirements": [{
            "package": "datasets", "required": "<5,>=3",
            "installed": None, "status": "missing",
        }],
        "limitations": ["metadata_only", "does_not_certify_accelerator_execution"],
    }
    monkeypatch.setattr(
        "trainlm.runtime.pretraining_profile.inspect_pretraining_profile",
        lambda **kwargs: profile,
    )
    report_path = tmp_path / "artifacts" / "report.json"

    with pytest.raises(TrainLMPretrainingError) as raised:
        run_pretraining(_config(tmp_path / "artifacts"), report_path=report_path)

    assert raised.value.__cause__.profile is profile
    assert raised.value.report["preflight"]["package_profile"] is profile
    assert (
        json.loads(report_path.read_text())["preflight"]["package_profile"] == profile
    )
