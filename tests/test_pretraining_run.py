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


def test_run_pretraining_accepts_raw_hf_dataset_objects_and_records_identity(
    tmp_path,
):
    class TextDataset:
        column_names = ["text"]

        def __init__(self, fingerprint, label):
            self._fingerprint = fingerprint
            self.rows = [
                {"text": f"{label} corpus sample about stars and trees {index}. " * 5}
                for index in range(16)
            ]

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

    config = replace(
        _config(tmp_path / "provided-artifacts"),
        dataset_source="provided",
        dataset_id=None,
        dataset_revision=None,
    )
    result = run_pretraining(
        config,
        train_dataset=TextDataset("train-source-123", "train"),
        eval_dataset=TextDataset("eval-source-456", "validation"),
    )

    assert result.status == "passed"
    assert result.report["data"]["source_type"] == "huggingface_dataset_objects"
    assert result.report["data"]["provided_datasets"]["train"]["fingerprint"] == (
        "train-source-123"
    )
    assert result.report["resume"]["required_dataset_fingerprints"]["validation"][
        "fingerprint"
    ] == "eval-source-456"
    assert "train_dataset=train_dataset" in result.report["resume"]["code"]
    compile(result.report["resume"]["code"], "resume.py", "exec")
    restored = restore_pretraining_report(result.report_path)
    assert restored.config.dataset_source == "provided"


def test_run_pretraining_accepts_tokenized_hf_dataset_and_records_label_policy(
    tmp_path,
):
    from trainlm.pretraining import (
        _token_id_documents,
        _train_byte_level_tokenizer,
    )

    corpus_rows = [
        {"text": f"A compact language modeling document number {index}. " * 4}
        for index in range(16)
    ]
    tokenizer, _ = _train_byte_level_tokenizer(
        corpus_rows,
        max_examples=16,
        vocab_size=260,
    )
    tokenizer_dir = tmp_path / "selected-tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    class TokenDataset:
        column_names = ["input_ids", "attention_mask"]

        def __init__(self, label):
            self._fingerprint = f"{label}-token-fingerprint"
            self.rows = []
            self.expected_token_count = 0
            for index in range(16):
                ids = tokenizer.encode(
                    f"{label} language modeling document {index}. " * 4,
                    add_special_tokens=False,
                )
                if index % 2:
                    ids.append(tokenizer.eos_token_id)
                self.expected_token_count += (
                    len(ids) if ids[-1] == tokenizer.eos_token_id else len(ids) + 1
                )
                padding = max(0, 96 - len(ids))
                self.rows.append({
                    "input_ids": ids + [tokenizer.pad_token_id] * padding,
                    "attention_mask": [1] * len(ids) + [0] * padding,
                })

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

    train_data = TokenDataset("train")
    eval_data = TokenDataset("validation")
    config = replace(
        _config(tmp_path / "token-artifacts"),
        dataset_source="provided",
        data_format="tokens",
        dataset_id=None,
        dataset_revision=None,
        tokenizer_name_or_path=str(tokenizer_dir),
        max_train_examples=16,
        max_eval_examples=16,
    )
    result = run_pretraining(
        config, train_dataset=train_data, eval_dataset=eval_data
    )

    assert result.status == "passed"
    assert result.report["data"]["data_format"] == "tokens"
    assert result.report["data"]["causal_target_policy"] == (
        "TrainLM copies input_ids to labels and applies one next-token shift."
    )
    assert result.report["data"]["tokenizer_dataset_match"] == (
        "not_verified_by_trainlm"
    )
    assert result.report["data"]["tokenizer_provenance"]["mapping_sha256"]
    assert result.report["data"]["provided_datasets"]["train"]["fingerprint"] == (
        "train-token-fingerprint"
    )
    cache = json.loads(
        (config.output_dir / "preparation_cache.json").read_text(encoding="utf-8")
    )
    assert cache["inputs"]["algorithm"] == "prepared_hf_token_ids_v1"
    assert cache["inputs"]["data_format"] == "tokens"
    train_manifest = json.loads(
        (config.output_dir / "data/train/train.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert train_manifest["token_count"] == train_data.expected_token_count
    assert "dataset_tokenizer_match" not in result.report["tokenizer"]
    compile(result.report["resume"]["code"], "resume.py", "exec")

    with pytest.raises(ValueError, match="TrainLM generates next-token targets"):
        list(_token_id_documents(
            [{"input_ids": [1, 2], "labels": [1, 2]}],
            tokenizer,
            max_examples=1,
        ))


def test_run_pretraining_reports_and_restores_local_text_files(tmp_path, monkeypatch):
    rows = [
        {"text": f"Local corpus document about lakes and mountains {index}. " * 4}
        for index in range(16)
    ]
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    train_file = tmp_path / "train.txt"
    validation_file = tmp_path / "validation.txt"
    train_file.write_text("training data\n", encoding="utf-8")
    validation_file.write_text("validation data\n", encoding="utf-8")
    config = replace(
        _config(tmp_path / "local-artifacts"),
        dataset_id=None,
        dataset_revision=None,
        train_file=train_file,
        validation_file=validation_file,
    )

    result = run_pretraining(config)

    assert result.status == "passed"
    assert result.report["data"]["source_type"] == "local_text_files"
    assert result.report["data"]["local_files"]["train"]["path"] == str(
        train_file.resolve()
    )
    assert len(result.report["data"]["local_files"]["train"]["sha256"]) == 64
    restored = restore_pretraining_report(result.report_path)
    assert restored.config.train_file == train_file.resolve()
    assert restored.config.validation_file == validation_file.resolve()
    assert "train_file" in result.report["resume"]["code"]


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
