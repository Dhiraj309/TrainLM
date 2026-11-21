"""End-to-end contracts for the simple raw-text pretraining API."""

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import pytest

from trainlm import (
    PackedBinDataset,
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


def test_run_pretraining_accepts_named_multicolumn_text_formatter(tmp_path):
    class PromptDataset:
        column_names = ["prompt", "completion"]

        def __init__(self, fingerprint, label):
            self._fingerprint = fingerprint
            self.rows = [
                {
                    "prompt": f"Explain language model training {index}",
                    "completion": (
                        f"{label} response with enough text to tokenize {index}. " * 5
                    ),
                }
                for index in range(16)
            ]

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

    config = replace(
        _config(tmp_path / "formatted-artifacts"),
        dataset_source="provided",
        dataset_id=None,
        dataset_revision=None,
        dataset_config_name=None,
        text_formatter_name="prompt-completion-v1",
    )
    result = run_pretraining(
        config,
        train_dataset=PromptDataset("train-prompt-v1", "train"),
        eval_dataset=PromptDataset("eval-prompt-v1", "validation"),
        formatting_func=lambda row: (
            f"Question: {row['prompt']}\nAnswer: {row['completion']}"
        ),
    )

    assert result.status == "passed"
    assert result.report["data"]["text_formatter_name"] == "prompt-completion-v1"
    assert result.report["data"]["provided_datasets"]["train"]["fingerprint"] == (
        "train-prompt-v1"
    )
    resume_code = result.report["resume"]["code"]
    assert "formatting_func=formatting_func" in resume_code
    assert "prompt-completion-v1" in resume_code
    compile(resume_code, "resume.py", "exec")


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
        column_names = ["input_ids", "attention_mask", "labels", "loss_mask"]

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
                labels = list(ids)
                labels[1] = -100
                loss_mask = [1] * len(ids)
                loss_mask[3] = 0
                self.expected_token_count += (
                    len(ids) if ids[-1] == tokenizer.eos_token_id else len(ids) + 1
                )
                padding = max(0, 96 - len(ids))
                self.rows.append({
                    "input_ids": ids + [tokenizer.pad_token_id] * padding,
                    "attention_mask": [1] * len(ids) + [0] * padding,
                    "labels": labels + [-100] * padding,
                    "loss_mask": loss_mask + [0] * padding,
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
        "TrainLM preserves aligned labels/loss masks, generates labels from "
        "input_ids when absent, and applies one next-token shift."
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
    assert cache["inputs"]["algorithm"] == "prepared_hf_token_ids_v2_supervision"
    assert cache["inputs"]["data_format"] == "tokens"
    assert cache["inputs"]["token_supervision_policy"] == (
        "preserve_aligned_labels_and_loss_mask_v1"
    )
    train_manifest = json.loads(
        (config.output_dir / "data/train/train.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert train_manifest["token_count"] == train_data.expected_token_count
    assert train_manifest["target_ids"]["dtype"] == "int64"
    assert train_manifest["loss_mask"]["dtype"] == "uint8"
    assert "dataset_tokenizer_match" not in result.report["tokenizer"]
    compile(result.report["resume"]["code"], "resume.py", "exec")

    prepared = list(
        _token_id_documents(
            [
                {
                    "input_ids": [1, 2],
                    "labels": [1, -100],
                    "loss_mask": [True, False],
                }
            ],
            tokenizer,
            max_examples=1,
        )
    )[0]
    assert prepared.target_ids == [1, -100, -100]
    assert prepared.loss_mask == [1, 0, 0]

    validation_packed = PackedBinDataset.from_directory(
        config.output_dir / "data" / "validation",
        sequence_length=config.sequence_length,
        split="validation",
    )
    try:
        first_batch = next(iter(validation_packed))
    finally:
        validation_packed.close()
    first_row = eval_data.rows[0]
    active_length = sum(first_row["attention_mask"])
    expected_ids = first_row["input_ids"][:active_length]
    expected_labels = first_row["labels"][:active_length]
    expected_mask = first_row["loss_mask"][:active_length]
    if expected_ids[-1] != tokenizer.eos_token_id:
        expected_ids = expected_ids + [tokenizer.eos_token_id]
        expected_labels = expected_labels + [
            tokenizer.eos_token_id if expected_labels[-1] != -100 else -100
        ]
        expected_mask = expected_mask + [expected_mask[-1]]
    assert first_batch["input_ids"].tolist() == expected_ids[:8]
    assert first_batch["labels"].tolist() == expected_labels[:8]
    assert first_batch["loss_mask"].tolist() == [
        bool(value) for value in expected_mask[:8]
    ]


@pytest.mark.parametrize(
    ("suffix", "expected_format", "field"),
    [(".txt", "text", "text"), (".csv", "csv", "body")],
)
def test_run_pretraining_reports_and_restores_local_files(
    tmp_path, monkeypatch, suffix, expected_format, field
):
    rows = [
        {field: f"Local corpus document about lakes and mountains {index}. " * 4}
        for index in range(16)
    ]
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    train_file = tmp_path / f"train{suffix}"
    validation_file = tmp_path / f"validation{suffix}"
    train_file.write_text("local training data\n", encoding="utf-8")
    validation_file.write_text("local validation data\n", encoding="utf-8")
    config = replace(
        _config(tmp_path / "local-artifacts"),
        dataset_id=None,
        dataset_revision=None,
        train_file=train_file,
        validation_file=validation_file,
        text_field=field,
    )

    result = run_pretraining(config)

    assert result.status == "passed"
    assert result.report["data"]["source_type"] == (
        "local_text_files" if expected_format == "text" else "local_structured_files"
    )
    assert result.report["data"]["local_data_format"] == expected_format
    if expected_format == "text":
        assert "format" not in result.report["data"]["local_files"]
    else:
        assert result.report["data"]["local_files"]["format"] == expected_format
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


def test_streamed_token_cpt_repeats_finite_corpus_and_reloads_export(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
    from trainlm.pretraining import _train_byte_level_tokenizer

    base = tmp_path / "base"
    tokenizer, _ = _train_byte_level_tokenizer(
        [{"text": "A short local story."}], max_examples=1, vocab_size=260,
    )
    tokenizer.save_pretrained(base)
    GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=8, n_embd=8, n_layer=1, n_head=2,
    )).save_pretrained(base)
    calls = []

    def stream(repo, *, revision, split, streaming):
        calls.append((repo, revision, split, streaming))
        token = 4 if split == "train" else 5
        return iter([{"input_ids": [token] * 7 + [tokenizer.eos_token_id]}])

    datasets = ModuleType("datasets")
    datasets.load_dataset = stream
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    config = replace(
        _config(tmp_path / "run"), model_name_or_path=base,
        model_embedding_size=768, model_layers=6, model_attention_heads=12,
        model_initialization="pretrained", data_format="tokens", max_tokens=21,
        repeat_train_data=True,
    )
    result = run_pretraining(config)
    assert calls == [(config.dataset_id, "a" * 40, split, True)
                     for split in ("train", "validation")]
    assert result.report["training"]["supervised_tokens"] == 21
    assert result.report["data"]["tokenizer_provenance"]["mapping_sha256"]
    assert result.report["data"]["repeat_train_data"] is True
    model = AutoModelForCausalLM.from_pretrained(result.export_path, local_files_only=True)
    assert model.generate(torch.tensor([[4]]), max_new_tokens=1).shape == (1, 2)


def test_moved_run_directory_resumes_to_a_larger_token_budget(tmp_path, monkeypatch):
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: iter([{"text": "A short story. " * 40}] * 12)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    original = tmp_path / "host-a"
    fresh = run_pretraining(_config(original))
    target = tmp_path / "host-b"
    original.rename(target)
    saved = restore_pretraining_report(target / "report.json", restored_output_dir=target)
    assert saved.checkpoint_path.parent == target / "checkpoints"
    resumed = run_pretraining(
        replace(saved.config, max_tokens=22),
        resume_from_checkpoint=saved.checkpoint_path, extend_training_budget=True,
        report_path=target / "resume-report.json",
    )
    assert resumed.report["training"]["step"] > fresh.report["training"]["step"]
    assert resumed.report["training"]["supervised_tokens"] >= 22
