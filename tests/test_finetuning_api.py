import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from trainlm import (
    TrainLMFineTuningConfig,
    TrainLMFineTuningResult,
    TrainLMTrainer,
    TrainLMTrainingArguments,
    restore_finetuning_config,
    run_finetuning,
)


def _config(output_dir: Path, **overrides) -> TrainLMFineTuningConfig:
    values = {
        "model_name_or_path": "org/model",
        "dataset_id": "org/text-data",
        "dataset_revision": "a" * 40,
        "args": TrainLMTrainingArguments(
            output_dir=output_dir,
            max_steps=4,
            sequence_length=8,
            accelerator="cpu",
        ),
    }
    values.update(overrides)
    return TrainLMFineTuningConfig(**values)


def test_finetuning_config_validates_optional_dataset_metadata_and_splits(tmp_path):
    with pytest.raises(ValueError, match="immutable 40-character"):
        _config(tmp_path, dataset_revision="main")

    local_config = _config(
        tmp_path,
        dataset_id=None,
        dataset_revision=None,
    )
    assert local_config.dataset_id is None
    assert local_config.dataset_revision is None

    with pytest.raises(ValueError, match="must be distinct"):
        _config(tmp_path, validation_split="train")

    with pytest.raises(TypeError, match="TrainLMTrainingArguments"):
        _config(tmp_path, args={"max_steps": 1})

    with pytest.raises(ValueError, match="tokenizer_name_or_path must be non-empty"):
        _config(tmp_path, tokenizer_name_or_path=" ")
    with pytest.raises(ValueError, match="tokenizer_revision must be non-empty"):
        _config(tmp_path, tokenizer_revision=" ")


def test_run_finetuning_requires_pinned_metadata_for_hub_splits(tmp_path):
    config = _config(
        tmp_path,
        dataset_id=None,
        dataset_revision=None,
    )

    with pytest.raises(ValueError, match="required when .*loads splits"):
        run_finetuning(config)

    assert not (tmp_path / "finetuning-summary.json").exists()


def test_restore_finetuning_config_rejects_unrelated_summary(tmp_path):
    summary_path = tmp_path / "report.json"
    summary_path.write_text(
        json.dumps({"schema_version": 1, "scope": "pretraining"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a supported"):
        restore_finetuning_config(summary_path, max_steps=10)


def test_restore_finetuning_config_extends_epoch_budget(tmp_path):
    config = _config(
        tmp_path,
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=None,
            num_train_epochs=1,
            accelerator="cpu",
        ),
    )
    training_arguments = {
        name: getattr(config.args, name)
        for name in config.args.__dataclass_fields__
    }
    training_arguments["output_dir"] = str(tmp_path)
    summary_path = tmp_path / "finetuning-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "pretrained_causal_lm_finetuning",
                "model_name_or_path": config.model_name_or_path,
                "model_revision": None,
                "dataset_id": config.dataset_id,
                "dataset_revision": config.dataset_revision,
                "training_arguments": training_arguments,
                "data": {
                    "text_field": config.text_field,
                    "train_split": config.train_split,
                    "validation_split": config.validation_split,
                },
                "step": 1,
                "last_known_state": {"epoch": 1},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    torch.save({"schema_version": 1}, checkpoint / "trainer_state.pt")

    restored = restore_finetuning_config(
        summary_path,
        num_train_epochs=2,
    )

    assert restored.args.max_steps is None
    assert restored.args.num_train_epochs == 2
    assert restored.args.output_dir == tmp_path
    with pytest.raises(ValueError, match="greater than the saved epoch budget"):
        restore_finetuning_config(summary_path, num_train_epochs=1)


def test_run_finetuning_loads_trains_evaluates_and_exports(tmp_path, monkeypatch):
    calls = {"datasets": [], "trainer": {}}

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = list(rows[0])

        def __len__(self):
            return len(self.rows)

        def map(self, function, *, batched, remove_columns):
            assert batched is True
            assert set(remove_columns).issubset(self.column_names)
            batch = {
                name: [row[name] for row in self.rows]
                for name in self.column_names
            }
            mapped = function(batch)
            rows = [
                {
                    **{
                        name: row[name]
                        for name in self.column_names
                        if name not in remove_columns
                    },
                    **{name: values[index] for name, values in mapped.items()},
                }
                for index, row in enumerate(self.rows)
            ]
            return FakeDataset(rows)

    datasets = ModuleType("datasets")

    def load_dataset(dataset_id, *, revision, split):
        calls["datasets"].append((dataset_id, revision, split))
        return FakeDataset([{"text": f"{split} sample", "metadata": "remove"}])

    datasets.load_dataset = load_dataset

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token_id = None
        init_kwargs = {
            "tokenizer_file": (
                "/cache/models--org--model/snapshots/"
                + "c" * 40
                + "/tokenizer.json"
            )
        }

        @property
        def pad_token(self):
            return self.eos_token if self.pad_token_id == 2 else None

        @pad_token.setter
        def pad_token(self, value):
            assert value == self.eos_token
            self.pad_token_id = 2

        def __call__(self, texts, *, truncation, max_length):
            assert truncation is True
            assert max_length == 8
            calls["trainer"].setdefault("tokenized_texts", []).append(list(texts))
            return {
                "input_ids": [[index + 10, 2] for index, _ in enumerate(texts)],
                "attention_mask": [[1, 1] for _ in texts],
            }

        def get_vocab(self):
            return {"<eos>": 2, "sample": 10}

        def pad(self, features, *, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            length = max(len(item["input_ids"]) for item in features)
            input_ids = []
            attention_mask = []
            for item in features:
                pad_length = length - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_length)
                attention_mask.append(item["attention_mask"] + [0] * pad_length)
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }

    tokenizer = FakeTokenizer()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda model, **kwargs: (
            calls["trainer"].update(tokenizer_load=(model, kwargs)) or tokenizer
        )
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class FakeTrainer:
        def __init__(self):
            self.model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))
            self.loaded = SimpleNamespace(
                metadata=SimpleNamespace(resolved_revision="c" * 40)
            )
            self.engine = SimpleNamespace(state=SimpleNamespace(step=0, loss=None))
            calls["trainer"]["train_calls"] = 0

        def train(self, *, resume_from_checkpoint):
            calls["trainer"]["train_calls"] += 1
            calls["trainer"]["resume_from_checkpoint"] = resume_from_checkpoint
            if calls["trainer"].get("fail_train"):
                self.engine.state = SimpleNamespace(step=2, loss=0.75)
                raise RuntimeError("simulated failure; token=must-not-be-saved")
            return SimpleNamespace(step=4, loss=1.25)

        def evaluate(self):
            return {"eval_loss": 1.5, "eval_perplexity": 4.48}

        def save_model(self, path):
            calls["trainer"]["export_path"] = Path(path)
            return Path(path)

        @property
        def latest_checkpoint(self):
            return calls["trainer"].get("latest_checkpoint")

        @property
        def best_model_selection(self):
            return calls["trainer"].get("best_model_selection")

    fake_trainer = FakeTrainer()

    def from_pretrained(cls, model_name_or_path, **kwargs):
        calls["trainer"]["model_load"] = (model_name_or_path, kwargs)
        return fake_trainer

    monkeypatch.setattr(
        TrainLMTrainer, "from_pretrained", classmethod(from_pretrained)
    )

    config = _config(tmp_path / "run", model_revision="main")
    checkpoint_path = config.args.output_dir / "checkpoint-4"
    checkpoint_path.mkdir(parents=True)
    checkpoint_state = checkpoint_path / "trainer_state.pt"
    torch.save({"schema_version": 1}, checkpoint_state)
    calls["trainer"]["latest_checkpoint"] = str(checkpoint_path)
    calls["trainer"]["best_model_selection"] = {
        "metric_name": "eval_loss",
        "greater_is_better": False,
        "value": 1.25,
        "step": 3,
        "checkpoint": str(config.args.output_dir / "checkpoint-3"),
    }
    result = run_finetuning(config, resume_from_checkpoint="latest")

    assert isinstance(result, TrainLMFineTuningResult)
    assert calls["datasets"] == [
        ("org/text-data", "a" * 40, "train"),
        ("org/text-data", "a" * 40, "validation"),
    ]
    assert calls["trainer"]["tokenizer_load"] == (
        "org/model",
        {"revision": "main"},
    )
    model_name, trainer_kwargs = calls["trainer"]["model_load"]
    assert model_name == "org/model"
    assert trainer_kwargs["revision"] == "c" * 40
    assert trainer_kwargs["args"] == config.args
    assert trainer_kwargs["processing_class"] is tokenizer
    assert fake_trainer.model.config.pad_token_id == tokenizer.pad_token_id
    assert calls["trainer"]["resume_from_checkpoint"] == "latest"
    assert result.model_path == tmp_path / "run" / "model"
    assert result.step == 4
    assert result.train_loss == 1.25
    assert result.evaluation["eval_loss"] == 1.5
    assert result.summary()["dataset_revision"] == "a" * 40
    assert result.model_revision == "main"
    assert result.resolved_model_revision == "c" * 40
    assert result.resolved_tokenizer_revision == "c" * 40
    assert result.model_tokenizer_revision_match is True
    assert result.tokenizer_name_or_path == "org/model"
    assert result.tokenizer_revision == "main"
    assert result.tokenizer_compatibility["token_semantics"] == (
        "not_verified_by_trainlm"
    )
    assert result.summary()["resolved_model_revision"] == "c" * 40
    assert result.summary()["resolved_tokenizer_revision"] == "c" * 40
    assert result.summary()["model_tokenizer_revision_match"] is True
    assert result.summary_path == tmp_path / "run" / "finetuning-summary.json"
    assert result.summary()["best_model_selection"] == calls["trainer"][
        "best_model_selection"
    ]
    assert result.summary()["training_budget"] == {
        "num_train_epochs": None,
        "train_batches_per_epoch": None,
        "max_train_batches": None,
        "effective_max_steps": 4,
    }
    report = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["status"] == "completed"
    assert report["data"] == {
        "source": "huggingface_hub",
        "text_field": "text",
        "train_split": "train",
        "validation_split": "validation",
        "formatting_function": None,
    }
    assert report["training_arguments"]["output_dir"] == str(tmp_path / "run")
    assert report["training_arguments"]["max_steps"] == 4
    assert report["model_tokenizer_revision_match"] is True
    assert report["evaluation"]["eval_loss"] == 1.5
    assert report["best_model_selection"] == calls["trainer"][
        "best_model_selection"
    ]
    assert report["last_known_state"]["step"] == 4
    assert report["last_known_state"]["latest_checkpoint"] == str(checkpoint_path)
    restored = restore_finetuning_config(result.summary_path, max_steps=8)
    assert restored.model_name_or_path == config.model_name_or_path
    assert restored.model_revision == "c" * 40
    assert restored.tokenizer_name_or_path == "org/model"
    assert restored.tokenizer_revision == "c" * 40
    assert restored.dataset_id == config.dataset_id
    assert restored.dataset_revision == config.dataset_revision
    assert restored.text_field == config.text_field
    assert restored.train_split == config.train_split
    assert restored.validation_split == config.validation_split
    assert restored.args == replace(
        config.args,
        output_dir=result.summary_path.parent,
        max_steps=8,
    )
    legacy_report = json.loads(json.dumps(report))
    legacy_report["training_arguments"].pop("num_train_epochs")
    legacy_report["training_arguments"].pop("save_strategy")
    legacy_report["training_arguments"].pop("eval_strategy")
    legacy_report["training_arguments"].pop("load_best_model_at_end")
    legacy_report["training_arguments"].pop("metric_for_best_model")
    legacy_report["training_arguments"].pop("greater_is_better")
    legacy_report["training_arguments"].pop("early_stopping_patience")
    legacy_report["training_arguments"].pop("early_stopping_threshold")
    legacy_summary_path = result.summary_path.parent / "legacy-summary.json"
    legacy_summary_path.write_text(json.dumps(legacy_report), encoding="utf-8")
    restored_legacy = restore_finetuning_config(
        legacy_summary_path,
        max_steps=8,
    )
    assert restored_legacy.args.num_train_epochs is None
    with pytest.raises(ValueError, match="greater than the last recorded"):
        restore_finetuning_config(result.summary_path, max_steps=4)
    checkpoint_state.write_bytes(b"truncated checkpoint")
    with pytest.raises(ValueError, match="not a readable Torch archive"):
        restore_finetuning_config(result.summary_path, max_steps=8)
    checkpoint_state.unlink()
    with pytest.raises(ValueError, match="No published training checkpoint"):
        restore_finetuning_config(result.summary_path, max_steps=8)
    shard_name = "checkpoint-4-rank-00000.pt"
    (checkpoint_path / shard_name).write_bytes(b"committed shard")
    manifest = {
        "schema_version": 1,
        "status": "committed",
        "step": 4,
        "micro_step": 4,
        "world_size": 1,
        "shards": [shard_name],
    }
    manifest_path = checkpoint_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    restored_manifest_config = restore_finetuning_config(
        result.summary_path,
        max_steps=8,
    )
    assert restored_manifest_config.args.max_steps == 8
    manifest["status"] = "pending"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest or shard set"):
        restore_finetuning_config(result.summary_path, max_steps=8)

    collator = trainer_kwargs["data_collator"]
    batch = collator(
        [
            {"input_ids": [7, 2], "attention_mask": [1, 1]},
            {"input_ids": [8], "attention_mask": [1]},
        ]
    )
    assert batch["labels"].tolist() == [[7, 2], [8, -100]]

    fake_trainer.loaded.metadata.resolved_revision = "d" * 40
    with pytest.raises(RuntimeError, match="resolved to different Hub commits"):
        run_finetuning(config)
    assert calls["trainer"]["train_calls"] == 1
    mismatch_report = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert mismatch_report["status"] == "failed"
    assert mismatch_report["phase"] == "model_loading"
    assert mismatch_report["error_type"] == "RuntimeError"
    assert mismatch_report["resolved_model_revision"] == "d" * 40
    assert mismatch_report["resolved_tokenizer_revision"] == "c" * 40

    fake_trainer.loaded.metadata.resolved_revision = "c" * 40
    calls["trainer"]["fail_train"] = True
    calls["trainer"]["latest_checkpoint"] = str(
        tmp_path / "run" / "checkpoints" / "checkpoint-2"
    )
    with pytest.raises(RuntimeError, match="simulated failure"):
        run_finetuning(config)
    failed_report_text = result.summary_path.read_text(encoding="utf-8")
    failed_report = json.loads(failed_report_text)
    assert failed_report["status"] == "failed"
    assert failed_report["phase"] == "training"
    assert failed_report["error_type"] == "RuntimeError"
    assert failed_report["last_known_state"] == {
        "step": 2,
        "epoch": None,
        "train_loss": 0.75,
        "evaluation": {},
        "latest_checkpoint": str(
            tmp_path / "run" / "checkpoints" / "checkpoint-2"
        ),
    }
    assert "must-not-be-saved" not in failed_report_text
    assert calls["trainer"]["train_calls"] == 2

    provided_config = _config(
        tmp_path / "provided",
        dataset_id=None,
        dataset_revision=None,
    )
    calls["trainer"]["fail_train"] = False
    calls["trainer"]["latest_checkpoint"] = None
    provided_calls_before = list(calls["datasets"])

    def format_example(example):
        return f"Question: {example['prompt']}\nAnswer: {example['completion']}"

    def custom_collator(features):
        return features

    provided_result = run_finetuning(
        provided_config,
        train_dataset=FakeDataset(
            [{"prompt": "Write a greeting", "completion": "Hello."}]
        ),
        eval_dataset=FakeDataset(
            [{"prompt": "Write a farewell", "completion": "Goodbye."}]
        ),
        tokenizer=tokenizer,
        formatting_func=format_example,
        data_collator=custom_collator,
    )
    assert calls["datasets"] == provided_calls_before
    assert calls["trainer"]["tokenized_texts"][-2:] == [
        ["Question: Write a greeting\nAnswer: Hello."],
        ["Question: Write a farewell\nAnswer: Goodbye."],
    ]
    _, provided_trainer_kwargs = calls["trainer"]["model_load"]
    assert provided_trainer_kwargs["data_collator"] is custom_collator
    assert provided_trainer_kwargs["train_dataset"].column_names == [
        "prompt",
        "completion",
        "input_ids",
        "attention_mask",
    ]
    provided_report = json.loads(
        provided_result.summary_path.read_text(encoding="utf-8")
    )
    assert provided_report["data"]["source"] == "provided_dataset"
    assert provided_report["data"]["formatting_function"].endswith(
        ".format_example"
    )
    assert provided_report["dataset_id"] is None
    assert provided_report["dataset_revision"] is None
    assert provided_result.model_tokenizer_revision_match is None
    assert provided_report["tokenizer_name_or_path"] is None
    assert provided_report["tokenizer_revision"] is None
    assert provided_result.tokenizer_object_required is True
    assert provided_report["tokenizer_object_required"] is True
    provided_checkpoint = provided_result.summary_path.parent / "checkpoint-4"
    provided_checkpoint.mkdir()
    torch.save({"schema_version": 1}, provided_checkpoint / "trainer_state.pt")
    restored_provided_config = restore_finetuning_config(
        provided_result.summary_path,
        max_steps=8,
    )
    assert restored_provided_config.dataset_id is None
    assert restored_provided_config.dataset_revision is None
    with pytest.raises(ValueError, match="Pass that tokenizer object again"):
        run_finetuning(
            restored_provided_config,
            train_dataset=FakeDataset([{"text": "train"}]),
            eval_dataset=FakeDataset([{"text": "eval"}]),
        )

    different_tokenizer = FakeTokenizer()
    different_tokenizer.init_kwargs = {
        "tokenizer_file": (
            "/cache/models--org--other/snapshots/"
            + "d" * 40
            + "/tokenizer.json"
        )
    }
    with pytest.raises(ValueError, match="does not match the immutable revision"):
        run_finetuning(
            restored_provided_config,
            train_dataset=FakeDataset([{"text": "train"}]),
            eval_dataset=FakeDataset([{"text": "eval"}]),
            tokenizer=different_tokenizer,
        )

    # Model and tokenizer repositories resolve different commit SHAs. The
    # tokenizer revision must never be used to select the model snapshot or
    # compared as if both repositories shared a commit history.
    fake_trainer.loaded.metadata.resolved_revision = "d" * 40
    independent_config = _config(
        tmp_path / "independent-sources",
        model_revision="model-ref",
        tokenizer_name_or_path="org/compatible-tokenizer",
        tokenizer_revision="tokenizer-ref",
    )
    independent_result = run_finetuning(independent_config)
    independent_model_name, independent_model_kwargs = calls["trainer"][
        "model_load"
    ]
    assert independent_model_name == "org/model"
    assert independent_model_kwargs["revision"] == "model-ref"
    assert calls["trainer"]["tokenizer_load"] == (
        "org/compatible-tokenizer",
        {"revision": "tokenizer-ref"},
    )
    assert independent_result.resolved_model_revision == "d" * 40
    assert independent_result.resolved_tokenizer_revision == "c" * 40
    assert independent_result.model_tokenizer_revision_match is None
    assert independent_result.tokenizer_name_or_path == "org/compatible-tokenizer"
    assert independent_result.tokenizer_revision == "tokenizer-ref"
    independent_report = json.loads(
        independent_result.summary_path.read_text(encoding="utf-8")
    )
    assert independent_report["model_revision"] == "model-ref"
    assert independent_report["tokenizer_revision"] == "tokenizer-ref"
    assert independent_result.tokenizer_compatibility["source_relation"] == (
        "separate_source"
    )

    # Structural incompatibilities are recorded and rejected before training;
    # TrainLM does not silently resize pretrained embeddings or change IDs.
    fake_trainer.model.get_input_embeddings = lambda: SimpleNamespace(
        num_embeddings=8
    )
    bounds_config = _config(
        tmp_path / "tokenizer-out-of-range",
        tokenizer_name_or_path="org/compatible-tokenizer",
    )
    train_calls_before = calls["trainer"]["train_calls"]
    with pytest.raises(ValueError, match="will not resize embeddings"):
        run_finetuning(bounds_config)
    assert calls["trainer"]["train_calls"] == train_calls_before
    bounds_report = json.loads(
        (bounds_config.args.output_dir / "finetuning-summary.json").read_text()
    )
    assert bounds_report["tokenizer_compatibility"]["token_id_bounds"] == (
        "incompatible"
    )

    fake_trainer.model.get_input_embeddings = lambda: SimpleNamespace(
        num_embeddings=11
    )
    fake_trainer.model.config.max_position_embeddings = 4
    context_config = _config(
        tmp_path / "sequence-too-long",
        tokenizer_name_or_path="org/compatible-tokenizer",
    )
    with pytest.raises(ValueError, match="exceeds the model's configured context"):
        run_finetuning(context_config)
    context_report = json.loads(
        (context_config.args.output_dir / "finetuning-summary.json").read_text()
    )
    assert context_report["tokenizer_compatibility"]["context_length"] == (
        "exceeds_model_limit"
    )


def test_run_finetuning_rejects_legacy_xla_export_path(tmp_path):
    config = _config(
        tmp_path,
        args=TrainLMTrainingArguments(
            output_dir=tmp_path,
            max_steps=4,
            sequence_length=8,
            accelerator="tpu",
        ),
    )

    with pytest.raises(ValueError, match="legacy PyTorch/XLA coordinator"):
        run_finetuning(config)
