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


def test_finetuning_config_requires_pinned_dataset_and_distinct_splits(tmp_path):
    with pytest.raises(ValueError, match="immutable 40-character"):
        _config(tmp_path, dataset_revision="main")

    with pytest.raises(ValueError, match="must be distinct"):
        _config(tmp_path, validation_split="train")

    with pytest.raises(TypeError, match="TrainLMTrainingArguments"):
        _config(tmp_path, args={"max_steps": 1})


def test_restore_finetuning_config_rejects_unrelated_summary(tmp_path):
    summary_path = tmp_path / "report.json"
    summary_path.write_text(
        json.dumps({"schema_version": 1, "scope": "pretraining"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a supported"):
        restore_finetuning_config(summary_path, max_steps=10)


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
    assert result.summary()["resolved_model_revision"] == "c" * 40
    assert result.summary()["resolved_tokenizer_revision"] == "c" * 40
    assert result.summary()["model_tokenizer_revision_match"] is True
    assert result.summary_path == tmp_path / "run" / "finetuning-summary.json"
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
    assert report["last_known_state"]["step"] == 4
    assert report["last_known_state"]["latest_checkpoint"] == str(checkpoint_path)
    restored = restore_finetuning_config(result.summary_path, max_steps=8)
    assert restored.model_name_or_path == config.model_name_or_path
    assert restored.model_revision == config.model_revision
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
        dataset_id="org/custom-instructions",
        dataset_revision="b" * 40,
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
    assert provided_report["dataset_id"] == "org/custom-instructions"
    assert provided_report["dataset_revision"] == "b" * 40


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
