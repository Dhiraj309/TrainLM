from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from trainlm import (
    TrainLMFineTuningConfig,
    TrainLMFineTuningResult,
    TrainLMTrainer,
    TrainLMTrainingArguments,
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
            assert remove_columns == self.column_names
            batch = {
                name: [row[name] for row in self.rows]
                for name in self.column_names
            }
            mapped = function(batch)
            rows = [
                {name: values[index] for name, values in mapped.items()}
                for index in range(len(self.rows))
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

        def train(self, *, resume_from_checkpoint):
            calls["trainer"]["resume_from_checkpoint"] = resume_from_checkpoint
            return SimpleNamespace(step=4, loss=1.25)

        def evaluate(self):
            return {"eval_loss": 1.5, "eval_perplexity": 4.48}

        def save_model(self, path):
            calls["trainer"]["export_path"] = Path(path)
            return Path(path)

    fake_trainer = FakeTrainer()

    def from_pretrained(cls, model_name_or_path, **kwargs):
        calls["trainer"]["model_load"] = (model_name_or_path, kwargs)
        return fake_trainer

    monkeypatch.setattr(
        TrainLMTrainer, "from_pretrained", classmethod(from_pretrained)
    )

    config = _config(tmp_path / "run", model_revision="b" * 40)
    result = run_finetuning(config, resume_from_checkpoint="latest")

    assert isinstance(result, TrainLMFineTuningResult)
    assert calls["datasets"] == [
        ("org/text-data", "a" * 40, "train"),
        ("org/text-data", "a" * 40, "validation"),
    ]
    assert calls["trainer"]["tokenizer_load"] == (
        "org/model",
        {"revision": "b" * 40},
    )
    model_name, trainer_kwargs = calls["trainer"]["model_load"]
    assert model_name == "org/model"
    assert trainer_kwargs["revision"] == "b" * 40
    assert trainer_kwargs["args"] == config.args
    assert trainer_kwargs["processing_class"] is tokenizer
    assert fake_trainer.model.config.pad_token_id == tokenizer.pad_token_id
    assert calls["trainer"]["resume_from_checkpoint"] == "latest"
    assert result.model_path == tmp_path / "run" / "model"
    assert result.step == 4
    assert result.train_loss == 1.25
    assert result.evaluation["eval_loss"] == 1.5
    assert result.summary()["dataset_revision"] == "a" * 40
    assert result.model_revision == "b" * 40
    assert result.resolved_model_revision == "c" * 40
    assert result.summary()["resolved_model_revision"] == "c" * 40

    collator = trainer_kwargs["data_collator"]
    batch = collator(
        [
            {"input_ids": [7, 2], "attention_mask": [1, 1]},
            {"input_ids": [8], "attention_mask": [1]},
        ]
    )
    assert batch["labels"].tolist() == [[7, 2], [8, -100]]


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
