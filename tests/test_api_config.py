from pathlib import Path

import pytest

from trainlm import PUBLIC_API_VERSION, TrainLMTrainer, TrainLMTrainingArguments
from trainlm.config import ModelSourceConfig


class CapturingTrainer(TrainLMTrainer):
    def __init__(self, **kwargs):
        self.captured = kwargs


def test_from_pretrained_builds_explicit_revision_pinned_source():
    trainer = CapturingTrainer.from_pretrained(
        "org/model", revision="abc123", train_dataset=object()
    )

    assert trainer.captured["model"] == ModelSourceConfig(
        provider="huggingface",
        initialization="pretrained",
        name_or_path="org/model",
        revision="abc123",
    )


def test_from_config_supports_yaml_pretraining_from_hf_config(tmp_path: Path):
    path = tmp_path / "train.yaml"
    path.write_text(
        """api_version: '1'
model:
  initialization: config
  model_type: gpt2
  config_overrides:
    n_layer: 4
training_args:
  output_dir: runs/gpt2
  max_steps: 20
  accelerator: cpu
""",
        encoding="utf-8",
    )

    trainer = CapturingTrainer.from_config(path, train_dataset=object())

    assert PUBLIC_API_VERSION == "1"
    assert trainer.captured["model"].initialization == "config"
    assert trainer.captured["model"].model_type == "gpt2"
    assert trainer.captured["args"] == TrainLMTrainingArguments(
        output_dir="runs/gpt2", max_steps=20, accelerator="cpu"
    )


def test_deprecated_args_key_warns_and_unknown_keys_fail():
    with pytest.warns(DeprecationWarning, match="training_args"):
        trainer = CapturingTrainer.from_config(
            {"model": "org/model", "args": {"max_steps": 2}},
            train_dataset=object(),
        )
    assert trainer.captured["args"].max_steps == 2

    with pytest.raises(ValueError, match="Unknown trainer configuration keys"):
        CapturingTrainer.from_config(
            {"model": "org/model", "worker_command": ["python"]},
            train_dataset=object(),
        )


def test_public_config_rejects_unsupported_api_version():
    with pytest.raises(ValueError, match="Unsupported public API version"):
        CapturingTrainer.from_config(
            {"api_version": "2", "model": "org/model"},
            train_dataset=object(),
        )
