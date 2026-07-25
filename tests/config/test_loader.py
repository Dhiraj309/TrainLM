import pytest

from trainlm.config import load_config


def test_load_minimal_config(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
trainer:
  max_steps: 100
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.model.hidden_size == 768
    assert config.optimizer.name == "adamw"

    assert config.trainer.max_steps == 100
    assert config.trainer.max_tokens is None
    assert config.trainer.gradient_accumulation_steps == 1


def test_override_config(tmp_path):
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"

    base.write_text(
        """
trainer:
  max_steps: 100

model:
  hidden_size: 768

optimizer:
  learning_rate: 0.001
""",
        encoding="utf-8",
    )

    override.write_text(
        """
optimizer:
  learning_rate: 0.0003
""",
        encoding="utf-8",
    )

    config = load_config(
        base,
        override,
    )

    assert config.model.hidden_size == 768
    assert config.optimizer.learning_rate == 0.0003
    assert config.trainer.max_steps == 100
