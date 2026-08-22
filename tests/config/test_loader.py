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

    assert config.model.provider == "external"
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
  provider: huggingface
  model_type: llama
  config_overrides:
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

    assert config.model.provider == "huggingface"
    assert config.model.model_type == "llama"
    assert config.model.config_overrides["hidden_size"] == 768
    assert config.optimizer.learning_rate == 0.0003
    assert config.trainer.max_steps == 100


def test_trainlm_reference_model_must_be_selected_explicitly(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
model:
  provider: trainlm
  config_overrides:
    hidden_size: 768
trainer:
  max_steps: 1
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.model.provider == "trainlm"
    assert config.model.config_overrides == {"hidden_size": 768}


def test_implicit_legacy_model_architecture_is_rejected(tmp_path):
    config_file = tmp_path / "legacy.yaml"
    config_file.write_text(
        """
model:
  hidden_size: 768
trainer:
  max_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.config_overrides"):
        load_config(config_file)


def test_legacy_runtime_compile_has_unambiguous_migration(tmp_path):
    config_file = tmp_path / "legacy-runtime.yaml"
    config_file.write_text(
        """
runtime:
  compile: true
trainer:
  max_steps: 1
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.optimizations.compile is True


def test_unknown_root_section_is_rejected(tmp_path):
    config_file = tmp_path / "unknown.yaml"
    config_file.write_text(
        """
trainer:
  max_steps: 1
typo_section: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="typo_section"):
        load_config(config_file)


def test_unknown_model_provider_is_rejected(tmp_path):
    config_file = tmp_path / "unknown-provider.yaml"
    config_file.write_text(
        """
model:
  provider: accidental_typo
trainer:
  max_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported model provider"):
        load_config(config_file)
