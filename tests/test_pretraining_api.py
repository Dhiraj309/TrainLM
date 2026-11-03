"""Public raw-text pretraining API contracts and a tiny CPU lifecycle smoke."""

from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from trainlm import (
    TrainLMPretrainingConfig,
    TrainLMTrainer,
)


def _config(output_dir: Path, **overrides) -> TrainLMPretrainingConfig:
    values = {
        "output_dir": output_dir,
        "dataset_id": "org/tiny-stories",
        "dataset_revision": "a" * 40,
        "max_tokens": 8,
        "sequence_length": 8,
        "max_train_examples": 12,
        "max_eval_examples": 8,
        "tokenizer_vocab_size": 260,
        "model_embedding_size": 16,
        "model_layers": 1,
        "model_attention_heads": 2,
        "accelerator": "cpu",
    }
    values.update(overrides)
    return TrainLMPretrainingConfig(**values)


def test_pretraining_config_requires_immutable_revision_and_disjoint_splits(tmp_path):
    with pytest.raises(ValueError, match="40-character commit SHA"):
        _config(tmp_path, dataset_revision="main")
    with pytest.raises(ValueError, match="must be distinct"):
        _config(tmp_path, validation_split="train")
    with pytest.raises(ValueError, match="divisible"):
        _config(tmp_path, model_embedding_size=15)
    with pytest.raises(ValueError, match="packed uint16"):
        _config(tmp_path, tokenizer_vocab_size=65_537)
    with pytest.raises(ValueError, match="requires accelerator='native_tpu'"):
        _config(tmp_path, torch_compile=True)


def test_from_scratch_builds_and_trains_a_small_cpu_model(tmp_path, monkeypatch):
    rows = [
        {"text": f"A tiny story about stars, trees, and friendly dogs number {i}. " * 4}
        for i in range(16)
    ]
    calls = []

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    def load_dataset(dataset_id, *, revision, split, streaming):
        calls.append((dataset_id, revision, split, streaming))
        return StreamingRows()

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    config = _config(tmp_path / "run")
    trainer = TrainLMTrainer.from_scratch(config)
    assert trainer.args.max_steps == 2
    assert trainer.args.max_tokens == 8
    assert trainer.model.config.model_type == "gpt2"
    assert calls == [
        ("org/tiny-stories", "a" * 40, "train", True),
        ("org/tiny-stories", "a" * 40, "train", True),
        ("org/tiny-stories", "a" * 40, "validation", True),
    ]

    state = trainer.train()
    metrics = trainer.evaluate()
    export = trainer.save_model(tmp_path / "run" / "model")

    assert state.step == 2
    assert state.tokens_seen >= config.max_tokens
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
    assert (export / "config.json").is_file()
    assert (export / "model.safetensors").is_file()
    assert (export / "tokenizer_config.json").is_file()
    assert (tmp_path / "run" / "tokenizer" / "tokenizer.json").is_file()


def test_auto_refuses_detected_tpu_before_loading_the_corpus(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_tpu", ModuleType("torch_tpu"))
    monkeypatch.setattr(
        "trainlm.runtime.torchtpu_discovery.discover_tpu",
        lambda *args, **kwargs: {
            "status": "available_unverified",
            "device_type": "tpu",
            "device_count_visible": 1,
            "torch_tpu_version": "0.2.0",
        },
    )
    config = _config(tmp_path, accelerator="auto")

    with pytest.raises(RuntimeError, match="automatic TPU routing remains disabled"):
        TrainLMTrainer.from_scratch(config)


def test_native_tpu_allocation_failure_precedes_dataset_download(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_tpu", ModuleType("torch_tpu"))
    monkeypatch.setattr(
        "trainlm.runtime.torchtpu_discovery.discover_tpu",
        lambda *args, **kwargs: {
            "status": "allocation_failed",
            "device_type": "tpu",
            "device_count_visible": 1,
            "error_type": "RuntimeError",
        },
    )
    config = _config(tmp_path, accelerator="native_tpu")

    with pytest.raises(RuntimeError, match="allocation preflight failed"):
        TrainLMTrainer.from_scratch(config)
