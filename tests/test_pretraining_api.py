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
    with pytest.raises(ValueError, match="tokenizer-owned fields"):
        _config(tmp_path, model_config_overrides={"vocab_size": 32000})
    with pytest.raises(ValueError, match="JSON-compatible"):
        _config(tmp_path, model_config_overrides={"custom_value": object()})
    with pytest.raises(ValueError, match="configure only the default GPT-2"):
        _config(
            tmp_path,
            model_type="llama",
            model_config_overrides={"hidden_size": 16},
        )


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

    # A valid prepared cache must avoid even importing the optional dataset
    # package, let alone reconnecting to the pinned streaming source. A changed
    # sequence length can use the same tokenized corpus.
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    cached_trainer = TrainLMTrainer.from_scratch(
        _config(config.output_dir, sequence_length=4)
    )
    assert cached_trainer.model.config.model_type == "gpt2"
    assert cached_trainer.model.config.n_positions == 4
    assert calls == [
        ("org/tiny-stories", "a" * 40, "train", True),
        ("org/tiny-stories", "a" * 40, "train", True),
        ("org/tiny-stories", "a" * 40, "validation", True),
    ]
    assert (tmp_path / "run" / "preparation_cache.json").is_file()

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


def test_preparation_cache_rejects_changed_data_settings(tmp_path, monkeypatch):
    rows = [
        {"text": f"Tiny story about data cache {index}. " * 4}
        for index in range(16)
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
    output_dir = tmp_path / "run"
    TrainLMTrainer.from_scratch(_config(output_dir))

    with pytest.raises(ValueError, match="different dataset/tokenizer settings"):
        TrainLMTrainer.from_scratch(
            _config(output_dir, dataset_revision="b" * 40)
        )
    assert len(calls) == 3


def test_preparation_cache_rejects_corrupted_packed_payload(tmp_path, monkeypatch):
    rows = [
        {"text": f"Tiny story about packed payload {index}. " * 4}
        for index in range(16)
    ]

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: StreamingRows()
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    output_dir = tmp_path / "run"
    TrainLMTrainer.from_scratch(_config(output_dir))

    packed_data = output_dir / "data" / "train" / "train.bin"
    with packed_data.open("r+b") as stream:
        first_byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first_byte[0] ^ 1]))

    with pytest.raises(RuntimeError, match="Prepared-data cache failed validation"):
        TrainLMTrainer.from_scratch(_config(output_dir))


def test_from_scratch_builds_a_configured_huggingface_causal_model(
    tmp_path, monkeypatch
):
    rows = [
        {"text": f"A tiny story about stars and friendly dogs number {i}. " * 4}
        for i in range(12)
    ]

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    def load_dataset(dataset_id, *, revision, split, streaming):
        return StreamingRows()

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    config = _config(
        tmp_path / "llama-run",
        model_type="llama",
        model_config_overrides={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 8,
        },
        model_embedding_size=768,
        model_layers=6,
        model_attention_heads=12,
        max_tokens=3,
        sequence_length=4,
        max_eval_batches=1,
    )
    trainer = TrainLMTrainer.from_scratch(config)

    assert trainer.model.config.model_type == "llama"
    assert trainer.model.config.vocab_size == len(trainer.processing_class)
    assert trainer.model.config.pad_token_id == trainer.processing_class.pad_token_id
    state = trainer.train()
    assert state.step == 1
    assert torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"]))


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
