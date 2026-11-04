"""Public raw-text pretraining API contracts and a tiny CPU lifecycle smoke."""

import json
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

    # A valid prepared cache and matching run manifest must avoid even importing
    # the optional dataset package, let alone reconnecting to the pinned stream.
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    cached_trainer = TrainLMTrainer.from_scratch(
        _config(config.output_dir)
    )
    assert cached_trainer.model.config.model_type == "gpt2"
    assert calls == [
        ("org/tiny-stories", "a" * 40, "train", True),
        ("org/tiny-stories", "a" * 40, "train", True),
        ("org/tiny-stories", "a" * 40, "validation", True),
    ]
    assert (tmp_path / "run" / "preparation_cache.json").is_file()
    run_manifest = json.loads(
        (tmp_path / "run" / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_manifest["configuration"]["training_args"]["learning_rate"] == 3e-4
    assert "AdamW" in run_manifest["configuration"]["optimizer"]["type"]
    assert run_manifest["configuration"]["scheduler"]["schedule"] == "cosine"

    checkpoint = trainer._save_training_checkpoint(
        trainer.engine, "checkpoint-0"
    )
    state = cached_trainer.train(resume_from_checkpoint=checkpoint)
    metrics = cached_trainer.evaluate()
    export = cached_trainer.save_model(tmp_path / "run" / "model")

    assert state.step == 2
    assert state.tokens_seen >= config.max_tokens
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
    assert (export / "config.json").is_file()
    assert (export / "model.safetensors").is_file()
    assert (export / "tokenizer_config.json").is_file()
    assert (tmp_path / "run" / "tokenizer" / "tokenizer.json").is_file()
    checkpoint = torch.load(
        tmp_path / "run" / "checkpoints" / "checkpoint-1" / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["run_configuration"] == run_manifest
    with pytest.raises(RuntimeError, match="already bound to a different"):
        TrainLMTrainer.from_scratch(
            _config(config.output_dir, learning_rate=6e-4)
        )


def test_scratch_resume_rejects_changed_run_settings_before_state_restore(
    tmp_path, monkeypatch
):
    rows = [
        {"text": f"Tiny deterministic training sample {index}. " * 5}
        for index in range(16)
    ]

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: StreamingRows()
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    source = TrainLMTrainer.from_scratch(_config(tmp_path / "source"))
    checkpoint = source._save_training_checkpoint(source.engine, "checkpoint-0")
    resumed = TrainLMTrainer.from_scratch(
        _config(tmp_path / "resume", learning_rate=6e-4)
    )
    before_model = {
        name: value.detach().clone()
        for name, value in resumed.model.state_dict().items()
    }
    before_optimizer = resumed.optimizer.state_dict()
    before_scheduler = resumed.scheduler.state_dict().copy()
    before_trainer_state = resumed.engine.state
    before_rng_state = torch.get_rng_state().clone()
    before_dataset_state = resumed.train_dataset.state_dict()

    with pytest.raises(ValueError, match="run configuration does not match"):
        resumed.train(resume_from_checkpoint=checkpoint)

    assert all(
        torch.equal(before_model[name], value)
        for name, value in resumed.model.state_dict().items()
    )
    assert resumed.optimizer.state_dict() == before_optimizer
    assert resumed.scheduler.state_dict() == before_scheduler
    assert resumed.engine.state is before_trainer_state
    assert torch.equal(torch.get_rng_state(), before_rng_state)
    assert resumed.train_dataset.state_dict() == before_dataset_state


def test_scratch_resume_can_extend_token_budget_with_cosine_restart(
    tmp_path, monkeypatch
):
    rows = [
        {"text": f"A deterministic packed training example number {index}. " * 5}
        for index in range(16)
    ]

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: StreamingRows()
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    output_dir = tmp_path / "extendable-run"
    initial = TrainLMTrainer.from_scratch(_config(output_dir, max_tokens=8))
    initial_state = initial.train()
    assert initial_state.step == 2
    checkpoint_path = output_dir / "checkpoints" / "checkpoint-2" / "trainer_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_optimizer_steps = [
        int(state["step"].item())
        for state in checkpoint["optimizer"]["state"].values()
        if "step" in state
    ]
    assert saved_optimizer_steps and max(saved_optimizer_steps) == 2

    run_config_path = output_dir / "run_config.json"
    original_manifest = json.loads(run_config_path.read_text(encoding="utf-8"))
    extended = TrainLMTrainer.from_scratch(
        _config(output_dir, max_tokens=22),
        extend_training_budget=True,
    )
    assert json.loads(run_config_path.read_text(encoding="utf-8")) == original_manifest
    assert extended.args.max_steps == 4

    state = extended.train(resume_from_checkpoint="latest")

    assert state.step == 4
    assert state.tokens_seen >= 22
    assert extended.scheduler.T_max == 2
    assert extended.scheduler.last_epoch == 2
    assert extended.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    final_manifest = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert final_manifest["configuration"]["pretraining"]["max_tokens"] == 22
    assert final_manifest["configuration"]["scheduler"]["training_horizon_steps"] == 4
    extension_checkpoint = torch.load(
        output_dir / "checkpoints" / "checkpoint-3" / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert extension_checkpoint["scheduler"]["T_max"] == 2
    assert extension_checkpoint["scheduler"]["base_lrs"] == [pytest.approx(3e-4)]
    optimizer_steps = [
        int(state["step"].item())
        for state in extended.optimizer.state.values()
        if "step" in state
    ]
    assert optimizer_steps and max(optimizer_steps) == 4


def test_scratch_budget_extension_rejects_non_budget_changes(tmp_path, monkeypatch):
    rows = [
        {"text": f"A deterministic packed training example number {index}. " * 5}
        for index in range(16)
    ]

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: StreamingRows()
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    output_dir = tmp_path / "fixed-run"
    trainer = TrainLMTrainer.from_scratch(_config(output_dir, max_tokens=8))
    trainer.train()
    run_config_path = output_dir / "run_config.json"
    original_manifest = json.loads(run_config_path.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="already bound to a different"):
        TrainLMTrainer.from_scratch(_config(output_dir, max_tokens=22))
    with pytest.raises(RuntimeError, match="only permits a strictly larger"):
        TrainLMTrainer.from_scratch(
            _config(output_dir, max_tokens=22, learning_rate=6e-4),
            extend_training_budget=True,
        )
    assert json.loads(run_config_path.read_text(encoding="utf-8")) == original_manifest


def test_scratch_budget_extension_recovers_after_checkpoint_publish(
    tmp_path, monkeypatch
):
    rows = [
        {"text": f"A deterministic packed training example number {index}. " * 5}
        for index in range(16)
    ]

    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: StreamingRows()
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    output_dir = tmp_path / "recoverable-run"
    initial = TrainLMTrainer.from_scratch(_config(output_dir, max_tokens=8))
    initial.train()
    run_config_path = output_dir / "run_config.json"
    original_manifest = json.loads(run_config_path.read_text(encoding="utf-8"))

    extending = TrainLMTrainer.from_scratch(
        _config(output_dir, max_tokens=22),
        extend_training_budget=True,
    )

    def fail_manifest_publish(*args, **kwargs):
        raise OSError("simulated manifest publish interruption")

    monkeypatch.setattr(
        extending,
        "_write_scratch_run_configuration",
        fail_manifest_publish,
    )
    with pytest.raises(OSError, match="simulated manifest publish interruption"):
        extending.train(resume_from_checkpoint="latest")
    assert json.loads(run_config_path.read_text(encoding="utf-8")) == original_manifest
    published = torch.load(
        output_dir / "checkpoints" / "checkpoint-3" / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert published["run_configuration"]["configuration"]["pretraining"][
        "max_tokens"
    ] == 22

    recovered = TrainLMTrainer.from_scratch(
        _config(output_dir, max_tokens=22),
        extend_training_budget=True,
    )
    state = recovered.train(resume_from_checkpoint="latest")

    assert state.step == 4
    assert json.loads(run_config_path.read_text(encoding="utf-8")) == published[
        "run_configuration"
    ]


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
