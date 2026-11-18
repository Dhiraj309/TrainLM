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


pytestmark = pytest.mark.usefixtures("compatible_pretraining_profile")


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


def test_pretraining_config_accepts_hub_revisions_and_disjoint_splits(tmp_path):
    assert _config(tmp_path, dataset_revision="main").dataset_revision == "main"
    assert _config(tmp_path, dataset_revision=None).dataset_revision == "main"
    with pytest.raises(ValueError, match="same source split"):
        _config(tmp_path, validation_split="train")
    with pytest.raises(ValueError, match="must not overlap"):
        _config(
            tmp_path,
            train_split="train[:90%]",
            validation_split="train[80%:]",
        )
    assert _config(
        tmp_path,
        train_split="train[:80%]",
        validation_split="train[80%:]",
    ).validation_split == "train[80%:]"
    with pytest.raises(ValueError, match="simple HF slices"):
        _config(tmp_path, train_split="train[-10:]")
    with pytest.raises(ValueError, match="requires dataset_id"):
        _config(tmp_path, dataset_id=None, dataset_revision=None,
                dataset_config_name="default")
    with pytest.raises(ValueError, match="divisible"):
        _config(tmp_path, model_embedding_size=15)
    assert _config(tmp_path, tokenizer_vocab_size=65_537).tokenizer_vocab_size == 65_537
    with pytest.raises(ValueError, match="packed uint32"):
        _config(tmp_path, tokenizer_vocab_size=(1 << 32) + 1)
    assert _config(
        tmp_path,
        model_name_or_path="org/model",
        tokenizer_mode="train_bpe",
        model_embedding_size=768,
        model_layers=6,
        model_attention_heads=12,
    ).tokenizer_mode == "train_bpe"
    with pytest.raises(ValueError, match="cannot use a newly trained tokenizer"):
        _config(
            tmp_path,
            model_name_or_path="org/model",
            model_initialization="pretrained",
            tokenizer_mode="train_bpe",
            model_embedding_size=768,
            model_layers=6,
            model_attention_heads=12,
        )
    with pytest.raises(ValueError, match="requires accelerator='native_tpu'"):
        _config(tmp_path, torch_compile=True)
    with pytest.raises(ValueError, match="gradient_checkpointing must be booleans"):
        _config(tmp_path, gradient_checkpointing="yes")
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


def test_pretraining_config_accepts_paired_local_text_files(tmp_path):
    train_file = tmp_path / "train.txt"
    validation_file = tmp_path / "validation.txt"
    train_file.write_text("training corpus\n", encoding="utf-8")
    validation_file.write_text("validation corpus\n", encoding="utf-8")
    local_source = {
        "dataset_id": None,
        "dataset_revision": None,
        "train_file": train_file,
        "validation_file": validation_file,
    }

    config = _config(tmp_path / "run", **local_source)
    assert config.dataset_id is None
    assert config.dataset_revision is None
    assert config.train_file == train_file.resolve()
    assert config.validation_file == validation_file.resolve()

    with pytest.raises(ValueError, match="configured together"):
        _config(
            tmp_path / "partial",
            dataset_id=None,
            dataset_revision=None,
            train_file=train_file,
        )
    with pytest.raises(ValueError, match="not both"):
        _config(tmp_path / "mixed", train_file=train_file,
                validation_file=validation_file)
    with pytest.raises(ValueError, match="must be distinct"):
        _config(
            tmp_path / "same",
            dataset_id=None,
            dataset_revision=None,
            train_file=train_file,
            validation_file=train_file,
        )


def test_tokenizer_capacity_uses_sparse_and_added_token_ids():
    from trainlm.pretraining import _tokenizer_vocab_capacity

    class SparseTokenizer:
        unk_token_id = 0
        pad_token_id = 1
        bos_token_id = 2
        eos_token_id = 70_000

        def __len__(self):
            return 4

        def get_vocab(self):
            return {"base": 0, "added": 70_000}

    assert _tokenizer_vocab_capacity(SparseTokenizer()) == 70_001


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
        _config(config.output_dir, save_total_limit=2)
    )
    assert cached_trainer.args.save_total_limit == 2
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
    assert "save_total_limit" not in run_manifest["configuration"]["training_args"]
    assert "save_total_limit" not in run_manifest["configuration"]["pretraining"]
    assert "gradient_checkpointing" not in run_manifest["configuration"]["pretraining"]

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
    with pytest.raises(RuntimeError, match="already bound to a different"):
        TrainLMTrainer.from_scratch(
            _config(config.output_dir, gradient_checkpointing=True)
        )


def test_from_scratch_resolves_hub_config_and_sliced_splits_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    rows = [
        {"text": f"A selected dataset row about trees and stars number {index}. " * 4}
        for index in range(16)
    ]
    calls = []
    datasets_module = ModuleType("datasets")

    def load_dataset(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return iter(rows)

    datasets_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    huggingface_hub = ModuleType("huggingface_hub")

    class FakeHfApi:
        def dataset_info(self, *, repo_id, revision):
            assert repo_id == "org/custom-corpus"
            assert revision == "main"
            return SimpleNamespace(sha="b" * 40)

    huggingface_hub.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    config = _config(
        tmp_path / "hub-config-run",
        dataset_id="org/custom-corpus",
        dataset_config_name="default",
        dataset_revision="main",
        train_split="train[:80%]",
        validation_split="train[80%:]",
    )
    trainer = TrainLMTrainer.from_scratch(config)

    assert trainer.model.config.model_type == "gpt2"
    assert len(calls) == 3
    assert all(dataset_id == "org/custom-corpus" for dataset_id, _ in calls)
    assert [kwargs["split"] for _, kwargs in calls] == [
        "train[:80%]",
        "train[:80%]",
        "train[80%:]",
    ]
    assert all(kwargs["revision"] == "b" * 40 for _, kwargs in calls)
    assert all(kwargs["name"] == "default" for _, kwargs in calls)

    cache = json.loads(
        (config.output_dir / "preparation_cache.json").read_text(encoding="utf-8")
    )
    assert cache["inputs"]["dataset_revision"] == "b" * 40
    assert cache["inputs"]["dataset_config_name"] == "default"
    run_config = json.loads(
        (config.output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert (
        run_config["configuration"]["pretraining"]["dataset_revision"] == "b" * 40
    )

    from trainlm._notebook_resume import build_resume_report
    from trainlm.pretraining_run import _capture_local_text_sources

    report = {"data": {}}
    _capture_local_text_sources(report, config)
    assert report["data"]["dataset_revision_requested"] == "main"
    assert report["data"]["dataset_revision_resolved"] == "b" * 40
    checkpoint = trainer._save_training_checkpoint(
        trainer.engine, "checkpoint-0"
    )
    resume = build_resume_report(
        config,
        trainer=trainer,
        checkpoint={"latest_path": str(checkpoint)},
    )
    assert resume["pretraining_config"]["dataset_revision"] == "b" * 40


def test_from_scratch_streams_local_text_files_and_checks_content_hashes(
    tmp_path, monkeypatch
):
    train_file = tmp_path / "train.txt"
    validation_file = tmp_path / "validation.txt"
    train_file.write_text(
        "".join(
            f"A local training sample about stories and trees {index}.\n"
            for index in range(16)
        ),
        encoding="utf-8",
    )
    validation_file.write_text(
        "".join(
            f"A held-out validation story about stars {index}.\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    calls = []

    class StreamingRows:
        def __init__(self, path):
            self.path = Path(path)

        def __iter__(self):
            return iter(
                {"text": line}
                for line in self.path.read_text(encoding="utf-8").splitlines()
            )

    def load_dataset(builder, *, data_files, split, streaming):
        calls.append((builder, data_files[split], split, streaming))
        return StreamingRows(data_files[split])

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    config = _config(
        tmp_path / "local-run",
        dataset_id=None,
        dataset_revision=None,
        train_file=train_file,
        validation_file=validation_file,
    )

    trainer = TrainLMTrainer.from_scratch(config)
    assert trainer.model.config.model_type == "gpt2"
    assert calls == [
        ("text", str(train_file.resolve()), "train", True),
        ("text", str(train_file.resolve()), "train", True),
        ("text", str(validation_file.resolve()), "validation", True),
    ]
    run_config = json.loads(
        (config.output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["configuration"]["pretraining"]["train_file"] == str(
        train_file.resolve()
    )
    cache = json.loads(
        (config.output_dir / "preparation_cache.json").read_text(encoding="utf-8")
    )
    assert cache["inputs"]["local_text_files"]["train"]["path"] == str(
        train_file.resolve()
    )
    assert len(cache["inputs"]["local_text_files"]["train"]["sha256"]) == 64

    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    cached = TrainLMTrainer.from_scratch(config)
    assert cached.model.config.model_type == "gpt2"
    assert len(calls) == 3

    train_file.write_text("changed corpus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different dataset/tokenizer settings"):
        TrainLMTrainer.from_scratch(config)


def test_scratch_gradient_checkpointing_trains_and_exports_with_cache(
    tmp_path, monkeypatch,
):
    rows = [
        {"text": f"A deterministic checkpointing sample number {index}. " * 5}
        for index in range(16)
    ]
    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    config = _config(tmp_path / "checkpointed-run", gradient_checkpointing=True)
    trainer = TrainLMTrainer.from_scratch(config)

    assert trainer.model.is_gradient_checkpointing
    assert trainer.model.config.use_cache is False
    manifest = json.loads(
        (config.output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert manifest["configuration"]["pretraining"]["gradient_checkpointing"] is True

    state = trainer.train()
    assert state.step == 2
    assert torch.isfinite(torch.tensor(state.loss))

    export = trainer.save_model(config.output_dir / "export")
    assert trainer.model.config.use_cache is False
    from transformers import AutoModelForCausalLM

    exported_model = AutoModelForCausalLM.from_pretrained(export)
    assert exported_model.config.use_cache is True


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
        _config(output_dir, max_tokens=22, save_total_limit=1),
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
    assert (output_dir / "checkpoints" / "checkpoint-2" / "trainer_state.pt").is_file()
    published = torch.load(
        output_dir / "checkpoints" / "checkpoint-3" / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert published["run_configuration"]["configuration"]["pretraining"][
        "max_tokens"
    ] == 22

    recovered = TrainLMTrainer.from_scratch(
        _config(output_dir, max_tokens=22, save_total_limit=1),
        extend_training_budget=True,
    )
    state = recovered.train(resume_from_checkpoint="latest")

    assert state.step == 4
    # Publication failed before checkpoint-3 gained its ownership marker. It
    # stays protected even after a later complete save prunes checkpoints 1/2.
    assert {path.name for path in (output_dir / "checkpoints").iterdir()} == {
        "checkpoint-3", "checkpoint-4"
    }
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


def _local_hf_tokenizer(path: Path, rows: list[dict[str, str]]):
    from trainlm.pretraining import _train_byte_level_tokenizer

    tokenizer, _ = _train_byte_level_tokenizer(
        rows, max_examples=len(rows), vocab_size=260
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def _install_streaming_rows(monkeypatch, rows):
    class StreamingRows:
        def __iter__(self):
            return iter(rows)

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: StreamingRows()
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)


def test_from_scratch_reuses_hf_tokenizer_and_random_hf_config_without_weights(
    tmp_path, monkeypatch
):
    from transformers import AutoModelForCausalLM, GPT2Config

    rows = [
        {"text": f"A small corpus for tokenizer reuse, row {index}. " * 5}
        for index in range(12)
    ]
    model_source = tmp_path / "model-config"
    tokenizer_source = tmp_path / "tokenizer"
    model_source.mkdir()
    tokenizer = _local_hf_tokenizer(tokenizer_source, rows)
    GPT2Config(
        vocab_size=len(tokenizer), n_positions=8, n_ctx=8,
        n_embd=16, n_layer=1, n_head=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).save_pretrained(model_source)
    _install_streaming_rows(monkeypatch, rows)

    def refuse_pretrained_weights(*args, **kwargs):
        raise AssertionError("random initialization must not load model weights")

    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", refuse_pretrained_weights
    )
    monkeypatch.setattr(
        "trainlm.pretraining._train_byte_level_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an HF tokenizer must not be trained again")
        ),
    )

    config = _config(
        tmp_path / "run",
        model_name_or_path=str(model_source),
        tokenizer_name_or_path=str(tokenizer_source),
        local_files_only=True,
        sequence_length=4,
        model_embedding_size=768,
        model_layers=6,
        model_attention_heads=12,
    )
    trainer = TrainLMTrainer.from_scratch(config)
    assert trainer.model.config.n_embd == 16
    assert trainer.model.config.n_layer == 1
    assert trainer.model.config.vocab_size == len(tokenizer)
    assert type(trainer.processing_class).__name__ == type(tokenizer).__name__
    manifest = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert manifest["configuration"]["model"]["source"]["initialization"] == "config"
    assert manifest["configuration"]["tokenizer"]["source"]["source_relation"] == (
        "independent_source_semantics_unverified"
    )
    cached_trainer = TrainLMTrainer.from_scratch(config)
    assert cached_trainer.model.config.vocab_size == len(tokenizer)


def test_from_scratch_continues_pretrained_hf_model_with_its_tokenizer(
    tmp_path, monkeypatch
):
    from transformers import AutoModelForCausalLM, GPT2Config

    rows = [
        {"text": f"A local checkpoint corpus, row {index}. " * 5}
        for index in range(12)
    ]
    model_source = tmp_path / "pretrained-model"
    tokenizer = _local_hf_tokenizer(model_source, rows)
    hf_config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=8, n_ctx=8,
        n_embd=16, n_layer=1, n_head=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    expected_model = AutoModelForCausalLM.from_config(hf_config)
    expected_weight = next(expected_model.parameters()).detach().clone()
    expected_model.save_pretrained(model_source)
    _install_streaming_rows(monkeypatch, rows)
    monkeypatch.setattr(
        "trainlm.pretraining._train_byte_level_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("continued pretraining must reuse the model tokenizer")
        ),
    )

    trainer = TrainLMTrainer.from_scratch(
        _config(
            tmp_path / "run",
            model_name_or_path=str(model_source),
            model_initialization="pretrained",
            local_files_only=True,
            sequence_length=4,
            model_embedding_size=768,
            model_layers=6,
            model_attention_heads=12,
        )
    )
    assert torch.equal(next(trainer.model.parameters()), expected_weight)
    assert len(trainer.processing_class) == trainer.model.config.vocab_size


def test_from_scratch_rejects_tokenizer_ids_outside_pretrained_embeddings(
    tmp_path, monkeypatch
):
    from transformers import AutoModelForCausalLM, GPT2Config

    rows = [
        {"text": f"A local checkpoint corpus, row {index}. " * 5}
        for index in range(12)
    ]
    model_source = tmp_path / "pretrained-model"
    tokenizer_source = tmp_path / "tokenizer"
    model_source.mkdir()
    tokenizer = _local_hf_tokenizer(tokenizer_source, rows)
    hf_config = GPT2Config(
        vocab_size=len(tokenizer) - 1, n_positions=8, n_ctx=8,
        n_embd=16, n_layer=1, n_head=2,
    )
    AutoModelForCausalLM.from_config(hf_config).save_pretrained(model_source)
    _install_streaming_rows(monkeypatch, rows)

    with pytest.raises(ValueError, match="will not resize pretrained embeddings"):
        TrainLMTrainer.from_scratch(
            _config(
                tmp_path / "run",
                model_name_or_path=str(model_source),
                model_initialization="pretrained",
                tokenizer_name_or_path=str(tokenizer_source),
                local_files_only=True,
                sequence_length=4,
                model_embedding_size=768,
                model_layers=6,
                model_attention_heads=12,
            )
        )


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


def test_incompatible_package_profile_precedes_dataset_preparation(
    tmp_path, monkeypatch,
):
    profile = {
        "schema_version": 1,
        "scope": "installed_metadata_only",
        "status": "incompatible",
        "accelerator": "cpu",
        "requirements": [{
            "package": "torch", "required": ">=2.9,<2.15",
            "installed": "2.15.0", "status": "mismatch",
        }],
        "limitations": ["metadata_only", "does_not_certify_accelerator_execution"],
    }
    monkeypatch.setattr(
        "trainlm.runtime.pretraining_profile.inspect_pretraining_profile",
        lambda **kwargs: profile,
    )
    prepared = []
    monkeypatch.setattr(
        "trainlm.api._prepare_scratch_training",
        lambda config: prepared.append(config),
    )

    with pytest.raises(RuntimeError, match="package preflight failed") as raised:
        TrainLMTrainer.from_scratch(_config(tmp_path))

    assert raised.value.profile is profile
    assert prepared == []
