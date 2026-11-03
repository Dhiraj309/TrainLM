from collections.abc import Mapping
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from trainlm import PackedBinDataset, TrainLMTrainer, TrainLMTrainingArguments
from trainlm.data.writer import write_packed_binary_shard


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 5)
        self.lm_head = nn.Linear(5, 8)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


def _write_dataset(path, tokens):
    write_packed_binary_shard(
        [tokens], path, shard_id="train", vocab_size=8
    )


def test_packed_bin_dataset_state_resumes_next_example_and_rejects_stale_data(
    tmp_path,
):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, [1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5])
    original = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="validation"
    )
    resumed = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="validation"
    )
    try:
        iterator = iter(original)
        first = next(iterator)
        state = json.loads(json.dumps(original.state_dict()))
        expected_next = next(iterator)

        resumed.load_state_dict(state)
        actual_next = next(iter(resumed))

        assert first["input_ids"].tolist() == [1, 2, 3, 4]
        assert torch.equal(actual_next["input_ids"], expected_next["input_ids"])

        stale_dir = tmp_path / "stale"
        _write_dataset(stale_dir, [7, 6, 5, 4, 3, 2, 1, 0, 2, 3, 4, 5])
        stale = PackedBinDataset.from_directory(
            stale_dir, sequence_length=4, split="validation"
        )
        try:
            with pytest.raises(ValueError, match="dataset_fingerprint"):
                stale.load_state_dict(state)
        finally:
            stale.close()
    finally:
        original.close()
        resumed.close()


def test_packed_iterable_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    data_dir = tmp_path / "packed"
    _write_dataset(data_dir, [1, 2, 3, 4, 5, 6, 7, 0] * 2)
    initial_model = TinyCausalLM()

    def make_trainer(model, data, output_dir, max_steps, save_steps=None):
        return TrainLMTrainer(
            model=model,
            train_dataset=data,
            args=TrainLMTrainingArguments(
                output_dir=output_dir,
                max_steps=max_steps,
                sequence_length=4,
                accelerator="cpu",
                learning_rate=1e-3,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                logging_steps=1,
                save_steps=save_steps,
                seed=31,
            ),
        )

    first_data = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="train", seed=31
    )
    first = make_trainer(
        deepcopy(initial_model), first_data, tmp_path / "resumed", 2, save_steps=2
    )
    try:
        assert first.train().step == 2
    finally:
        first_data.close()
    checkpoint = tmp_path / "resumed" / "checkpoint-2"
    assert (checkpoint / "trainer_state.pt").is_file()

    resumed_data = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="train", seed=31
    )
    resumed = make_trainer(
        deepcopy(initial_model), resumed_data, tmp_path / "resumed", 4
    )
    try:
        assert resumed.train(resume_from_checkpoint=checkpoint).step == 4
        resumed_weights = resumed.model.state_dict()

        reference_data = PackedBinDataset.from_directory(
            data_dir, sequence_length=4, split="train", seed=31
        )
        reference = make_trainer(
            deepcopy(initial_model), reference_data, tmp_path / "reference", 4
        )
        try:
            assert reference.train().step == 4
            for name, expected in reference.model.state_dict().items():
                torch.testing.assert_close(
                    resumed_weights[name], expected, rtol=0, atol=0
                )
        finally:
            reference_data.close()
    finally:
        resumed_data.close()


class StatefulExamples(IterableDataset):
    """Custom iterable with a small, versioned cursor contract."""

    def __init__(self, examples, *, dataset_id="tiny-stateful-examples-v1"):
        self.examples = examples
        self.dataset_id = dataset_id
        self.position = 0

    def __iter__(self):
        while self.position < len(self.examples):
            example = self.examples[self.position]
            self.position += 1
            yield example

    def state_dict(self):
        return {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "position": self.position,
        }

    def validate_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("Unsupported custom iterable dataset state schema.")
        if state.get("dataset_id") != self.dataset_id:
            raise ValueError("Custom iterable dataset identity does not match.")
        position = state.get("position")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position <= len(self.examples)
        ):
            raise ValueError("Custom iterable dataset position is invalid.")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.position = state["position"]


def test_custom_stateful_iterable_resume_matches_uninterrupted_training(tmp_path):
    examples = [
        {
            "input_ids": torch.tensor([(index + offset) % 8 for offset in range(4)]),
            "labels": torch.tensor([(index + offset) % 8 for offset in range(4)]),
        }
        for index in range(8)
    ]
    initial_model = TinyCausalLM()

    def make_trainer(model, dataset, output_dir, max_steps):
        return TrainLMTrainer(
            model=model,
            train_dataset=dataset,
            args=TrainLMTrainingArguments(
                output_dir=output_dir,
                max_steps=max_steps,
                sequence_length=4,
                accelerator="cpu",
                per_device_train_batch_size=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                seed=31,
            ),
        )

    first_data = StatefulExamples(deepcopy(examples))
    first = make_trainer(
        deepcopy(initial_model), first_data, tmp_path / "resumed", max_steps=2
    )
    assert first.train().step == 2
    checkpoint = first.save_state(tmp_path / "checkpoint")
    assert first_data.position == 2

    resumed_data = StatefulExamples(deepcopy(examples))
    resumed = make_trainer(
        deepcopy(initial_model), resumed_data, tmp_path / "resumed", max_steps=4
    )
    assert resumed.train(resume_from_checkpoint=checkpoint).step == 4

    reference_data = StatefulExamples(deepcopy(examples))
    reference = make_trainer(
        deepcopy(initial_model), reference_data, tmp_path / "reference", max_steps=4
    )
    assert reference.train().step == 4

    assert resumed_data.position == reference_data.position == 4
    for name, expected in reference.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name], expected, rtol=0, atol=0
        )


def test_packed_cursor_preflight_rejects_stale_data_before_trainer_mutation(tmp_path):
    data_dir = tmp_path / "packed"
    _write_dataset(data_dir, [1, 2, 3, 4, 5, 6, 7, 0] * 2)
    initial_model = TinyCausalLM()

    def make_trainer(model, data, output_dir):
        return TrainLMTrainer(
            model=model,
            train_dataset=data,
            args=TrainLMTrainingArguments(
                output_dir=output_dir,
                max_steps=2,
                sequence_length=4,
                accelerator="cpu",
                learning_rate=1e-3,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                seed=31,
            ),
        )

    source_data = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="train", seed=31
    )
    source = make_trainer(deepcopy(initial_model), source_data, tmp_path / "source")
    try:
        source.train()
        checkpoint = source.save_state(tmp_path / "checkpoint")
    finally:
        source_data.close()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["train_dataset_state"]["cursor"]["dataset_fingerprint"] = "stale"
    torch.save(payload, checkpoint)

    resumed_data = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="train", seed=31
    )
    resumed = make_trainer(
        deepcopy(initial_model), resumed_data, tmp_path / "resumed"
    )
    before_model = {
        name: value.detach().clone()
        for name, value in resumed.model.state_dict().items()
    }
    before_scheduler = deepcopy(resumed.scheduler.state_dict())
    before_trainer = deepcopy(resumed.engine.state)
    before_dataset = resumed_data.state_dict()
    try:
        with pytest.raises(ValueError, match="dataset_fingerprint"):
            resumed._load_training_checkpoint(resumed.engine, checkpoint)

        assert all(
            torch.equal(before_model[name], value)
            for name, value in resumed.model.state_dict().items()
        )
        assert resumed.engine.optimizer.state == {}
        assert resumed.scheduler.state_dict() == before_scheduler
        assert resumed.engine.state == before_trainer
        assert resumed_data.state_dict() == before_dataset
    finally:
        resumed_data.close()


def test_native_resume_gate_accepts_stateful_packed_iterable(tmp_path):
    data_dir = tmp_path / "packed"
    _write_dataset(data_dir, [1, 2, 3, 4, 5, 6, 7, 0])
    dataset = PackedBinDataset.from_directory(
        data_dir, sequence_length=4, split="train"
    )
    trainer = object.__new__(TrainLMTrainer)
    trainer.runtime = SimpleNamespace(name="torchtpu")
    trainer._train_loader_generator = None
    try:
        engine = SimpleNamespace(
            train_dataloader=DataLoader(dataset, batch_size=1, num_workers=0)
        )
        trainer._validate_native_resume_data(engine)

        class UntrackedRows(IterableDataset):
            def __iter__(self):
                return iter(())

        unsupported = SimpleNamespace(
            train_dataloader=DataLoader(UntrackedRows(), batch_size=1)
        )
        with pytest.raises(ValueError, match="state_dict/load_state_dict cursor"):
            trainer._validate_native_resume_data(unsupported)
    finally:
        dataset.close()
