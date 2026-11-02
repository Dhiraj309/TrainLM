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
