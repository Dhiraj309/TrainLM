from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

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


def test_packed_public_trainer_update_matches_direct_causal_lm_reference(tmp_path):
    write_packed_binary_shard(
        [[1, 2, 3, 4]],
        tmp_path / "packed",
        shard_id="train",
        vocab_size=8,
    )
    reference_data = PackedBinDataset.from_directory(
        tmp_path / "packed", sequence_length=4, split="validation"
    )
    training_data = PackedBinDataset.from_directory(
        tmp_path / "packed", sequence_length=4, split="train", seed=17
    )
    try:
        packed_example = next(iter(reference_data))
        input_ids = packed_example["input_ids"].unsqueeze(0).long()

        torch.manual_seed(17)
        initial_model = TinyCausalLM()
        reference_model = deepcopy(initial_model)
        trainer_model = deepcopy(initial_model)

        learning_rate = 0.025
        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(), lr=learning_rate
        )
        logits = reference_model(input_ids).logits
        reference_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        reference_loss.backward()
        reference_grad_norm = torch.nn.utils.clip_grad_norm_(
            reference_model.parameters(), max_norm=1.0
        )
        reference_optimizer.step()

        trainer = TrainLMTrainer(
            model=trainer_model,
            train_dataset=training_data,
            optimizer=torch.optim.SGD(
                trainer_model.parameters(), lr=learning_rate
            ),
            args=TrainLMTrainingArguments(
                output_dir=tmp_path / "run",
                max_steps=1,
                sequence_length=4,
                accelerator="cpu",
                learning_rate=learning_rate,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                logging_steps=1,
            ),
        )
        state = trainer.train()

        assert state.step == 1
        assert state.loss == pytest.approx(reference_loss.item(), rel=1e-6)
        assert state.grad_norm == pytest.approx(reference_grad_norm.item(), rel=1e-6)
        for reference, trained in zip(
            reference_model.parameters(), trainer_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                trained.grad, reference.grad, rtol=1e-6, atol=1e-7
            )
            torch.testing.assert_close(trained, reference, rtol=1e-6, atol=1e-7)
    finally:
        reference_data.close()
        training_data.close()
