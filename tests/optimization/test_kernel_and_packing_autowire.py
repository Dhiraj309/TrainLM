"""CPU tests for Step 3 kernel, attention, and sequence packing auto-wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from trainlm.api import (
    TrainLMTrainer,
    TrainLMTrainingArguments,
    _maybe_pack_token_dataset,
)
from trainlm.optimization.torchtpu_kernels import (
    PackedGatedMLPBlock,
    apply_torchtpu_kernel_plan,
)


class _DummyGatedMLP(nn.Module):
    def __init__(self, hidden: int = 8, inter: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=True)
        self.up_proj = nn.Linear(hidden, inter, bias=True)
        self.down_proj = nn.Linear(inter, hidden, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _DummyCausalModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, inter: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=inter,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_act="silu",
            tie_word_embeddings=False,
            is_encoder_decoder=False,
            _attn_implementation="eager",
        )
        self.embed = nn.Embedding(vocab, hidden)
        self.mlp = _DummyGatedMLP(hidden, inter)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def set_attn_implementation(self, impl: str) -> None:
        self.config._attn_implementation = impl

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.mlp(self.embed(input_ids))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return SimpleNamespace(loss=loss, logits=logits)


class _TinySeqDataset(Dataset):
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self._rows[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def test_training_arguments_validate_kernel_and_packing_fields(tmp_path) -> None:
    args = TrainLMTrainingArguments(
        output_dir=str(tmp_path / "ok"),
        attn_implementation="sdpa",
        pack_gated_mlp=True,
        packing=True,
    )
    assert args.attn_implementation == "sdpa"
    assert args.pack_gated_mlp is True
    assert args.packing is True

    with pytest.raises(ValueError, match="attn_implementation"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad-attn"),
            attn_implementation="unknown",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="pack_gated_mlp"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad-mlp"),
            pack_gated_mlp="yes",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="packing"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "bad-pack"),
            packing="yes",  # type: ignore[arg-type]
        )


def test_kernel_plan_rollback_preserves_trained_packed_weights() -> None:
    model = _DummyCausalModel()
    plan = apply_torchtpu_kernel_plan(model, pack_mlp=True)
    assert plan.packed_mlp_paths == ("mlp",)
    assert isinstance(model.mlp, PackedGatedMLPBlock)

    with torch.no_grad():
        model.mlp.packed_gate_up.weight[:16].fill_(2.5)
        model.mlp.packed_gate_up.weight[16:].fill_(-1.25)
        assert model.mlp.packed_gate_up.bias is not None
        model.mlp.packed_gate_up.bias[:16].fill_(0.5)
        model.mlp.packed_gate_up.bias[16:].fill_(-0.75)

    plan.rollback(model)
    assert isinstance(model.mlp, _DummyGatedMLP)
    assert torch.allclose(model.mlp.gate_proj.weight, torch.full((16, 8), 2.5))
    assert torch.allclose(model.mlp.up_proj.weight, torch.full((16, 8), -1.25))
    assert torch.allclose(model.mlp.gate_proj.bias, torch.full((16,), 0.5))
    assert torch.allclose(model.mlp.up_proj.bias, torch.full((16,), -0.75))


def test_trainer_autowires_mlp_packing_sdpa_and_rolls_back_on_save(tmp_path) -> None:
    model = _DummyCausalModel()
    ds = _TinySeqDataset([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 3, 5, 7]])
    trainer = TrainLMTrainer(
        model=model,
        train_dataset=ds,
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            sequence_length=4,
            per_device_train_batch_size=2,
            accelerator="cpu",
            attn_implementation="sdpa",
            pack_gated_mlp=True,
            packing=True,
        ),
    )
    assert model.config._attn_implementation == "sdpa"
    assert isinstance(model.mlp, PackedGatedMLPBlock)
    trainer.train()

    save_dir = trainer.save_model(tmp_path / "saved")
    assert isinstance(model.mlp, _DummyGatedMLP)
    state = torch.load(save_dir / "pytorch_model.bin", map_location="cpu")
    assert "mlp.gate_proj.weight" in state
    assert "mlp.up_proj.weight" in state
    assert "mlp.packed_gate_up.weight" not in state


def test_maybe_pack_token_dataset_merges_variable_sequences() -> None:
    ds = _TinySeqDataset([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    packed = _maybe_pack_token_dataset(ds, sequence_length=4, eos_token_id=0)
    # [1,2,3,0] -> chunk 0; [4,5,0,6] -> chunk 1; remainder [7,8,9,0] -> chunk 2
    assert len(packed) == 3
    assert packed[0]["input_ids"].tolist() == [1, 2, 3, 0]
    assert packed[1]["input_ids"].tolist() == [4, 5, 0, 6]
    assert packed[2]["input_ids"].tolist() == [7, 8, 9, 0]
