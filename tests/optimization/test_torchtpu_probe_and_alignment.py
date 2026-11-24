from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from trainlm.api import TrainLMTrainingArguments
from trainlm.optimization import (
    inspect_model_alignment,
    pad_vocab_to_multiple,
    probe_torchtpu_kernels,
    probe_torchtpu_splash_kernel,
)


class _DummyEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.resized_to: int | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        self.resized_to = new_num_tokens
        self.config.vocab_size = new_num_tokens
        self.embed = nn.Embedding(new_num_tokens, self.config.hidden_size)
        return self.embed


def test_inspect_model_alignment_and_pad_vocab() -> None:
    model = _DummyEmbeddingModel(vocab_size=50257, hidden_size=256)
    report = inspect_model_alignment(model, sequence_length=512, alignment=128)
    assert report.vocab_aligned is False
    assert report.padded_vocab_size == 50304
    assert len(report.warnings) == 1

    new_vocab = pad_vocab_to_multiple(model, multiple=128)
    assert new_vocab == 50304
    assert model.resized_to == 50304

    aligned_report = inspect_model_alignment(model, sequence_length=512, alignment=128)
    assert aligned_report.vocab_aligned is True
    assert aligned_report.warnings == ()
    assert pad_vocab_to_multiple(model, multiple=128) is None


def test_pad_vocab_to_multiple_argument_validation() -> None:
    args = TrainLMTrainingArguments(output_dir="out")
    assert args.pad_vocab_to_multiple is None
    args_auto = TrainLMTrainingArguments(output_dir="out", pad_vocab_to_multiple="auto")
    assert args_auto.pad_vocab_to_multiple == "auto"
    args_int = TrainLMTrainingArguments(output_dir="out", pad_vocab_to_multiple=128)
    assert args_int.pad_vocab_to_multiple == 128
    with pytest.raises(ValueError, match="pad_vocab_to_multiple"):
        TrainLMTrainingArguments(output_dir="out", pad_vocab_to_multiple=0)


def test_probe_torchtpu_kernels_discovers_torch_tpu_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_splash(*args, **kwargs):
        return None

    fake_torch_tpu = SimpleNamespace(
        __version__="0.2.0",
        ops=SimpleNamespace(splash_attention=_fake_splash),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", fake_torch_tpu)

    inventory = probe_torchtpu_kernels()
    assert inventory.torch_tpu_version == "0.2.0"
    assert inventory.has_splash_attention is True
    assert inventory.splash_attention is _fake_splash
    assert probe_torchtpu_splash_kernel() is _fake_splash
