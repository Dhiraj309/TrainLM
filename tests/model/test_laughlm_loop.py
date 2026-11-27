"""Tests for LaughLM-Loop config, model, AutoConfig/AutoModel registration, and Trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.config import LaughLMLoopConfig
from trainlm.model import LaughLMLoopForCausalLM, LaughLMLoopModel


def _tiny_loop_config(**overrides) -> LaughLMLoopConfig:
    defaults = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_prelude_layers": 1,
        "num_core_layers": 1,
        "num_coda_layers": 1,
        "num_loop_iterations": 4,
        "train_min_loops": 2,
        "train_max_loops": 4,
        "truncated_bptt_steps": 4,
        "use_pallas_recurrent_norm": False,
        "max_position_embeddings": 64,
    }
    defaults.update(overrides)
    return LaughLMLoopConfig(**defaults)


def test_laughlm_loop_135m_preset_geometry_and_param_budget():
    cfg = LaughLMLoopConfig.laughlm_loop_135m()
    assert cfg.model_type == "trainlm_loop"
    assert cfg.head_dim == 128
    assert cfg.hidden_size % 128 == 0
    assert cfg.intermediate_size % 128 == 0
    assert cfg.vocab_size % 128 == 0
    assert cfg.effective_num_layers() == 20
    assert cfg.effective_num_layers(6) == 28

    model = LaughLMLoopForCausalLM(cfg)
    unique_params = {id(p): p.numel() for p in model.parameters()}
    stored_params = sum(unique_params.values())
    assert 100_000_000 <= stored_params <= 135_000_000
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_autoconfig_automodel_and_roundtrip_save_load(tmp_path: Path):
    cfg = _tiny_loop_config(use_pallas_recurrent_norm=True)
    built_cfg = AutoConfig.for_model("trainlm_loop", **cfg.to_dict())
    assert isinstance(built_cfg, LaughLMLoopConfig)

    model = AutoModelForCausalLM.from_config(built_cfg)
    assert isinstance(model, LaughLMLoopForCausalLM)
    assert isinstance(model.model, LaughLMLoopModel)
    assert model.lm_head.weight is model.model.embed_tokens.weight

    out_dir = tmp_path / "ckpt"
    model.save_pretrained(out_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir)
    assert isinstance(reloaded, LaughLMLoopForCausalLM)
    assert reloaded.config.num_loop_iterations == 4
    assert reloaded.config.truncated_bptt_steps == 4


def test_forward_backward_static_compile_and_truncated_bptt():
    torch.manual_seed(0)
    cfg = _tiny_loop_config(
        num_prelude_layers=1,
        num_core_layers=1,
        num_coda_layers=1,
        num_loop_iterations=4,
        train_min_loops=4,
        train_max_loops=4,
        truncated_bptt_steps=2,
        use_pallas_recurrent_norm=True,
    )
    model = LaughLMLoopForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None
    out.loss.backward()
    assert model.model.recurrent_gate.grad is not None
    assert model.model.recurrent_norm.weight.grad is not None
    assert model.model.core_layers[0].mlp.down_proj.weight.grad is not None

    # Compare variable inference depth (R=1 vs R=4) with deterministic eval outputs.
    model.eval()
    with torch.no_grad():
        logits_r1 = model(input_ids=input_ids, num_loops=1).logits
        logits_r4 = model(input_ids=input_ids, num_loops=4).logits
    assert not torch.allclose(logits_r1, logits_r4)

    # Fullgraph torch.compile compatibility on CausalLM forward + backward.
    model.train()
    model.zero_grad(set_to_none=True)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    comp_out = compiled(input_ids=input_ids, labels=labels)
    comp_out.loss.backward()
    assert model.model.core_layers[0].self_attn.q_proj.weight.grad is not None


def test_trainer_integration_with_chunked_linear_loss(tmp_path: Path):
    cfg = _tiny_loop_config(train_min_loops=2, train_max_loops=4)
    model = LaughLMLoopForCausalLM(cfg)
    gen = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, cfg.vocab_size, (4, 32), generator=gen)
    dataset = [{"input_ids": row, "labels": row.clone()} for row in tokens]

    trainer = TrainLMTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "run"),
            accelerator="cpu",
            per_device_train_batch_size=2,
            sequence_length=32,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            torch_compile=False,
            loss_implementation="chunked_linear",
            logits_chunk_size=16,
            optim="adamw_torch",
            report_to="none",
        ),
    )
    state = trainer.train()
    assert state.step == 2
