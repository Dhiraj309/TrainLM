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
    cfg_kwargs = {k: v for k, v in cfg.to_dict().items() if k != "model_type"}
    built_cfg = AutoConfig.for_model("trainlm_loop", **cfg_kwargs)
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


def test_deep_init_scale_scales_output_projections_only():
    import math

    cfg_base = _tiny_loop_config(
        num_prelude_layers=1,
        num_core_layers=2,
        num_coda_layers=1,
        num_loop_iterations=4,
        deep_init_scale=False,
    )
    cfg_scaled = _tiny_loop_config(
        num_prelude_layers=1,
        num_core_layers=2,
        num_coda_layers=1,
        num_loop_iterations=4,
        deep_init_scale=True,
    )
    torch.manual_seed(123)
    m_base = LaughLMLoopForCausalLM(cfg_base)
    torch.manual_seed(123)
    m_scaled = LaughLMLoopForCausalLM(cfg_scaled)

    effective_layers = cfg_scaled.effective_num_layers()
    assert effective_layers == 1 + 2 * 4 + 1
    expected_scale = 1.0 / math.sqrt(2.0 * effective_layers)

    assert torch.equal(m_base.model.embed_tokens.weight, m_scaled.model.embed_tokens.weight)
    for grp_base, grp_scaled in (
        (m_base.model.prelude_layers, m_scaled.model.prelude_layers),
        (m_base.model.core_layers, m_scaled.model.core_layers),
        (m_base.model.coda_layers, m_scaled.model.coda_layers),
    ):
        for lb, ls in zip(grp_base, grp_scaled):
            assert torch.equal(lb.self_attn.q_proj.weight, ls.self_attn.q_proj.weight)
            assert torch.equal(lb.self_attn.k_proj.weight, ls.self_attn.k_proj.weight)
            assert torch.equal(lb.self_attn.v_proj.weight, ls.self_attn.v_proj.weight)
            assert torch.equal(lb.mlp.gate_proj.weight, ls.mlp.gate_proj.weight)
            assert torch.equal(lb.mlp.up_proj.weight, ls.mlp.up_proj.weight)
            assert torch.allclose(lb.self_attn.o_proj.weight * expected_scale, ls.self_attn.o_proj.weight)
            assert torch.allclose(lb.mlp.down_proj.weight * expected_scale, ls.mlp.down_proj.weight)


def test_backward_compatibility_for_config_without_new_fields(tmp_path: Path):
    import json

    cfg = _tiny_loop_config()
    model = LaughLMLoopForCausalLM(cfg)
    ckpt = tmp_path / "legacy_ckpt"
    model.save_pretrained(ckpt)

    cfg_path = ckpt / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw.pop("deep_init_scale", None)
    raw.pop("loop_depth_schedule", None)
    cfg_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    reloaded = LaughLMLoopForCausalLM.from_pretrained(ckpt)
    assert reloaded.config.deep_init_scale is False
    assert reloaded.config.loop_depth_schedule is None


def test_explicit_num_loops_respected_under_is_compiling(monkeypatch: pytest.MonkeyPatch):
    cfg = _tiny_loop_config(num_loop_iterations=4)
    model = LaughLMLoopForCausalLM(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with torch.no_grad():
        out_r2 = model(input_ids=input_ids, num_loops=2).logits
        out_r4 = model(input_ids=input_ids, num_loops=4).logits
    assert not torch.allclose(out_r2, out_r4)


def test_strict_backward_compatibility_state_dict_and_logits(tmp_path: Path):
    import json

    torch.manual_seed(7)
    legacy_cfg = _tiny_loop_config()
    legacy_model = LaughLMLoopForCausalLM(legacy_cfg)
    legacy_sd = {k: v.detach().clone() for k, v in legacy_model.state_dict().items()}

    # Default config must load legacy state dict strictly with zero new parameter keys.
    default_cfg = _tiny_loop_config()
    assert default_cfg.pack_doc_boundary_mask is False
    assert default_cfg.doc_boundary_eos_token_id is None
    assert default_cfg.use_value_residual is False
    assert default_cfg.use_qk_norm is False
    assert default_cfg.use_attn_output_gate is False

    target_model = LaughLMLoopForCausalLM(default_cfg)
    incompat = target_model.load_state_dict(legacy_sd, strict=True)
    assert incompat.missing_keys == []
    assert incompat.unexpected_keys == []

    ckpt = tmp_path / "pilot_compat"
    legacy_model.save_pretrained(ckpt)
    cfg_path = ckpt / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in (
        "pack_doc_boundary_mask",
        "doc_boundary_eos_token_id",
        "use_value_residual",
        "use_qk_norm",
        "use_attn_output_gate",
    ):
        raw.pop(key, None)
    cfg_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    reloaded = LaughLMLoopForCausalLM.from_pretrained(ckpt)
    reloaded.eval()
    legacy_model.eval()
    x = torch.randint(0, legacy_cfg.vocab_size, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=x).logits, legacy_model(input_ids=x).logits)


def test_doc_boundary_masking_leak_freedom_logits_and_gradients():
    from trainlm.model import compute_packed_doc_ids

    eos = 0
    seq1 = torch.tensor([[11, 12, 13, eos, 21, 22, 23, 24]], dtype=torch.long)
    seq2 = torch.tensor([[91, 92, 93, eos, 21, 22, 23, 24]], dtype=torch.long)

    doc_ids = compute_packed_doc_ids(seq1, eos_token_id=eos)
    assert doc_ids.dtype == torch.int32
    assert torch.equal(doc_ids, torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32))

    # 1. With pack_doc_boundary_mask=True, doc 2 (positions 4..7) has zero cross-doc leakage.
    torch.manual_seed(17)
    cfg_masked = _tiny_loop_config(
        eos_token_id=eos,
        pack_doc_boundary_mask=True,
        use_value_residual=True,
        use_qk_norm=True,
        use_attn_output_gate=True,
        num_loop_iterations=3,
        train_min_loops=3,
        train_max_loops=3,
    )
    model_masked = LaughLMLoopForCausalLM(cfg_masked)
    model_masked.eval()

    model_masked.zero_grad(set_to_none=True)
    out1 = model_masked(input_ids=seq1).logits
    out1[:, 4:, :].sum().backward()
    grad1 = model_masked.model.embed_tokens.weight.grad[[21, 22, 23, 24]].detach().clone()

    model_masked.zero_grad(set_to_none=True)
    out2 = model_masked(input_ids=seq2).logits
    out2[:, 4:, :].sum().backward()
    grad2 = model_masked.model.embed_tokens.weight.grad[[21, 22, 23, 24]].detach().clone()

    torch.testing.assert_close(out1[:, 4:, :], out2[:, 4:, :], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad1, grad2, atol=1e-6, rtol=1e-6)

    # 2. With pack_doc_boundary_mask=False, prefix differences leak into doc 2.
    model_masked.config.pack_doc_boundary_mask = False
    model_masked.zero_grad(set_to_none=True)
    unmasked1 = model_masked(input_ids=seq1).logits
    unmasked1[:, 4:, :].sum().backward()
    ugrad1 = model_masked.model.embed_tokens.weight.grad[[21, 22, 23, 24]].detach().clone()

    model_masked.zero_grad(set_to_none=True)
    unmasked2 = model_masked(input_ids=seq2).logits
    unmasked2[:, 4:, :].sum().backward()
    ugrad2 = model_masked.model.embed_tokens.weight.grad[[21, 22, 23, 24]].detach().clone()

    assert not torch.allclose(unmasked1[:, 4:, :], unmasked2[:, 4:, :], atol=1e-6)
    assert not torch.allclose(ugrad1, ugrad2, atol=1e-6)


def test_value_residual_qk_norm_and_output_gate_grads_and_compile():
    torch.manual_seed(29)
    cfg = _tiny_loop_config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_prelude_layers=1,
        num_core_layers=2,
        num_coda_layers=1,
        num_loop_iterations=3,
        train_min_loops=3,
        train_max_loops=3,
        truncated_bptt_steps=3,
        eos_token_id=0,
        pack_doc_boundary_mask=True,
        use_value_residual=True,
        use_qk_norm=True,
        use_attn_output_gate=True,
    )
    assert cfg.head_dim == 128

    model = LaughLMLoopForCausalLM(cfg)
    assert model.model.prelude_layers[0].self_attn.value_residual_logit is None
    for layer in (*model.model.core_layers, *model.model.coda_layers):
        v_logit = layer.self_attn.value_residual_logit
        assert v_logit is not None
        assert v_logit.shape == (1,)
        assert torch.allclose(v_logit, torch.tensor([-2.0]))
        assert layer.self_attn.q_norm is not None
        assert layer.self_attn.k_norm is not None
        assert layer.self_attn.o_gate_proj is not None
        assert layer.self_attn.o_gate_proj.weight.shape == (2, 256)

    x = torch.randint(1, 128, (2, 16))
    x[:, 7] = 0
    out = model(input_ids=x, labels=x)
    assert out.loss is not None
    out.loss.backward()

    for layer in (*model.model.prelude_layers, *model.model.core_layers, *model.model.coda_layers):
        for p in (
            layer.self_attn.q_norm.weight,
            layer.self_attn.k_norm.weight,
            layer.self_attn.o_gate_proj.weight,
        ):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().sum().item() > 0.0
        if layer.self_attn.value_residual_logit is not None:
            g = layer.self_attn.value_residual_logit.grad
            assert g is not None and torch.isfinite(g).all()
            assert g.abs().sum().item() > 0.0

    # Fullgraph torch.compile check with all opt-in features enabled.
    model.zero_grad(set_to_none=True)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    comp_out = compiled(input_ids=x, labels=x)
    comp_out.loss.backward()
    assert model.model.core_layers[0].self_attn.value_residual_logit.grad is not None
    assert model.model.core_layers[0].self_attn.o_gate_proj.weight.grad is not None

    # Also verify num_prelude_layers=0 captures V_1 on core pass 0 and mixes on pass >= 1.
    cfg_no_pre = _tiny_loop_config(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_prelude_layers=0,
        num_core_layers=1,
        num_coda_layers=0,
        num_loop_iterations=2,
        train_min_loops=2,
        train_max_loops=2,
        use_value_residual=True,
    )
    m_no_pre = LaughLMLoopForCausalLM(cfg_no_pre)
    assert m_no_pre.model.core_layers[0].self_attn.value_residual_logit is not None
    m_no_pre(input_ids=x, labels=x).loss.backward()
    assert m_no_pre.model.core_layers[0].self_attn.value_residual_logit.grad is not None


