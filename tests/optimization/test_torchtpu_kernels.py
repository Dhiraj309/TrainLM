import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from trainlm.optimization import (
    AttentionMaskSpec,
    CanonicalAttentionSpec,
    OptimizationPlanner,
    PackedGatedMLPBlock,
    TorchTPUSplashAttentionRuntime,
    apply_torchtpu_kernel_plan,
    causal_loss_provider_specs,
    causal_loss_request,
    probe_torchtpu_splash_kernel,
    torchtpu_splash_attention_provider,
)

from .test_capabilities import capabilities


def test_torchtpu_splash_provider_guards_and_dispatch():
    calls = []

    def kernel(query, key, value, *, causal, scale, query_heads, key_value_heads, sliding_window, block_q, block_k):
        calls.append((causal, scale, query_heads, key_value_heads, sliding_window, block_q, block_k))
        return query

    runtime = TorchTPUSplashAttentionRuntime(
        torch_tpu_version="0.2.0",
        tested_torch_tpu_versions=("0.2.0",),
        kernel=kernel,
        backward_verified=True,
        block_q=128,
        block_k=64,
    )
    spec = CanonicalAttentionSpec(
        schema_version=1,
        layout="gqa",
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        position_encoding="rope",
        mask=AttentionMaskSpec(layout="causal"),
    )
    provider = torchtpu_splash_attention_provider(runtime, spec)

    assert provider.mask_factory(batch_size=2, sequence_length=512) is None
    q = torch.randn(1, 4, 4, 8)
    out, weights = provider.attention_forward(object(), q, q, q)
    assert out is q
    assert weights is None
    assert calls == [(True, spec.effective_scale, 4, 2, None, 128, 64)]

    with pytest.raises(RuntimeError, match="untested torch_tpu version"):
        TorchTPUSplashAttentionRuntime(
            torch_tpu_version="0.1.0",
            tested_torch_tpu_versions=("0.2.0",),
            kernel=kernel,
            backward_verified=True,
        ).require_supported()

    with pytest.raises(RuntimeError, match="backward correctness evidence"):
        TorchTPUSplashAttentionRuntime(
            torch_tpu_version="0.2.0",
            tested_torch_tpu_versions=("0.2.0",),
            kernel=kernel,
            backward_verified=False,
        ).require_supported()


def test_apply_torchtpu_kernel_plan_packs_swiglu_preserves_parity_and_rolls_back():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = LlamaForCausalLM(config)
    input_ids = torch.randint(0, 32, (2, 6))

    ref_logits = model(input_ids=input_ids).logits.detach().clone()

    splash_runtime = TorchTPUSplashAttentionRuntime(
        torch_tpu_version="0.2.0",
        tested_torch_tpu_versions=("0.2.0",),
        kernel=lambda q, k, v, **kw: q,
        backward_verified=True,
    )
    previous_impl = getattr(model.config, "_attn_implementation", None)

    plan_result = apply_torchtpu_kernel_plan(
        model,
        splash_runtime=splash_runtime,
        pack_mlp=True,
    )

    assert plan_result.attention_installation is not None
    assert model.config._attn_implementation == "trainlm.torchtpu_splash"
    assert len(plan_result.packed_mlp_paths) == 2
    assert isinstance(model.model.layers[0].mlp, PackedGatedMLPBlock)

    # Restore attention implementation to eager to verify packed-MLP parity on CPU.
    model.set_attn_implementation(previous_impl or "eager")
    packed_out = model(input_ids=input_ids).logits
    torch.testing.assert_close(packed_out, ref_logits)
    packed_out.sum().backward()
    assert model.model.layers[0].mlp.packed_gate_up.weight.grad is not None

    plan_result.rollback(model)
    assert not isinstance(model.model.layers[0].mlp, PackedGatedMLPBlock)
    assert getattr(model.config, "_attn_implementation", None) == previous_impl


def test_loss_catalog_plans_for_torchtpu_backend():
    plan = OptimizationPlanner(causal_loss_provider_specs()).plan(
        capabilities(),
        backend="torchtpu",
        precision="bf16",
        policy="auto",
        requests=(causal_loss_request(),),
    )

    assert plan.decisions[0].selected_provider == "trainlm.chunked_linear_ce"
    assert probe_torchtpu_splash_kernel() is None or callable(probe_torchtpu_splash_kernel())


def test_torchtpu_splash_provider_accepts_and_forwards_segment_ids():
    seen = []

    def kernel(
        query,
        key,
        value,
        *,
        causal,
        scale,
        query_heads,
        key_value_heads,
        sliding_window,
        block_q,
        block_k,
        segment_ids=None,
    ):
        seen.append((segment_ids, causal, query_heads, key_value_heads, block_q, block_k))
        return query.transpose(1, 2)

    runtime = TorchTPUSplashAttentionRuntime(
        torch_tpu_version="0.2.0",
        tested_torch_tpu_versions=("0.2.0",),
        kernel=kernel,
        backward_verified=True,
    )
    spec = CanonicalAttentionSpec(
        schema_version=1,
        layout="mqa",
        query_heads=4,
        key_value_heads=1,
        head_dim=8,
        position_encoding="rope",
        mask=AttentionMaskSpec(layout="causal", segment_ids=True),
    )
    provider = torchtpu_splash_attention_provider(runtime, spec)
    assert provider.supports_segments is True

    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 1, 6, 8)
    v = torch.randn(2, 1, 6, 8)
    seg = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.int32)
    out, weights = provider.attention_forward(object(), q, k, v, segment_ids=seg)
    assert weights is None
    assert out.shape == (2, 6, 4, 8)
    assert len(seen) == 1 and torch.equal(seen[0][0], seg)

    with pytest.raises(ValueError, match="2D"):
        provider.attention_forward(object(), q, k, v, segment_ids=seg.reshape(-1))

    unverified = TorchTPUSplashAttentionRuntime(
        torch_tpu_version="0.2.0",
        tested_torch_tpu_versions=("0.2.0",),
        kernel=kernel,
        backward_verified=True,
        segment_ids_verified=False,
    )
    with pytest.raises(RuntimeError, match="segment_ids_verified"):
        torchtpu_splash_attention_provider(unverified, spec)

