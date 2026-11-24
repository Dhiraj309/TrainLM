from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from trainlm.kernels import (
    is_pallas_splash_available,
    is_triton_available,
    pallas_splash_attention,
    probe_pallas_bridge,
    triton_rms_norm,
    triton_swiglu,
)
from trainlm.kernels.triton import apply_triton_rms_norm, restore_triton_rms_norm
from trainlm.optimization import probe_torchtpu_kernels, probe_torchtpu_splash_kernel


class _DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int = 16, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.variance_epsilon)).to(hidden_states.dtype)


def test_triton_kernels_fallback_cleanly_on_cpu() -> None:
    x = torch.randn(2, 4, 16, requires_grad=True)
    w = torch.ones(16, requires_grad=True)
    out = triton_rms_norm(x, w, eps=1e-6)
    expected = _DummyRMSNorm(16)(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    g = torch.randn(2, 4, 16, requires_grad=True)
    u = torch.randn(2, 4, 16, requires_grad=True)
    sw = triton_swiglu(g, u)
    torch.testing.assert_close(sw, torch.nn.functional.silu(g) * u)

    assert isinstance(is_triton_available(), bool)


def test_pallas_splash_fallback_and_bridge_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_custom_jax(fn, name=""):
        return fn

    fake_torch_tpu = SimpleNamespace(
        __version__="0.2.0",
        _internal=SimpleNamespace(pallas=SimpleNamespace(custom_jax_kernel=_fake_custom_jax)),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", fake_torch_tpu)

    assert probe_pallas_bridge() is _fake_custom_jax
    inv = probe_torchtpu_kernels()
    assert inv.has_pallas_bridge is True
    assert isinstance(is_pallas_splash_available(), bool)

    monkeypatch.setitem(
        sys.modules,
        "torch_tpu",
        SimpleNamespace(__version__="0.2.0", pallas=SimpleNamespace(jax_op=lambda name: name)),
    )
    assert probe_pallas_bridge() is None

    q = torch.randn(1, 4, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out = pallas_splash_attention(q, k, v, causal=True)
    assert out.shape == (1, 16, 4, 8)
    assert probe_torchtpu_splash_kernel() is None or callable(probe_torchtpu_splash_kernel())


def test_apply_triton_rms_norm_no_triton_returns_clean_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trainlm.kernels.triton.rmsnorm as triton_rmsnorm_mod

    monkeypatch.setattr(triton_rmsnorm_mod, "_HAS_TRITON", False)
    model = nn.Sequential(_DummyRMSNorm(16))
    res = apply_triton_rms_norm(model)
    assert res.applied is False
    assert res.patched_modules == 0
    assert restore_triton_rms_norm(model) == 0


def test_pallas_cross_entropy_casts_int64_targets_to_int32(monkeypatch) -> None:
    import types
    import torch
    from trainlm.kernels.pallas import cross_entropy as ce_mod

    captured: dict[str, object] = {}

    def fake_fwd_op(hidden, weight, bias, targets):
        captured["targets_dtype"] = targets.dtype
        captured["targets_values"] = targets.clone()
        return (
            torch.tensor(1.5, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(float(targets.numel()), dtype=torch.float32),
        )

    fake_ns = types.SimpleNamespace(fake_ce_fwd=fake_fwd_op)
    monkeypatch.setattr(ce_mod, "_build_pallas_ce_ops", lambda **kwargs: ("fake_ce_fwd", "fake_ce_bwd"))
    monkeypatch.setattr(torch.ops, "trainlm_pallas", fake_ns, raising=False)

    hidden = torch.randn(8, 16, dtype=torch.bfloat16)
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    labels = torch.tensor([0, 1, 2, -100, 4, 5, 6, 7], dtype=torch.int64)

    mean_loss, mean_z = ce_mod.pallas_chunked_linear_cross_entropy(hidden, weight, labels)
    assert mean_loss.item() == 1.5
    assert mean_z is None
    assert captured["targets_dtype"] == torch.int32
    assert torch.equal(captured["targets_values"], labels.to(torch.int32))


def test_pallas_splash_gqa_head_dim_and_unsupported_fallback(monkeypatch) -> None:
    import types
    import torch
    from trainlm.kernels.pallas import splash_attention as sa_mod

    class _FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return types.SimpleNamespace(type="tpu")

    monkeypatch.setattr(sa_mod, "is_pallas_splash_available", lambda: True)
    sa_mod._UNSUPPORTED_SPLASH_SPECS.clear()

    build_calls: list[dict[str, object]] = []
    op_inputs: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        if kwargs["n_heads"] == 8:
            raise ValueError("unsupported spec")
        return ("fake_splash_fwd", "fake_splash_bwd")

    def fake_splash_fwd(q, k, v):
        op_inputs.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)

    monkeypatch.setattr(sa_mod, "_build_splash_ops", fake_build)
    monkeypatch.setattr(
        torch.ops,
        "trainlm_pallas",
        types.SimpleNamespace(fake_splash_fwd=fake_splash_fwd),
        raising=False,
    )

    # 1. head_dim=64 (<128 / not multiple of 128) must fall back to SDPA without calling _build_splash_ops.
    q64 = torch.randn(1, 9, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k64 = torch.randn(1, 3, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v64 = torch.randn(1, 3, 128, 64, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out64 = sa_mod.pallas_splash_attention(q64, k64, v64)
    assert out64.shape == (1, 128, 9, 64)
    assert build_calls == []

    # 2. GQA with head_dim=128 repeats KV heads to n_heads before calling fwd_op.
    q128 = torch.randn(1, 4, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k128 = torch.randn(1, 2, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v128 = torch.randn(1, 2, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out128 = sa_mod.pallas_splash_attention(q128, k128, v128)
    assert out128.shape == (1, 128, 4, 128)
    assert len(build_calls) == 1
    assert build_calls[0]["n_kv_heads"] == 4
    assert op_inputs == [((1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128))]

    # 3. _build_splash_ops errors are cached in _UNSUPPORTED_SPLASH_SPECS and fall back to SDPA.
    q_bad = torch.randn(1, 8, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    k_bad = torch.randn(1, 4, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    v_bad = torch.randn(1, 4, 128, 128, dtype=torch.float32).as_subclass(_FakeTPUTensor)
    out_bad_1 = sa_mod.pallas_splash_attention(q_bad, k_bad, v_bad)
    out_bad_2 = sa_mod.pallas_splash_attention(q_bad, k_bad, v_bad)
    assert out_bad_1.shape == (1, 128, 8, 128)
    assert out_bad_2.shape == (1, 128, 8, 128)
    assert len([c for c in build_calls if c["n_heads"] == 8]) == 1
