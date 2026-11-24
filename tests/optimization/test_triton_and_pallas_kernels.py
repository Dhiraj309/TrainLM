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
