from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from trainlm.api import TrainLMTrainingArguments
from trainlm.optimization import probe_torchtpu_kernels, probe_torchtpu_splash_kernel
from trainlm.runtime.runtime import TorchRuntime


def test_2k_rematerialization_and_tpu_precision_args() -> None:
    args = TrainLMTrainingArguments(output_dir="out", sequence_length=2048)
    assert args.loss_rematerialization == "auto"
    assert args.tpu_allow_excess_precision == "auto"

    with pytest.raises(ValueError, match="loss_rematerialization"):
        TrainLMTrainingArguments(output_dir="out", loss_rematerialization="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tpu_allow_excess_precision"):
        TrainLMTrainingArguments(output_dir="out", tpu_allow_excess_precision="yes")  # type: ignore[arg-type]


def test_probe_torchtpu_kernels_discovers_internal_pallas_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_flash(*args, **kwargs):
        return None

    fake_torch_tpu = SimpleNamespace(
        __version__="0.2.0",
        _internal=SimpleNamespace(pallas=SimpleNamespace(pallas_flash_attention=_fake_flash)),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", fake_torch_tpu)

    inv = probe_torchtpu_kernels()
    assert inv.has_flash_attention is True
    assert inv.flash_attention is _fake_flash
    assert probe_torchtpu_splash_kernel() is _fake_flash


def test_torch_runtime_cuda_sota_flags_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TorchRuntime(device="cpu", compile_enabled=True, compile_mode="max-autotune")
    runtime._device = torch.device("cuda")

    fake_matmul = SimpleNamespace(allow_tf32=False)
    fake_cudnn = SimpleNamespace(allow_tf32=False)
    state = {"flash": False, "mem": False}
    fake_cuda = SimpleNamespace(
        matmul=fake_matmul,
        flash_sdp_enabled=lambda: state["flash"],
        enable_flash_sdp=lambda v: state.__setitem__("flash", v),
        mem_efficient_sdp_enabled=lambda: state["mem"],
        enable_mem_efficient_sdp=lambda v: state.__setitem__("mem", v),
    )
    monkeypatch.setattr(torch.backends, "cuda", fake_cuda, raising=False)
    monkeypatch.setattr(torch.backends, "cudnn", fake_cudnn, raising=False)

    fake_inductor = SimpleNamespace(
        config=SimpleNamespace(epilogue_fusion=False, coordinate_descent_tuning=False)
    )
    monkeypatch.setattr(torch, "_inductor", fake_inductor, raising=False)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda _: None)

    runtime.on_train_begin()
    assert fake_matmul.allow_tf32 is True
    assert fake_cudnn.allow_tf32 is True
    assert state["flash"] is True
    assert state["mem"] is True
    assert fake_inductor.config.epilogue_fusion is True
    assert fake_inductor.config.coordinate_descent_tuning is True

    runtime.on_train_end()
    assert fake_matmul.allow_tf32 is False
    assert fake_cudnn.allow_tf32 is False
    assert state["flash"] is False
    assert state["mem"] is False
    assert fake_inductor.config.epilogue_fusion is False
    assert fake_inductor.config.coordinate_descent_tuning is False
