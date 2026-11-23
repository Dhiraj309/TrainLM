"""CPU/mocked-CUDA tests for native CUDA parity and speed optimizations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainlm.api import TrainLMTrainingArguments
from trainlm.config import OptimizerConfig
from trainlm.optimization.optimizers import AdamWStateDtype, create_optimizer
from trainlm.runtime.base import BackendDiagnostics
from trainlm.runtime.runtime import TorchRuntime, _move_to_device
from trainlm.training.performance import _resolve_hardware_peak, build_performance_summary


def test_training_arguments_allow_torch_compile_for_cuda_and_auto(tmp_path) -> None:
    cuda_args = TrainLMTrainingArguments(
        output_dir=str(tmp_path / "cuda"),
        accelerator="cuda",
        torch_compile=True,
        bf16=True,
    )
    assert cuda_args.torch_compile is True

    auto_args = TrainLMTrainingArguments(
        output_dir=str(tmp_path / "auto"),
        accelerator="auto",
        torch_compile=True,
    )
    assert auto_args.torch_compile is True

    with pytest.raises(ValueError, match="torch_compile"):
        TrainLMTrainingArguments(
            output_dir=str(tmp_path / "cpu"),
            accelerator="cpu",
            torch_compile=True,
        )


def test_torch_runtime_cuda_tf32_lifecycle_and_diagnostics(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(
        torch, "set_float32_matmul_precision", lambda mode: calls.append(mode)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100-SXM4-40GB"
    )

    runtime = TorchRuntime(device="cuda", precision="bf16", compile_enabled=True)
    assert runtime._grad_scaler is None

    runtime.on_train_begin()
    runtime.on_train_end()
    assert calls == ["high", "highest"]

    diag = runtime.diagnostics()
    assert diag.device_type == "cuda"
    assert diag.values["compile_enabled"] is True
    assert diag.values["device_name"] == "NVIDIA A100-SXM4-40GB"


def test_torch_runtime_cuda_fp16_grad_scaler_lifecycle(monkeypatch) -> None:
    events: list[str] = []

    class FakeScaledLoss:
        def backward(self) -> None:
            events.append("scaled_backward")

    class FakeGradScaler:
        def scale(self, loss):
            events.append("scale")
            return FakeScaledLoss()

        def unscale_(self, optimizer) -> None:
            events.append(f"unscale:{id(optimizer)}")

        def step(self, optimizer) -> None:
            events.append(f"step:{id(optimizer)}")

        def update(self) -> None:
            events.append("update")

        def state_dict(self):
            return {"scale": 1024.0}

        def load_state_dict(self, state) -> None:
            events.append(f"load:{state['scale']}")

    fake_scaler = FakeGradScaler()
    monkeypatch.setattr(
        "trainlm.runtime.runtime._make_cuda_grad_scaler",
        lambda device, precision: fake_scaler if device.type == "cuda" and precision == "fp16" else None,
    )

    runtime = TorchRuntime(device="cuda", precision="fp16")
    param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.1)
    runtime.prepare_optimizer(optimizer)

    runtime.backward(torch.tensor(2.0))
    runtime.clip_gradients([param], 1.0)
    # Second clip_gradients in the same step should not unscale twice.
    runtime.clip_gradients([param], 1.0)
    runtime.optimizer_step(optimizer)

    assert events == [
        "scale",
        "scaled_backward",
        f"unscale:{id(optimizer)}",
        f"step:{id(optimizer)}",
        "update",
    ]
    saved = runtime.state_dict()
    assert saved == {"grad_scaler": {"scale": 1024.0}}
    runtime.load_state_dict(saved)
    assert events[-1] == "load:1024.0"


def test_move_to_device_passes_non_blocking_flag() -> None:
    class FakeTensor(torch.Tensor):
        pass

    recorded: list[tuple[torch.device, bool]] = []
    tensor = torch.zeros(2, 2)
    orig_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        if args and isinstance(args[0], torch.device):
            recorded.append((args[0], bool(kwargs.get("non_blocking", False))))
            return self
        return orig_to(self, *args, **kwargs)

    # Use monkeypatch-free local patch via subclassing isn't needed; patch torch.Tensor.to briefly.
    torch.Tensor.to = spy_to  # type: ignore[method-assign]
    try:
        out = _move_to_device({"x": tensor}, torch.device("cuda"), non_blocking=True)
        assert out["x"] is tensor
        assert recorded == [(torch.device("cuda"), True)]
    finally:
        torch.Tensor.to = orig_to  # type: ignore[method-assign]


def test_fused_fp32_adamw_fast_paths_python_state_casting(monkeypatch) -> None:
    param = nn.Parameter(torch.ones(4, dtype=torch.float32))
    optimizer = create_optimizer(
        [param],
        OptimizerConfig(mu_dtype="float32", nu_dtype="float32", fused=False),
    )
    assert isinstance(optimizer, AdamWStateDtype)
    # Simulate fused=True on the optimizer defaults without needing CUDA tensors.
    optimizer.defaults["fused"] = True

    def fail_if_called():
        raise AssertionError("_cast_moment_state should be skipped on fused fp32 fast path")

    monkeypatch.setattr(optimizer, "_cast_moment_state", fail_if_called)
    monkeypatch.setattr(optimizer, "_prepare_step_dtypes", fail_if_called)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, closure=None: None)

    optimizer.step()


@pytest.mark.parametrize(
    ("device_name", "precision", "expected_peak"),
    [
        ("NVIDIA A100-SXM4-40GB", "bf16", 312_000_000_000_000.0),
        ("NVIDIA H100 80GB HBM3", "bf16", 989_000_000_000_000.0),
        ("NVIDIA H100 PCIe", "fp16", 756_000_000_000_000.0),
        ("NVIDIA L4", "bf16", 121_000_000_000_000.0),
        ("Tesla T4", "fp16", 65_000_000_000_000.0),
    ],
)
def test_cuda_peak_flops_and_mfu_summary(device_name: str, precision: str, expected_peak: float) -> None:
    peak, source = _resolve_hardware_peak(
        device_type="cuda",
        device_name=device_name,
        precision=precision,
    )
    assert peak == expected_peak
    assert source is not None and source.startswith("https://")

    class FakeRuntime:
        def diagnostics(self):
            return BackendDiagnostics(
                backend="pytorch",
                device_type="cuda",
                precision=precision,
                world_size=1,
                rank=0,
                local_rank=0,
                values={"device_name": device_name},
            )

    class FakeModel:
        config = SimpleNamespace(
            vocab_size=32000,
            hidden_size=512,
            num_hidden_layers=8,
            tie_word_embeddings=True,
            is_encoder_decoder=False,
        )

        def parameters(self):
            yield torch.empty(100_000_000)

    summary = build_performance_summary(
        measurements={
            "wall_seconds": 1.0,
            "measured_optimizer_steps": 5,
            "supervised_tokens": 50_000,
            "tokens_per_second": 50_000.0,
            "dataloader_wait_seconds": 0.01,
            "dataloader_wait_fraction": 0.01,
        },
        model=FakeModel(),
        runtime=FakeRuntime(),
        sequence_length=512,
    )
    assert summary["mfu_status"] == "measured"
    assert summary["hardware"]["peak_flops_per_device"] == expected_peak
    assert summary["non_embedding_mfu"] > 0.0
