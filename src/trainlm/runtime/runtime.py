"""Portable single-process PyTorch execution backend."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .base import BackendDiagnostics, LogicalMesh, Precision


class TorchRuntime:
    """CPU/CUDA-capable baseline implementation of ``ExecutionBackend``."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        precision: Precision = "fp32",
        compile_enabled: bool = False,
    ) -> None:
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported runtime precision: {precision}")
        if not isinstance(compile_enabled, bool):
            raise ValueError("compile_enabled must be boolean.")
        self._device = torch.device(device)
        self._precision = precision
        self._compile_enabled = compile_enabled
        self._previous_matmul_precision: str | None = None
        self._optimizer: Optimizer | None = None
        self._unscaled_this_step = False
        self._grad_scaler = _make_cuda_grad_scaler(self._device, self._precision)

    @property
    def name(self) -> str:
        return "pytorch"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def precision(self) -> Precision:
        return self._precision

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def is_distributed(self) -> bool:
        return False

    @property
    def is_primary_process(self) -> bool:
        return True

    @property
    def mean_reduces_gradients_before_clipping(self) -> bool:
        return False

    def initialize(self) -> None:
        """Initialize backend resources."""

    def finalize(self) -> None:
        """Release backend resources."""

    def on_train_begin(self) -> None:
        """Handle the start of a training lifecycle."""
        if self._device.type == "cuda":
            getter = getattr(torch, "get_float32_matmul_precision", None)
            setter = getattr(torch, "set_float32_matmul_precision", None)
            if callable(getter) and callable(setter):
                self._previous_matmul_precision = getter()
                setter("high")

    def on_train_end(self) -> None:
        """Handle the end of a training lifecycle."""
        if self._previous_matmul_precision is not None:
            setter = getattr(torch, "set_float32_matmul_precision", None)
            if callable(setter):
                setter(self._previous_matmul_precision)
            self._previous_matmul_precision = None

    def on_step_begin(self, step: int) -> None:
        del step

    def on_step_end(self, step: int) -> None:
        del step

    def prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        return self.compile_model(model)

    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer:
        self._optimizer = optimizer
        return optimizer

    def prepare_dataloader(self, dataloader: Any) -> Any:
        return dataloader

    def prepare_batch(self, batch: Any) -> Any:
        if self.device.type == "cpu" or _all_tensors_on_device(batch, self.device):
            return batch
        return _move_to_device(
            batch,
            self.device,
            non_blocking=self.device.type == "cuda",
        )

    def configure_static_shapes(self, *, accumulation_steps: int) -> None:
        del accumulation_steps

    def observe_accumulation_steps(self, steps: int) -> None:
        del steps

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def compile_model(self, model: nn.Module) -> nn.Module:
        if not self._compile_enabled or getattr(model, "_orig_mod", None) is not None:
            return model
        return torch.compile(model)

    def compile_training_step(self, step_fn):
        """Compile single-microbatch step on CUDA when requested."""

        if not callable(step_fn):
            raise TypeError("step_fn must be callable.")
        if self._compile_enabled and self._device.type == "cuda":
            return torch.compile(step_fn)
        return step_fn

    def create_mesh(self, mesh: LogicalMesh) -> LogicalMesh:
        if mesh.size != self.world_size:
            raise ValueError(
                f"Logical mesh size {mesh.size} does not match runtime "
                f"world size {self.world_size}."
            )
        return mesh

    def shard_model(self, model: nn.Module, mesh: Any) -> nn.Module:
        del mesh
        return model

    def shard_optimizer(self, optimizer: Optimizer, mesh: Any) -> Optimizer:
        del mesh
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            return
        loss.backward()

    def clip_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> torch.Tensor | None:
        if (
            self._grad_scaler is not None
            and self._optimizer is not None
            and not self._unscaled_this_step
        ):
            self._grad_scaler.unscale_(self._optimizer)
            self._unscaled_this_step = True
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def scale_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        scale: float | torch.Tensor,
    ) -> None:
        if isinstance(scale, torch.Tensor):
            if scale.ndim != 0 or scale.requires_grad:
                raise ValueError("Gradient scale tensor must be a detached scalar.")
        elif (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or scale <= 0
        ):
            raise ValueError("Gradient scale must be positive.")
        grads = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not grads:
            return
        foreach_mul = getattr(torch, "_foreach_mul_", None)
        if callable(foreach_mul) and not isinstance(scale, torch.Tensor):
            foreach_mul(grads, scale)
        else:
            for grad in grads:
                grad.mul_(scale)

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Return a sum reduction; the portable runtime is single-process."""

        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be a torch.Tensor.")
        return value

    def optimizer_step(self, optimizer: Optimizer) -> None:
        if self._grad_scaler is not None:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            self._unscaled_this_step = False
            return
        optimizer.step()

    def zero_grad(self, optimizer: Optimizer) -> None:
        self._unscaled_this_step = False
        optimizer.zero_grad(set_to_none=True)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def barrier(self, name: str | None = None) -> None:
        del name
        self.synchronize()

    def before_checkpoint(self, name: str) -> None:
        self.barrier(f"before-checkpoint:{name}")

    def after_checkpoint(self, name: str, *, success: bool) -> None:
        del success
        self.barrier(f"after-checkpoint:{name}")

    def state_dict(self) -> Mapping[str, Any]:
        if self._grad_scaler is not None:
            return {"grad_scaler": self._grad_scaler.state_dict()}
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self._grad_scaler is not None and "grad_scaler" in state_dict:
            self._grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def diagnostics(self) -> BackendDiagnostics:
        values: dict[str, Any] = {"compile_enabled": self._compile_enabled}
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                values["device_name"] = torch.cuda.get_device_name(self.device)
            except Exception:
                pass
        return BackendDiagnostics(
            backend=self.name,
            device_type=self.device.type,
            precision=self.precision,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            values=values,
        )


def _make_cuda_grad_scaler(device: torch.device, precision: Precision) -> Any:
    if device.type != "cuda" or precision != "fp16":
        return None
    amp = getattr(torch, "amp", None)
    scaler_cls = getattr(amp, "GradScaler", None) if amp is not None else None
    if scaler_cls is not None:
        try:
            return scaler_cls("cuda")
        except TypeError:
            return scaler_cls()
    cuda_amp = getattr(torch.cuda, "amp", None)
    legacy_cls = getattr(cuda_amp, "GradScaler", None) if cuda_amp is not None else None
    return legacy_cls() if legacy_cls is not None else None


def _all_tensors_on_device(value: Any, device: torch.device) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device == device
    if isinstance(value, Mapping):
        return all(_all_tensors_on_device(item, device) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_tensors_on_device(item, device) for item in value)
    return True


def _move_to_device(
    value: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_to_device(item, device, non_blocking=non_blocking))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(
            _move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            _move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        ]
    return value


# Backward-compatible name for the portable default backend.
Runtime = TorchRuntime
