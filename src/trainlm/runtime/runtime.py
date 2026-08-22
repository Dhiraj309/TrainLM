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
        self._device = torch.device(device)
        self._precision = precision
        self._compile_enabled = compile_enabled

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

    def initialize(self) -> None:
        """Initialize backend resources."""

    def finalize(self) -> None:
        """Release backend resources."""

    def on_train_begin(self) -> None:
        """Handle the start of a training lifecycle."""

    def on_train_end(self) -> None:
        """Handle the end of a training lifecycle."""

    def on_step_begin(self, step: int) -> None:
        del step

    def on_step_end(self, step: int) -> None:
        del step

    def prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        return self.compile_model(model)

    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer:
        return optimizer

    def prepare_dataloader(self, dataloader: Any) -> Any:
        return dataloader

    def prepare_batch(self, batch: Any) -> Any:
        if self.device.type == "cpu":
            return batch
        return _move_to_device(batch, self.device)

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def compile_model(self, model: nn.Module) -> nn.Module:
        if not self._compile_enabled:
            return model
        return torch.compile(model)

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
        loss.backward()

    def clip_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def optimizer_step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def zero_grad(self, optimizer: Optimizer) -> None:
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
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        del state_dict

    def diagnostics(self) -> BackendDiagnostics:
        return BackendDiagnostics(
            backend=self.name,
            device_type=self.device.type,
            precision=self.precision,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            values={"compile_enabled": self._compile_enabled},
        )


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_to_device(item, device))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


# Backward-compatible name for the portable default backend.
Runtime = TorchRuntime

