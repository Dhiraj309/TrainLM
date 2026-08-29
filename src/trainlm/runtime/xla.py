"""Optional PyTorch/XLA execution backend.

The module is importable without ``torch_xla`` installed. The dependency is
loaded only when ``XlaRuntime`` is instantiated, keeping the core package
portable for CPU/CUDA users.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from types import ModuleType
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .base import BackendDiagnostics, LogicalMesh, Precision
from .runtime import _move_to_device


class XlaRuntime:
    """PyTorch/XLA backend with lazy dependency loading and no SPMD policy."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        precision: Precision = "bf16",
        xm_module: Any | None = None,
        torch_xla_module: Any | None = None,
    ) -> None:
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported runtime precision: {precision}")

        if xm_module is None or torch_xla_module is None:
            loaded_torch_xla, loaded_xm = _load_torch_xla()
            torch_xla_module = torch_xla_module or loaded_torch_xla
            xm_module = xm_module or loaded_xm

        self._xm = xm_module
        self._torch_xla = torch_xla_module
        self._precision = precision
        self._device = (
            torch.device(device)
            if device is not None
            else self._xm.xla_device()
        )

    @property
    def name(self) -> str:
        return "pytorch-xla"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def precision(self) -> Precision:
        return self._precision

    @property
    def world_size(self) -> int:
        return int(self._xm.xrt_world_size())

    @property
    def rank(self) -> int:
        return int(self._xm.get_ordinal())

    @property
    def local_rank(self) -> int:
        get_local_ordinal = getattr(self._xm, "get_local_ordinal", None)
        return int(get_local_ordinal()) if callable(get_local_ordinal) else self.rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary_process(self) -> bool:
        is_master = getattr(self._xm, "is_master_ordinal", None)
        return bool(is_master()) if callable(is_master) else self.rank == 0

    def initialize(self) -> None:
        """Validate that the XLA device is available and addressable."""

        self._xm.xla_device()

    def finalize(self) -> None:
        self._xm.mark_step()

    def on_train_begin(self) -> None:
        """XLA initializes lazily; no eager graph work is performed."""

    def on_train_end(self) -> None:
        self._xm.mark_step()

    def on_step_begin(self, step: int) -> None:
        del step

    def on_step_end(self, step: int) -> None:
        del step
        self._xm.mark_step()

    def prepare_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)

    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer:
        return optimizer

    def prepare_dataloader(self, dataloader: Any) -> Any:
        return dataloader

    def prepare_batch(self, batch: Any) -> Any:
        return _move_to_device(batch, self.device)

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="xla", dtype=dtype)

    def compile_model(self, model: nn.Module) -> nn.Module:
        return model

    def create_mesh(self, mesh: LogicalMesh) -> LogicalMesh:
        if mesh.size != self.world_size:
            raise ValueError(
                f"Logical mesh size {mesh.size} does not match XLA world "
                f"size {self.world_size}."
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

    def scale_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        scale: float,
    ) -> None:
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("Gradient scale must be positive.")
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def optimizer_step(self, optimizer: Optimizer) -> None:
        self._xm.optimizer_step(optimizer, barrier=False)

    def zero_grad(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def synchronize(self) -> None:
        self._xm.mark_step()

    def barrier(self, name: str | None = None) -> None:
        rendezvous = getattr(self._xm, "rendezvous", None)
        if callable(rendezvous):
            rendezvous(name or "trainlm-barrier")
        else:
            self._xm.mark_step()

    def before_checkpoint(self, name: str) -> None:
        self.barrier(f"before-checkpoint:{name}")

    def after_checkpoint(self, name: str, *, success: bool) -> None:
        del success
        self.barrier(f"after-checkpoint:{name}")

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "device": str(self.device),
            "precision": self.precision,
            "world_size": self.world_size,
            "rank": self.rank,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict and state_dict.get("backend") not in {None, self.name}:
            raise ValueError(
                f"Cannot load runtime state for {state_dict.get('backend')!r}."
            )

    def diagnostics(self) -> BackendDiagnostics:
        version = getattr(self._torch_xla, "__version__", None)
        return BackendDiagnostics(
            backend=self.name,
            device_type=self.device.type,
            precision=self.precision,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            values={
                "torch_xla_version": version,
                "device": str(self.device),
            },
        )


def _load_torch_xla() -> tuple[ModuleType, ModuleType]:
    """Import PyTorch/XLA only at explicit backend construction time."""

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
    except ImportError as exc:  # pragma: no cover - depends on TPU profile
        raise ImportError(
            "XlaRuntime requires the optional 'tpu-xla' dependencies. "
            "Install TrainLM with `pip install -e .[tpu-xla]`."
        ) from exc
    return torch_xla, xm
