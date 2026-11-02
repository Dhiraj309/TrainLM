"""Experimental single-device native TorchTPU execution backend.

This adapter deliberately reuses the portable PyTorch training engine. It does
not launch ranks, shard models, or claim multi-device support.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .base import BackendDiagnostics, Precision
from .runtime import TorchRuntime
from .torchtpu_discovery import discover_tpu


class TorchTPURuntime(TorchRuntime):
    """Single-process, single-visible-device native TorchTPU runtime.

    The native runtime is checked in the process that will own the model. The
    first allocation happens before moving model weights, so a failed/busy TPU
    produces a bounded diagnostic instead of a silent CPU fallback.
    """

    def __init__(
        self,
        *,
        precision: Precision = "fp32",
        seed: int = 42,
        compile_enabled: bool = False,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        # Explicit native_tpu selection is the opt-in allocation boundary.
        super().__init__(
            device="cpu", precision=precision, compile_enabled=compile_enabled
        )
        self._seed = seed
        self._native_report = discover_tpu(
            torch, verify_allocation=True, include_properties=False
        )
        if self._native_report["status"] != "ready":
            detail = self._native_report.get("error_type") or self._native_report["status"]
            raise RuntimeError(
                "Native TorchTPU initialization failed before model transfer "
                f"({detail}). Check the installed torch/TorchTPU profile and "
                "attached TPU runtime."
            )
        if self._native_report.get("device_count_visible") != 1:
            raise RuntimeError(
                "This experimental native TorchTPU runtime requires exactly "
                "one visible TPU device; multi-device training is not integrated."
            )
        try:
            self._device = torch.device("tpu")
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Native TorchTPU allocation passed, but this PyTorch process "
                "does not expose the registered 'tpu' device type."
            ) from exc

    @property
    def name(self) -> str:
        return "torchtpu"

    def prepare_model(self, model: nn.Module) -> nn.Module:
        return super().prepare_model(model)

    def initialize(self) -> None:
        # PyTorch's public seed helper covers the active accelerator backend.
        torch.manual_seed(self._seed)

    def synchronize(self) -> None:
        accelerator = getattr(torch, "accelerator", None)
        synchronize = getattr(accelerator, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError(
                "This PyTorch version does not expose the supported generic "
                "accelerator synchronization API required by native TorchTPU."
            )
        synchronize()

    def state_dict(self) -> Mapping[str, Any]:
        """Capture the device RNG required by a resumable native checkpoint."""
        accelerator = getattr(torch, "accelerator", None)
        get_rng_state = getattr(accelerator, "get_rng_state", None)
        if not callable(get_rng_state):
            raise RuntimeError(
                "This PyTorch version does not expose accelerator RNG capture; "
                "a resumable native TPU checkpoint cannot be written."
            )
        return {
            "backend": self.name,
            "precision": self.precision,
            "world_size": self.world_size,
            "seed": self._seed,
            "compile_enabled": self._compile_enabled,
            "device_rng_state": get_rng_state(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("backend") != self.name:
            raise ValueError(
                "Native TPU resume requires a checkpoint created by the "
                "native TorchTPU backend."
            )
        if state_dict.get("world_size") != 1:
            raise ValueError("Native single-device resume requires world_size=1.")
        if state_dict.get("precision") != self.precision:
            raise ValueError("Native TPU resume precision does not match the checkpoint.")
        if state_dict.get("compile_enabled", False) != self._compile_enabled:
            raise ValueError("Native TPU resume compile mode does not match the checkpoint.")
        if "device_rng_state" not in state_dict:
            raise ValueError("Native TPU checkpoint is missing device RNG state.")
        accelerator = getattr(torch, "accelerator", None)
        set_rng_state = getattr(accelerator, "set_rng_state", None)
        if not callable(set_rng_state):
            raise RuntimeError(
                "This PyTorch version does not expose accelerator RNG restore; "
                "native TPU device RNG restoration is unavailable."
            )
        set_rng_state(state_dict["device_rng_state"])

    def diagnostics(self) -> BackendDiagnostics:
        base = super().diagnostics()
        values = dict(base.values)
        report = self._native_report or {}
        values.update(
            {
                "native_tpu_status": report.get("status", "not_initialized"),
                "device_count_visible": report.get("device_count_visible"),
                "torch_tpu_version": report.get("torch_tpu_version"),
                "libtpu_version": report.get("libtpu_version"),
            }
        )
        return BackendDiagnostics(
            backend=base.backend,
            device_type=base.device_type,
            precision=base.precision,
            world_size=base.world_size,
            rank=base.rank,
            local_rank=base.local_rank,
            values=values,
        )


__all__ = ["TorchTPURuntime"]
