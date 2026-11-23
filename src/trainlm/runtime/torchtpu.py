"""Experimental single-device native TorchTPU execution backend.

This adapter deliberately reuses the portable PyTorch training engine. It does
not launch ranks, shard models, or claim multi-device support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from .base import BackendDiagnostics, Precision
from .runtime import TorchRuntime
from .torchtpu_discovery import discover_tpu


def _device_rng_getter() -> Callable[[], Any] | None:
    """Resolve TorchTPU RNG capture through the public API for this PyTorch."""
    accelerator = getattr(torch, "accelerator", None)
    random_api = getattr(accelerator, "random", None)
    getter = getattr(random_api, "get_rng_state", None)
    if callable(getter):
        return getter

    # Retain compatibility with early adapter builds that exposed a generic
    # getter directly on torch.accelerator.
    getter = getattr(accelerator, "get_rng_state", None)
    if callable(getter):
        return getter

    # TorchTPU exposes this backend-specific fallback only for PyTorch versions
    # predating torch.accelerator.random. In newer PyTorch it is internal.
    if random_api is None:
        getter = getattr(getattr(torch, "tpu", None), "get_rng_state", None)
        if callable(getter):
            return getter
    return None


def _device_rng_setter() -> Callable[[Any], None] | None:
    """Resolve TorchTPU RNG restore through a supported public API."""
    accelerator = getattr(torch, "accelerator", None)
    random_api = getattr(accelerator, "random", None)
    setter = getattr(random_api, "set_rng_state", None)
    if callable(setter):
        return setter

    # TorchTPU currently publishes set_rng_state under torch.tpu.
    setter = getattr(getattr(torch, "tpu", None), "set_rng_state", None)
    if callable(setter):
        return setter

    # Retain compatibility with early adapter builds exposing the setter on
    # torch.accelerator directly.
    setter = getattr(accelerator, "set_rng_state", None)
    if callable(setter):
        return setter
    return None


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
            torch, verify_allocation=True, include_properties=True
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

    def compile_model(self, model: nn.Module) -> nn.Module:
        if not self._compile_enabled:
            return model
        return _compile_with_tpu_options(model)

    def compile_training_step(self, step_fn):
        if not callable(step_fn):
            raise TypeError("step_fn must be callable.")
        if not self._compile_enabled:
            return step_fn
        return _compile_with_tpu_options(step_fn)

    def on_train_begin(self) -> None:
        self._previous_eager_mode = None
        if self._compile_enabled:
            return
        tpu_mod = getattr(torch, "tpu", None)
        eager_enum = getattr(tpu_mod, "EagerMode", None)
        setter = getattr(tpu_mod, "set_eager_mode", None)
        defer_and_fuse = getattr(eager_enum, "DEFER_AND_FUSE", None)
        if callable(setter) and defer_and_fuse is not None:
            getter = getattr(tpu_mod, "get_eager_mode", None)
            if callable(getter):
                try:
                    self._previous_eager_mode = getter()
                except Exception:
                    self._previous_eager_mode = None
            setter(defer_and_fuse)

    def on_train_end(self) -> None:
        prev = getattr(self, "_previous_eager_mode", None)
        if prev is None:
            return
        tpu_mod = getattr(torch, "tpu", None)
        setter = getattr(tpu_mod, "set_eager_mode", None)
        if callable(setter):
            try:
                setter(prev)
            finally:
                self._previous_eager_mode = None

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
        get_rng_state = _device_rng_getter()
        if not callable(get_rng_state):
            raise RuntimeError(
                "This PyTorch version does not expose supported accelerator "
                "RNG capture; "
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

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Check native checkpoint compatibility without changing device RNG."""
        if not isinstance(state_dict, Mapping):
            raise ValueError("Native TPU checkpoint state must be a mapping.")
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
        set_rng_state = _device_rng_setter()
        if not callable(set_rng_state):
            raise RuntimeError(
                "This PyTorch/TorchTPU profile does not expose supported TPU "
                "RNG restore; "
                "native TPU device RNG restoration is unavailable."
            )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        set_rng_state = _device_rng_setter()
        # validate_state_dict has already established that this API is callable.
        set_rng_state(state_dict["device_rng_state"])

    def diagnostics(self) -> BackendDiagnostics:
        base = super().diagnostics()
        values = dict(base.values)
        report = self._native_report or {}
        values.update(
            {
                "native_tpu_status": report.get("status", "not_initialized"),
                "device_count_visible": report.get("device_count_visible"),
                "device_name": report.get("device_name"),
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


def _compile_with_tpu_options(target: Any) -> Any:
    from . import runtime as base_runtime

    compile_fn = base_runtime.torch.compile
    try:
        return compile_fn(target, backend="tpu", dynamic=False)
    except TypeError:
        return compile_fn(target)


__all__ = ["TorchTPURuntime"]
