"""Reversible fused ``torch.nn.functional.rms_norm`` replacement for HF RMSNorm modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class RMSNormPatchResult:
    """Summary of reversible ``F.rms_norm`` patching on a model."""

    applied: bool
    patched_modules: int
    reason: str | None = None


def _is_standard_rms_norm_module(module: nn.Module) -> bool:
    """Return True if ``module`` matches standard ``weight * norm(x)`` RMSNorm."""
    class_name = type(module).__name__
    if "Liger" in class_name or "Gemma" in class_name:
        return False
    if not class_name.endswith("RMSNorm"):
        return False
    if getattr(module, "add_unit_offset", False):
        return False
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    return hasattr(module, "variance_epsilon") or hasattr(module, "eps")


def apply_fused_rms_norm(model: nn.Module) -> RMSNormPatchResult:
    """Replace HF RMSNorm forward methods with fused ``F.rms_norm`` when available."""
    rms_norm_fn = getattr(F, "rms_norm", None)
    if not callable(rms_norm_fn):
        return RMSNormPatchResult(
            applied=False,
            patched_modules=0,
            reason="torch.nn.functional.rms_norm unavailable",
        )

    patched = 0
    for module in model.modules():
        if getattr(module, "_trainlm_rms_norm_patched", False):
            continue
        if not _is_standard_rms_norm_module(module):
            continue
        orig_forward = module.forward
        eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))

        def _fused_forward(
            hidden_states: torch.Tensor,
            _mod: nn.Module = module,
            _eps: float = eps,
        ) -> torch.Tensor:
            weight = _mod.weight
            if weight is not None and weight.dtype != hidden_states.dtype:
                weight = weight.to(dtype=hidden_states.dtype)
            return rms_norm_fn(
                hidden_states,
                (hidden_states.shape[-1],),
                weight,
                _eps,
            )

        module._trainlm_orig_forward = orig_forward  # type: ignore[attr-defined]
        module._trainlm_rms_norm_patched = True  # type: ignore[attr-defined]
        module.forward = _fused_forward  # type: ignore[method-assign]
        patched += 1

    if patched == 0:
        return RMSNormPatchResult(
            applied=False,
            patched_modules=0,
            reason="no compatible RMSNorm modules found",
        )
    return RMSNormPatchResult(applied=True, patched_modules=patched)


def restore_rms_norm(model: nn.Module) -> int:
    """Restore original ``forward`` methods on modules patched by ``apply_fused_rms_norm``."""
    restored = 0
    for module in model.modules():
        if not getattr(module, "_trainlm_rms_norm_patched", False):
            continue
        orig_forward: Any = getattr(module, "_trainlm_orig_forward", None)
        if orig_forward is not None:
            module.forward = orig_forward  # type: ignore[method-assign]
        if hasattr(module, "_trainlm_orig_forward"):
            delattr(module, "_trainlm_orig_forward")
        if hasattr(module, "_trainlm_rms_norm_patched"):
            delattr(module, "_trainlm_rms_norm_patched")
        restored += 1
    return restored
