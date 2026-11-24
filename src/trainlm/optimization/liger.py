"""Optional Liger-Kernel Triton fusion adapter for CUDA HF causal LMs."""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True, slots=True)
class LigerKernelResult:
    """Record of Liger-Kernel Triton patches applied to a model."""

    model_type: str
    fused_rms_norm: bool
    fused_rope: bool
    fused_swiglu: bool
    fused_linear_cross_entropy: bool


def _resolve_liger_apply_fn(model_type: str) -> Any | None:
    try:
        import liger_kernel.transformers as liger_tx  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    try:
        from liger_kernel.transformers.monkey_patch import (  # type: ignore[import-not-found]
            MODEL_TYPE_TO_APPLY_LIGER_FN,
        )

        if isinstance(MODEL_TYPE_TO_APPLY_LIGER_FN, dict):
            fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
            if callable(fn):
                return fn
    except Exception:
        pass

    candidate = getattr(liger_tx, f"apply_liger_kernel_to_{model_type}", None)
    if callable(candidate):
        return candidate
    return None


def apply_liger_kernels(
    model: nn.Module,
    *,
    fuse_linear_cross_entropy: bool = True,
    strict: bool = False,
) -> LigerKernelResult | None:
    """Apply optional Liger-Kernel Triton ops to a supported HF causal LM."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    if importlib.util.find_spec("liger_kernel") is None:
        if strict:
            raise RuntimeError(
                "use_liger_kernel=True requires the 'liger-kernel' package "
                "(pip install liger-kernel)."
            )
        return None

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        if strict:
            raise ValueError(
                "use_liger_kernel=True requires model.config.model_type."
            )
        return None

    apply_fn = _resolve_liger_apply_fn(model_type)
    if apply_fn is None:
        if strict:
            raise ValueError(
                f"liger-kernel does not support model_type={model_type!r}."
            )
        return None

    sig = inspect.signature(apply_fn)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "model" in params:
        kwargs["model"] = model
    if "rope" in params:
        kwargs["rope"] = True
    if "rms_norm" in params:
        kwargs["rms_norm"] = True
    if "swiglu" in params:
        kwargs["swiglu"] = True
    if "geglu" in params:
        kwargs["geglu"] = True
    if "fused_linear_cross_entropy" in params:
        kwargs["fused_linear_cross_entropy"] = fuse_linear_cross_entropy
    if "cross_entropy" in params:
        kwargs["cross_entropy"] = not fuse_linear_cross_entropy

    apply_fn(**kwargs)
    return LigerKernelResult(
        model_type=model_type,
        fused_rms_norm=bool(kwargs.get("rms_norm", False)),
        fused_rope=bool(kwargs.get("rope", False)),
        fused_swiglu=bool(kwargs.get("swiglu", False) or kwargs.get("geglu", False)),
        fused_linear_cross_entropy=bool(kwargs.get("fused_linear_cross_entropy", False)),
    )


__all__ = ["LigerKernelResult", "apply_liger_kernels"]
