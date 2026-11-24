"""Opt-in 8-bit compute pretraining (FP8 / experimental INT8 via torchao)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

from torch import nn

QuantizedTrainingMode = Literal["none", "fp8", "int8"]


@dataclass(frozen=True, slots=True)
class QuantizedTrainingResult:
    """Record of applied 8-bit compute training conversion."""

    mode: Literal["fp8", "int8"]
    backend: str
    excluded_modules: tuple[str, ...]


def _default_module_filter(
    module: nn.Module,
    fqn: str,
    exclude_modules: tuple[str, ...],
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    leaf = fqn.rsplit(".", 1)[-1]
    if fqn in exclude_modules or leaf in exclude_modules:
        return False
    in_features = getattr(module, "in_features", 0)
    out_features = getattr(module, "out_features", 0)
    # Tensor Core / MXU 8-bit GEMMs require 16-multiple dimensions.
    return (in_features % 16 == 0) and (out_features % 16 == 0)


def apply_quantized_training(
    model: nn.Module,
    *,
    mode: Literal["fp8", "int8"],
    exclude_modules: tuple[str, ...] = ("lm_head", "embed_tokens"),
    allow_small_models: bool = False,
) -> QuantizedTrainingResult:
    """Convert eligible Linear layers for opt-in FP8 or INT8 training.

    Uses ``torchao.float8.convert_to_float8_training`` for ``mode="fp8"``
    (H100 / L4 / Ada) and ``torchao.prototype.quantized_training`` for
    ``mode="int8"`` (A100 / experimental).
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if mode not in {"fp8", "int8"}:
        raise ValueError(f"Unsupported quantized_training mode: {mode!r}")

    if mode == "int8" and not allow_small_models:
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if isinstance(hidden_size, int) and hidden_size < 768:
            warnings.warn(
                f"quantized_training='int8' on hidden_size={hidden_size} (< 768) "
                "often underperforms BF16 due to quantization overhead and narrow-layer "
                "outliers; prefer BF16 below ~500M params.",
                UserWarning,
                stacklevel=2,
            )

    def module_filter(module: nn.Module, fqn: str) -> bool:
        return _default_module_filter(module, fqn, exclude_modules)

    if mode == "fp8":
        try:
            from torchao.float8 import (  # type: ignore[import-not-found]
                convert_to_float8_training,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "quantized_training='fp8' requires 'torchao' with float8 support "
                "(pip install torchao)."
            ) from exc
        convert_to_float8_training(model, module_filter_fn=module_filter)
        return QuantizedTrainingResult(
            mode="fp8",
            backend="torchao.float8",
            excluded_modules=exclude_modules,
        )

    try:
        from torchao.prototype.quantized_training import (  # type: ignore[import-not-found]
            Int8MixedPrecisionTrainingConfig,
        )
        from torchao.quantization import quantize_  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "quantized_training='int8' requires 'torchao' with "
            "torchao.prototype.quantized_training support (pip install torchao)."
        ) from exc

    config = Int8MixedPrecisionTrainingConfig(
        output=True,
        grad_input=True,
        grad_weight=True,
    )
    quantize_(model, config, filter_fn=module_filter)
    return QuantizedTrainingResult(
        mode="int8",
        backend="torchao.prototype.quantized_training",
        excluded_modules=exclude_modules,
    )


__all__ = [
    "QuantizedTrainingMode",
    "QuantizedTrainingResult",
    "apply_quantized_training",
]
