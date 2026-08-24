"""
Optimizer configuration.

This module defines the configuration required to construct a PyTorch
optimizer. The actual optimizer implementation is provided by
``torch.optim`` and is created by the OptimizerFactory.

The configuration intentionally describes *what* optimizer to construct,
not *how* parameter groups are created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DTypeName = Literal["float32", "float16", "bfloat16"]
DecayMode = Literal["decoupled"]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """
    Optimizer configuration.

    Notes
    -----
    TrainLM delegates optimizer implementations to ``torch.optim``.
    """

    name: Literal[
        "adamw",
    ] = "adamw"

    learning_rate: float = 3e-4

    betas: tuple[float, float] = (0.9, 0.95)

    eps: float = 1e-8

    weight_decay: float = 0.1

    fused: bool = True

    parameter_dtype: DTypeName | None = None

    mu_dtype: DTypeName = "float32"

    nu_dtype: DTypeName = "float32"

    decay_mode: DecayMode = "decoupled"

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"Unsupported optimizer: {self.name}")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if (
            len(self.betas) != 2
            or any(
                not isinstance(beta, (int, float)) or not 0 <= beta < 1
                for beta in self.betas
            )
        ):
            raise ValueError("betas must contain two values in [0, 1).")
        if not isinstance(self.eps, (int, float)) or self.eps <= 0:
            raise ValueError("eps must be positive.")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not isinstance(self.fused, bool):
            raise ValueError("fused must be boolean.")
        valid_dtypes = {"float32", "float16", "bfloat16"}
        for name in ("parameter_dtype", "mu_dtype", "nu_dtype"):
            value = getattr(self, name)
            if value is not None and value not in valid_dtypes:
                raise ValueError(f"Unsupported {name}: {value}")
        if self.decay_mode != "decoupled":
            raise ValueError("Only decoupled AdamW weight decay is supported.")
