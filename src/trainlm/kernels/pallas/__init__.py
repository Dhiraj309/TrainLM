"""Native TPU Pallas kernels wired through TorchTPU's JAX-to-PyTorch bridge."""

from __future__ import annotations

from .splash_attention import (
    is_pallas_splash_available,
    pallas_splash_attention,
    probe_pallas_bridge,
)

__all__ = [
    "is_pallas_splash_available",
    "pallas_splash_attention",
    "probe_pallas_bridge",
]
