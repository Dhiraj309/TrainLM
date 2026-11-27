"""Native TPU Pallas kernels wired through TorchTPU's JAX-to-PyTorch bridge."""

from __future__ import annotations

from .cross_entropy import (
    is_pallas_ce_available,
    pallas_chunked_linear_cross_entropy,
)
from .recurrent_norm import (
    fused_recurrent_inject_rmsnorm,
    is_pallas_recurrent_norm_available,
)
from .splash_attention import (
    is_pallas_splash_available,
    pallas_splash_attention,
    probe_pallas_bridge,
)

__all__ = [
    "fused_recurrent_inject_rmsnorm",
    "is_pallas_ce_available",
    "is_pallas_recurrent_norm_available",
    "is_pallas_splash_available",
    "pallas_chunked_linear_cross_entropy",
    "pallas_splash_attention",
    "probe_pallas_bridge",
]
