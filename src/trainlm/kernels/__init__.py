"""Native GPU (Triton) and TPU (Pallas) fused kernels for TrainLM."""

from .pallas import is_pallas_splash_available, pallas_splash_attention, probe_pallas_bridge
from .triton import (
    apply_triton_rms_norm,
    is_triton_available,
    restore_triton_rms_norm,
    triton_rms_norm,
    triton_swiglu,
)

__all__ = [
    "apply_triton_rms_norm",
    "is_pallas_splash_available",
    "is_triton_available",
    "pallas_splash_attention",
    "probe_pallas_bridge",
    "restore_triton_rms_norm",
    "triton_rms_norm",
    "triton_swiglu",
]
