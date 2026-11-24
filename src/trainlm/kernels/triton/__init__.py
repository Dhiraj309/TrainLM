"""Native GPU Triton kernels registered cleanly for eager and torch.compile."""

from .rmsnorm import (
    TritonRMSNormPatchResult,
    apply_triton_rms_norm,
    is_triton_available,
    restore_triton_rms_norm,
    triton_rms_norm,
)
from .swiglu import triton_swiglu

__all__ = [
    "TritonRMSNormPatchResult",
    "apply_triton_rms_norm",
    "is_triton_available",
    "restore_triton_rms_norm",
    "triton_rms_norm",
    "triton_swiglu",
]
