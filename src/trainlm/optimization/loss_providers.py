"""Declarative provider catalog for linear causal-language-model loss."""

from __future__ import annotations

from .planner import OperationRequest, ProviderSpec

LINEAR_CAUSAL_LOSS_REQUIREMENTS = (
    "backward",
    "causal_shift",
    "ignore_index",
    "loss_mask",
    "z_loss",
)


def causal_loss_request() -> OperationRequest:
    return OperationRequest(
        component="lm_head",
        operation="linear_causal_cross_entropy",
        requirements=LINEAR_CAUSAL_LOSS_REQUIREMENTS,
    )


def causal_loss_provider_specs() -> tuple[ProviderSpec, ...]:
    """Return provider declarations without importing optional packages."""

    common = {
        "component": "lm_head",
        "operation": "linear_causal_cross_entropy",
        "capability_kinds": ("linear",),
        "supported_requirements": LINEAR_CAUSAL_LOSS_REQUIREMENTS,
    }
    return (
        ProviderSpec(
            provider_id="trainlm.pallas_linear_ce",
            backends=("xla",),
            precisions=("bf16", "fp32"),
            runtime_requirements=(
                "torch_xla",
                "pallas_linear_ce",
                "pallas_linear_ce_backward",
            ),
            priority=300,
            **common,
        ),
        ProviderSpec(
            provider_id="trainlm.tokamax_linear_ce",
            backends=("xla",),
            precisions=("bf16", "fp32"),
            runtime_requirements=(
                "tokamax_linear_ce",
                "tokamax_linear_ce_backward",
            ),
            priority=200,
            **common,
        ),
        ProviderSpec(
            provider_id="trainlm.chunked_linear_ce",
            backends=("cpu", "cuda", "xla"),
            precisions=("fp32", "fp16", "bf16"),
            fallback=True,
            priority=100,
            **common,
        ),
    )


__all__ = [
    "LINEAR_CAUSAL_LOSS_REQUIREMENTS",
    "causal_loss_provider_specs",
    "causal_loss_request",
]
