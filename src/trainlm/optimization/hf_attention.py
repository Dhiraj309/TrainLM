"""Guarded integration with Hugging Face attention and mask interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .attention import CanonicalAttentionSpec


@dataclass(frozen=True, slots=True)
class HFAttentionProvider:
    """Explicit semantic contract for one HF attention implementation."""

    provider_id: str
    attention_forward: Callable[..., Any]
    mask_factory: Callable[..., Any]
    layouts: tuple[str, ...]
    mask_layouts: tuple[str, ...]
    position_encodings: tuple[str, ...]
    supports_segments: bool = False
    supports_dropout: bool = False
    supports_soft_cap: bool = False
    supports_qk_normalization: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id cannot be empty.")
        if not callable(self.attention_forward) or not callable(self.mask_factory):
            raise TypeError("Attention and mask implementations must be callable.")
        for name in ("layouts", "mask_layouts", "position_encodings"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        for name in (
            "supports_segments",
            "supports_dropout",
            "supports_soft_cap",
            "supports_qk_normalization",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")

    def incompatibilities(self, spec: CanonicalAttentionSpec) -> tuple[str, ...]:
        reasons = []
        if spec.layout not in self.layouts:
            reasons.append(f"layout {spec.layout!r}")
        if spec.mask.layout not in self.mask_layouts:
            reasons.append(f"mask layout {spec.mask.layout!r}")
        if spec.position_encoding not in self.position_encodings:
            reasons.append(f"position encoding {spec.position_encoding!r}")
        if spec.mask.segment_ids and not self.supports_segments:
            reasons.append("segment IDs")
        if spec.dropout and not self.supports_dropout:
            reasons.append("dropout")
        if spec.soft_cap is not None and not self.supports_soft_cap:
            reasons.append("soft cap")
        if spec.qk_normalization and not self.supports_qk_normalization:
            reasons.append("QK normalization")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class HFAttentionInstallation:
    provider_id: str
    model_class: str
    registered_attention: bool
    registered_mask: bool


def install_hf_attention_provider(
    model: Any,
    spec: CanonicalAttentionSpec,
    provider: HFAttentionProvider,
    *,
    attention_interface: Any | None = None,
    mask_interface: Any | None = None,
) -> HFAttentionInstallation:
    """Validate and register matching HF attention and mask implementations."""

    reasons = provider.incompatibilities(spec)
    if reasons:
        raise ValueError(
            f"Provider {provider.provider_id!r} is incompatible with: "
            + ", ".join(reasons)
        )
    if attention_interface is None or mask_interface is None:
        from transformers import AttentionInterface, AttentionMaskInterface

        attention_interface = AttentionInterface
        mask_interface = AttentionMaskInterface
    attention_register = getattr(attention_interface, "register", None)
    mask_register = getattr(mask_interface, "register", None)
    if not callable(attention_register) or not callable(mask_register):
        raise TypeError("Hugging Face attention and mask interfaces must support register().")

    setter = getattr(model, "set_attn_implementation", None)
    config = getattr(model, "config", None)
    if not callable(setter) and (
        config is None or not hasattr(config, "_attn_implementation")
    ):
        raise TypeError("Model does not expose a public attention selection boundary.")

    # Register both halves only after every compatibility and model-boundary
    # check has passed. Register the mask first so an attention implementation
    # cannot become selectable without its corresponding causal-mask factory.
    # HF mask dispatch uses the same key and an attention-only registration can
    # silently route through incorrect causal-mask semantics.
    mask_register(provider.provider_id, provider.mask_factory)
    attention_register(provider.provider_id, provider.attention_forward)
    if callable(setter):
        setter(provider.provider_id)
    else:
        config._attn_implementation = provider.provider_id
    return HFAttentionInstallation(
        provider_id=provider.provider_id,
        model_class=type(model).__name__,
        registered_attention=True,
        registered_mask=True,
    )


def expected_causal_visibility(
    spec: CanonicalAttentionSpec,
    *,
    query_position: int,
    key_position: int,
    same_segment: bool = True,
) -> bool:
    """Reference visibility predicate used by provider leakage tests."""

    if min(query_position, key_position) < 0:
        raise ValueError("Attention positions must be non-negative.")
    if spec.mask.segment_ids and not same_segment:
        return False
    if key_position > query_position:
        return False
    if spec.mask.layout == "causal_sliding_window":
        return key_position > query_position - spec.mask.sliding_window
    return True


__all__ = [
    "HFAttentionInstallation",
    "HFAttentionProvider",
    "expected_causal_visibility",
    "install_hf_attention_provider",
]
