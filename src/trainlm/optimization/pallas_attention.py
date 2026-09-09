"""Version-guarded bridge for an explicitly supplied XLA Pallas MHA kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .hf_attention import HFAttentionProvider


@dataclass(frozen=True, slots=True)
class PallasAttentionRuntime:
    """Evidence required before a Pallas attention kernel becomes selectable."""

    torch_xla_version: str
    tested_torch_xla_versions: tuple[str, ...]
    kernel: Callable[..., Any]
    backward_verified: bool
    hlo_custom_call_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.torch_xla_version, str) or not self.torch_xla_version:
            raise ValueError("torch_xla_version cannot be empty.")
        if (
            not isinstance(self.tested_torch_xla_versions, tuple)
            or not self.tested_torch_xla_versions
            or any(
                not isinstance(version, str) or not version
                for version in self.tested_torch_xla_versions
            )
        ):
            raise ValueError("tested_torch_xla_versions must contain versions.")
        if not callable(self.kernel):
            raise TypeError("kernel must be callable.")
        if not isinstance(self.backward_verified, bool):
            raise TypeError("backward_verified must be boolean.")
        if not isinstance(self.hlo_custom_call_verified, bool):
            raise TypeError("hlo_custom_call_verified must be boolean.")

    def require_supported(self) -> None:
        if self.torch_xla_version not in self.tested_torch_xla_versions:
            raise RuntimeError(
                "Pallas attention is disabled for untested torch_xla version "
                f"{self.torch_xla_version!r}."
            )
        if not self.backward_verified:
            raise RuntimeError(
                "Pallas attention requires explicit backward correctness evidence."
            )


def pallas_mha_provider(
    runtime: PallasAttentionRuntime,
    *,
    provider_id: str = "trainlm.pallas_mha",
) -> HFAttentionProvider:
    """Build a guarded HF provider without importing optional XLA packages.

    ``runtime.kernel`` is an explicit adapter with the stable TrainLM call
    boundary ``kernel(query, key, value, *, causal, scale)``. This avoids
    guessing private torch_xla module paths or kernel signatures.
    """

    runtime.require_supported()

    def attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any | None = None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        del module, kwargs
        if attention_mask is not None:
            raise ValueError(
                "Pallas MHA currently accepts only its registered causal mask."
            )
        if dropout != 0.0:
            raise ValueError("Pallas MHA dropout is not implemented.")
        output = runtime.kernel(query, key, value, causal=True, scale=scaling)
        return output, None

    def causal_mask_factory(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        # The guarded kernel applies causality internally. Returning no tensor
        # prevents materialization of a dense quadratic mask.
        return None

    return HFAttentionProvider(
        provider_id=provider_id,
        attention_forward=attention_forward,
        mask_factory=causal_mask_factory,
        layouts=("mha",),
        mask_layouts=("causal",),
        position_encodings=("learned", "rope", "none"),
    )


__all__ = ["PallasAttentionRuntime", "pallas_mha_provider"]
