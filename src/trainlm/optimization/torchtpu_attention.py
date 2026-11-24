"""Guarded native TorchTPU Splash Attention provider and kernel probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .attention import CanonicalAttentionSpec
from .hf_attention import HFAttentionProvider
from .pallas_attention import KVHeadMapping


@dataclass(frozen=True, slots=True)
class TorchTPUSplashAttentionRuntime:
    """Evidence and block geometry required for native TorchTPU Splash Attention."""

    torch_tpu_version: str
    tested_torch_tpu_versions: tuple[str, ...]
    kernel: Callable[..., Any]
    backward_verified: bool
    grouped_query_verified: bool = True
    sliding_window_verified: bool = False
    block_q: int = 128
    block_k: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.torch_tpu_version, str) or not self.torch_tpu_version:
            raise ValueError("torch_tpu_version cannot be empty.")
        if (
            not isinstance(self.tested_torch_tpu_versions, tuple)
            or not self.tested_torch_tpu_versions
            or any(not isinstance(v, str) or not v for v in self.tested_torch_tpu_versions)
        ):
            raise ValueError("tested_torch_tpu_versions must contain versions.")
        if not callable(self.kernel):
            raise TypeError("kernel must be callable.")
        for name in ("backward_verified", "grouped_query_verified", "sliding_window_verified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        for name in ("block_q", "block_k"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 16:
                raise ValueError(f"{name} must be an integer >= 16.")

    def require_supported(self) -> None:
        if self.torch_tpu_version not in self.tested_torch_tpu_versions:
            raise RuntimeError(
                "TorchTPU Splash Attention is disabled for untested torch_tpu "
                f"version {self.torch_tpu_version!r}."
            )
        if not self.backward_verified:
            raise RuntimeError(
                "TorchTPU Splash Attention requires explicit backward correctness evidence."
            )


def torchtpu_splash_attention_provider(
    runtime: TorchTPUSplashAttentionRuntime,
    spec: CanonicalAttentionSpec,
    *,
    provider_id: str = "trainlm.torchtpu_splash",
) -> HFAttentionProvider:
    """Build a guarded HF attention provider backed by TorchTPU Splash Attention."""

    runtime.require_supported()
    if spec.layout != "mha" and not runtime.grouped_query_verified:
        raise RuntimeError(
            "Grouped-query Splash Attention requires grouped_query_verified=True."
        )
    if spec.mask.segment_ids:
        raise ValueError("TorchTPU Splash Attention does not support segment masks.")
    if (
        spec.mask.layout == "causal_sliding_window"
        and not runtime.sliding_window_verified
    ):
        raise RuntimeError(
            "Sliding-window Splash Attention requires sliding_window_verified=True."
        )
    if spec.position_encoding == "alibi":
        raise ValueError("TorchTPU Splash Attention does not support ALiBi slopes.")

    mapping = KVHeadMapping(spec.query_heads, spec.key_value_heads)

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
                "TorchTPU Splash Attention uses its registered causal mask."
            )
        if dropout != 0.0:
            raise ValueError("TorchTPU Splash Attention dropout is not implemented.")
        output = runtime.kernel(
            query,
            key,
            value,
            causal=True,
            scale=spec.effective_scale if scaling is None else scaling,
            query_heads=mapping.query_heads,
            key_value_heads=mapping.key_value_heads,
            sliding_window=spec.mask.sliding_window,
            block_q=runtime.block_q,
            block_k=runtime.block_k,
        )
        return output, None

    def causal_mask_factory(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    return HFAttentionProvider(
        provider_id=provider_id,
        attention_forward=attention_forward,
        mask_factory=causal_mask_factory,
        layouts=(spec.layout,),
        mask_layouts=(spec.mask.layout,),
        position_encodings=(spec.position_encoding,),
    )


@dataclass(frozen=True, slots=True)
class TorchTPUKernelInventory:
    """Discovered native TorchTPU kernel entrypoints in the active runtime."""

    torch_tpu_version: str | None
    splash_attention: Callable[..., Any] | None = None
    flash_attention: Callable[..., Any] | None = None
    rms_norm: Callable[..., Any] | None = None
    cross_entropy: Callable[..., Any] | None = None
    pallas_bridge: Callable[..., Any] | None = None

    @property
    def has_splash_attention(self) -> bool:
        return self.splash_attention is not None

    @property
    def has_flash_attention(self) -> bool:
        return self.flash_attention is not None

    @property
    def has_pallas_bridge(self) -> bool:
        return self.pallas_bridge is not None


def _find_callable_attr(containers: tuple[Any, ...], names: tuple[str, ...]) -> Callable[..., Any] | None:
    for container in containers:
        if container is None:
            continue
        for attr in names:
            candidate = getattr(container, attr, None)
            if callable(candidate):
                return candidate
    return None


def probe_torchtpu_kernels() -> TorchTPUKernelInventory:
    """Probe ``torch.tpu``, ``torch_tpu``, and Pallas bridge for native TPU kernels."""

    containers: list[Any] = []
    version: str | None = None
    try:
        import torch  # noqa: PLC0415

        tpu_mod = getattr(torch, "tpu", None)
        if tpu_mod is not None:
            containers.extend(
                [
                    tpu_mod,
                    getattr(tpu_mod, "ops", None),
                    getattr(tpu_mod, "kernels", None),
                    getattr(tpu_mod, "pallas", None),
                ]
            )
        ops_mod = getattr(torch, "ops", None)
        if ops_mod is not None:
            containers.extend(
                [
                    getattr(ops_mod, "tpu", None),
                    getattr(ops_mod, "torch_tpu", None),
                ]
            )
    except ImportError:
        pass

    try:
        import torch_tpu  # type: ignore[import-not-found]  # noqa: PLC0415

        version = str(getattr(torch_tpu, "__version__", "unknown"))
        containers.extend(
            [
                torch_tpu,
                getattr(torch_tpu, "ops", None),
                getattr(torch_tpu, "kernels", None),
                getattr(torch_tpu, "pallas", None),
            ]
        )
        internal_mod = getattr(torch_tpu, "_internal", None)
        if internal_mod is not None:
            containers.extend(
                [
                    getattr(internal_mod, "ops", None),
                    getattr(internal_mod, "kernels", None),
                    getattr(internal_mod, "pallas", None),
                ]
            )
    except ImportError:
        pass

    from trainlm.kernels.pallas import (
        is_pallas_splash_available,
        pallas_splash_attention,
        probe_pallas_bridge,
    )

    container_tuple = tuple(containers)
    splash = _find_callable_attr(
        container_tuple, ("splash_attention", "pallas_splash_attention")
    )
    if splash is None and is_pallas_splash_available():
        splash = pallas_splash_attention

    return TorchTPUKernelInventory(
        torch_tpu_version=version,
        splash_attention=splash,
        flash_attention=_find_callable_attr(
            container_tuple, ("flash_attention", "pallas_flash_attention", "tpu_flash_attention")
        ),
        rms_norm=_find_callable_attr(container_tuple, ("rms_norm", "fused_rms_norm")),
        cross_entropy=_find_callable_attr(
            container_tuple, ("cross_entropy", "fused_cross_entropy")
        ),
        pallas_bridge=probe_pallas_bridge(),
    )


def probe_torchtpu_splash_kernel() -> Callable[..., Any] | None:
    """Return an explicit Splash Attention callable if exposed by torch_tpu or Pallas, else None."""

    inv = probe_torchtpu_kernels()
    return inv.splash_attention or inv.flash_attention


__all__ = [
    "TorchTPUKernelInventory",
    "TorchTPUSplashAttentionRuntime",
    "probe_torchtpu_kernels",
    "probe_torchtpu_splash_kernel",
    "torchtpu_splash_attention_provider",
]
