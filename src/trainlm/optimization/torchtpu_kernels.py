"""Reversible Tier-2 kernel and projection packing plan for dense HF models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .attention import AttentionMaskSpec, CanonicalAttentionSpec
from .capabilities import ModelCapabilities
from .hf_attention import HFAttentionInstallation, install_hf_attention_provider
from .inspection import inspect_dense_causal_lm
from .mlp import GatedMLPProjectionSpec, PackedGatedMLPProjection
from .torchtpu_attention import (
    TorchTPUSplashAttentionRuntime,
    torchtpu_splash_attention_provider,
)


class PackedGatedMLPBlock(nn.Module):
    """Drop-in replacement for HF gate_proj/up_proj/down_proj MLP blocks."""

    def __init__(
        self,
        packed_gate_up: PackedGatedMLPProjection,
        down_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.packed_gate_up = packed_gate_up
        self.down_proj = down_proj

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(self.packed_gate_up(hidden_states))


@dataclass(frozen=True, slots=True)
class TorchTPUKernelPlanResult:
    """Record applied Tier-2 transformations and support exact rollback."""

    attention_installation: HFAttentionInstallation | None
    previous_attn_implementation: str | None
    packed_mlp_paths: tuple[str, ...]
    _mlp_snapshots: tuple[tuple[str, nn.Module], ...]

    def rollback(self, model: nn.Module) -> None:
        target = getattr(model, "_orig_mod", model)
        for path, original in reversed(self._mlp_snapshots):
            current = _get_submodule(target, path)
            if isinstance(current, PackedGatedMLPBlock):
                packed = current.packed_gate_up
                mid = packed.spec.intermediate_size
                with torch.no_grad():
                    original.gate_proj.weight.data.copy_(
                        packed.weight[:mid].to(
                            device=original.gate_proj.weight.device,
                            dtype=original.gate_proj.weight.dtype,
                        )
                    )
                    original.up_proj.weight.data.copy_(
                        packed.weight[mid:].to(
                            device=original.up_proj.weight.device,
                            dtype=original.up_proj.weight.dtype,
                        )
                    )
                    if (
                        packed.bias is not None
                        and original.gate_proj.bias is not None
                        and original.up_proj.bias is not None
                    ):
                        original.gate_proj.bias.data.copy_(
                            packed.bias[:mid].to(
                                device=original.gate_proj.bias.device,
                                dtype=original.gate_proj.bias.dtype,
                            )
                        )
                        original.up_proj.bias.data.copy_(
                            packed.bias[mid:].to(
                                device=original.up_proj.bias.device,
                                dtype=original.up_proj.bias.dtype,
                            )
                        )
            _set_submodule(target, path, original)
        if self.attention_installation is not None or self.previous_attn_implementation is not None:
            setter = getattr(target, "set_attn_implementation", None)
            if callable(setter) and self.previous_attn_implementation is not None:
                setter(self.previous_attn_implementation)
            elif (
                hasattr(target, "config")
                and hasattr(target.config, "_attn_implementation")
            ):
                target.config._attn_implementation = self.previous_attn_implementation


def apply_torchtpu_kernel_plan(
    model: nn.Module,
    *,
    capabilities: ModelCapabilities | None = None,
    splash_runtime: TorchTPUSplashAttentionRuntime | None = None,
    pack_mlp: bool = True,
    attention_interface: Any | None = None,
    mask_interface: Any | None = None,
) -> TorchTPUKernelPlanResult:
    """Apply capability-verified Splash Attention and SwiGLU/GeGLU packing."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    caps = capabilities if capabilities is not None else inspect_dense_causal_lm(model)

    previous_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    attention_installation: HFAttentionInstallation | None = None
    if splash_runtime is not None:
        spec = _canonical_attention_spec(model, caps)
        provider = torchtpu_splash_attention_provider(splash_runtime, spec)
        attention_installation = install_hf_attention_provider(
            model,
            spec,
            provider,
            attention_interface=attention_interface,
            mask_interface=mask_interface,
        )

    packed_paths: list[str] = []
    snapshots: list[tuple[str, nn.Module]] = []
    if (
        pack_mlp
        and caps.mlp.status in {"known", "inferred"}
        and caps.mlp.kind in {"feed_forward", "swiglu", "geglu"}
    ):
        activation = _resolve_gated_activation(caps, model)
        for path, module in list(model.named_modules()):
            if not path:
                continue
            if (
                isinstance(getattr(module, "gate_proj", None), nn.Linear)
                and isinstance(getattr(module, "up_proj", None), nn.Linear)
                and isinstance(getattr(module, "down_proj", None), nn.Module)
                and callable(getattr(module, "act_fn", None))
            ):
                gate: nn.Linear = module.gate_proj
                up: nn.Linear = module.up_proj
                if (
                    gate.in_features != up.in_features
                    or gate.out_features != up.out_features
                    or (gate.bias is not None) != (up.bias is not None)
                ):
                    continue
                spec = GatedMLPProjectionSpec(
                    prefix=path,
                    activation=activation,
                    input_size=gate.in_features,
                    intermediate_size=gate.out_features,
                    gate_weight_key=f"{path}.gate_proj.weight",
                    up_weight_key=f"{path}.up_proj.weight",
                    packed_weight_key=f"{path}.packed_gate_up.weight",
                    gate_bias_key=f"{path}.gate_proj.bias" if gate.bias is not None else None,
                    up_bias_key=f"{path}.up_proj.bias" if up.bias is not None else None,
                    packed_bias_key=f"{path}.packed_gate_up.bias" if gate.bias is not None else None,
                )
                packed_proj = PackedGatedMLPProjection.from_separate(
                    spec, gate, up, module.act_fn
                )
                snapshots.append((path, module))
                _set_submodule(model, path, PackedGatedMLPBlock(packed_proj, module.down_proj))
                packed_paths.append(path)

    return TorchTPUKernelPlanResult(
        attention_installation=attention_installation,
        previous_attn_implementation=previous_attn,
        packed_mlp_paths=tuple(packed_paths),
        _mlp_snapshots=tuple(snapshots),
    )


def _canonical_attention_spec(
    model: nn.Module,
    caps: ModelCapabilities,
) -> CanonicalAttentionSpec:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Splash Attention requires model.config geometry.")
    query_heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads", query_heads))
    hidden_size = int(getattr(config, "hidden_size"))
    head_dim = int(getattr(config, "head_dim", hidden_size // query_heads))
    layout = "mha" if kv_heads == query_heads else ("mqa" if kv_heads == 1 else "gqa")
    pos_kind = caps.position.kind if caps.position.kind in {"learned", "rope", "alibi", "none"} else "rope"
    return CanonicalAttentionSpec(
        schema_version=1,
        layout=layout,
        query_heads=query_heads,
        key_value_heads=kv_heads,
        head_dim=head_dim,
        position_encoding=pos_kind,
        mask=AttentionMaskSpec(layout="causal"),
    )


def _resolve_gated_activation(caps: ModelCapabilities, model: nn.Module) -> str:
    if caps.mlp.kind in {"swiglu", "geglu"}:
        return caps.mlp.kind
    hidden_act = getattr(getattr(model, "config", None), "hidden_act", "")
    if isinstance(hidden_act, str) and "gelu" in hidden_act.lower():
        return "geglu"
    return "swiglu"


def _get_submodule(model: nn.Module, path: str) -> Any:
    current: Any = model
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _set_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent: Any = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


__all__ = [
    "PackedGatedMLPBlock",
    "TorchTPUKernelPlanResult",
    "apply_torchtpu_kernel_plan",
]
