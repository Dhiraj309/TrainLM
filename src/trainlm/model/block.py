"""Conservative transformer-block layout detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

NormKind = Literal["layernorm", "rmsnorm", "unknown"]
MlpKind = Literal["gelu", "swiglu", "geglu", "unknown"]
ResidualKind = Literal["serial", "parallel", "unknown"]


@dataclass(frozen=True, slots=True)
class BlockLayout:
    """Normalization, activation/MLP, and residual layout evidence."""

    normalization: NormKind
    mlp: MlpKind
    residual: ResidualKind
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.normalization not in {"layernorm", "rmsnorm", "unknown"}:
            raise ValueError(f"Unsupported normalization kind: {self.normalization}")
        if self.mlp not in {"gelu", "swiglu", "geglu", "unknown"}:
            raise ValueError(f"Unsupported MLP kind: {self.mlp}")
        if self.residual not in {"serial", "parallel", "unknown"}:
            raise ValueError(f"Unsupported residual kind: {self.residual}")
        if not self.evidence or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError("Block layout requires non-empty evidence.")


def _explicit_string(config: Any, names: tuple[str, ...]) -> list[tuple[str, str]]:
    values = []
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, str) and value.strip():
            values.append((name, value.strip().lower()))
    return values


def _module_names(model: Any | None) -> tuple[str, ...]:
    if model is None:
        return ()
    modules = getattr(model, "modules", None)
    names: list[str] = []
    if callable(modules):
        names.extend(type(module).__name__.lower() for module in modules())
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        names.extend(name.lower() for name, _ in named_modules() if name)
    return tuple(names)


def detect_block_layout(
    config: Any,
    model: Any | None = None,
) -> BlockLayout:
    """Classify common dense block layouts without family-specific branches."""

    if config is None:
        raise TypeError("config is required.")

    evidence: list[str] = []
    norm_values = _explicit_string(
        config, ("normalization", "norm_type", "norm", "normalization_type")
    )
    has_rms_flag = any(
        getattr(config, name, None) is True
        for name in ("rms_norm", "use_rms_norm", "rmsnorm")
    )
    has_layer_flag = any(
        getattr(config, name, None) is True
        for name in ("layer_norm", "use_layer_norm", "layernorm")
    )
    has_rms_epsilon = any(
        getattr(config, name, None) is not None
        for name in ("rms_norm_eps", "rms_norm_epsilon")
    )
    has_layer_epsilon = any(
        getattr(config, name, None) is not None
        for name in ("layer_norm_eps", "layer_norm_epsilon")
    )
    has_rms_flag = has_rms_flag or has_rms_epsilon
    has_layer_flag = has_layer_flag or has_layer_epsilon
    rms = has_rms_flag or any(
        value in {"rmsnorm", "rms_norm", "rms"} for _, value in norm_values
    )
    layer = has_layer_flag or any(
        value in {"layernorm", "layer_norm", "layer"} for _, value in norm_values
    )
    norm: NormKind = "rmsnorm" if rms and not layer else (
        "layernorm" if layer and not rms else "unknown"
    )
    evidence.extend(f"config.{name}={value}" for name, value in norm_values)
    if has_rms_flag:
        evidence.append("explicit RMSNorm flag or epsilon")
    if has_layer_flag:
        evidence.append("explicit LayerNorm flag or epsilon")

    mlp_values = _explicit_string(
        config, ("mlp_type", "ffn_type", "activation_function", "hidden_act")
    )
    mlp_kinds = set()
    for _, value in mlp_values:
        if value in {"gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_fast"}:
            mlp_kinds.add("gelu")
        elif value in {"swiglu", "swi_glu"}:
            mlp_kinds.add("swiglu")
        elif value in {"geglu", "ge_glu"}:
            mlp_kinds.add("geglu")
    if getattr(config, "gated_mlp", None) is True:
        mlp_kinds.add("swiglu")
    mlp: MlpKind = next(iter(mlp_kinds)) if len(mlp_kinds) == 1 else "unknown"
    evidence.extend(f"config.{name}={value}" for name, value in mlp_values)

    residual_values = _explicit_string(
        config, ("residual_type", "residual_connection", "block_type")
    )
    parallel_flags = any(
        getattr(config, name, None) is True
        for name in ("parallel_block", "parallel_residual", "use_parallel_residual")
    )
    serial_flags = any(
        getattr(config, name, None) is True
        for name in ("serial_block", "serial_residual", "use_serial_residual")
    )
    parallel = parallel_flags or any(
        value in {"parallel", "parallel_residual"}
        for _, value in residual_values
    )
    serial = serial_flags or any(
        value in {"serial", "serial_residual"} for _, value in residual_values
    )
    residual: ResidualKind = "parallel" if parallel and not serial else (
        "serial" if serial and not parallel else "unknown"
    )
    evidence.extend(f"config.{name}={value}" for name, value in residual_values)

    names = _module_names(model)
    if names:
        if norm == "unknown":
            has_rms_module = any(
                "rmsnorm" in name or "rms_norm" in name for name in names
            )
            has_layer_module = any(
                "layernorm" in name or "layer_norm" in name for name in names
            )
            if has_rms_module != has_layer_module:
                norm = "rmsnorm" if has_rms_module else "layernorm"
        if mlp == "unknown":
            has_geglu = any("geglu" in name for name in names)
            has_swiglu = any(
                "swiglu" in name or "swi_glu" in name for name in names
            ) or ("gate_proj" in " ".join(names) and "up_proj" in " ".join(names))
            has_gelu = any("gelu" in name for name in names)
            kinds = [
                kind
                for kind, found in (
                    ("geglu", has_geglu),
                    ("swiglu", has_swiglu),
                    ("gelu", has_gelu),
                )
                if found
            ]
            if len(kinds) == 1:
                mlp = kinds[0]  # type: ignore[assignment]
        if residual == "unknown" and any(
            "parallelresidual" in name for name in names
        ):
            residual = "parallel"

    if not evidence:
        evidence.append(
            "layout inferred from model module structure"
            if names
            else "no explicit block-layout declaration"
        )
    if norm == "unknown":
        evidence.append("normalization layout is unknown or contradictory")
    if mlp == "unknown":
        evidence.append("MLP activation/layout is unknown or contradictory")
    if residual == "unknown":
        evidence.append("residual layout is unknown or contradictory")
    return BlockLayout(norm, mlp, residual, tuple(evidence))
