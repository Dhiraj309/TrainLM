"""Family-neutral structural inspection for dense causal language models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from torch import nn

from .capabilities import CapabilityFact, ComponentCapability, ModelCapabilities


@dataclass(frozen=True, slots=True)
class StructuralInspectionEvidence:
    """Adapter-supplied evidence for semantics structure cannot prove alone."""

    model_class: str
    config_class: str
    source_provider: str
    normalization: ComponentCapability | None = None
    residual: ComponentCapability | None = None

    def __post_init__(self) -> None:
        for name in ("model_class", "config_class", "source_provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} evidence boundary cannot be empty.")
        for name in ("normalization", "residual"):
            capability = getattr(self, name)
            if capability is None:
                continue
            if not isinstance(capability, ComponentCapability):
                raise TypeError(f"{name} evidence must be a ComponentCapability.")
            if (
                capability.status not in {"known", "inferred", "unsupported"}
                or capability.kind is None
            ):
                raise ValueError(
                    f"{name} evidence must make an explicit semantic claim."
                )
            if not capability.evidence:
                raise ValueError(f"{name} evidence must cite its evidence source.")


def _fact(name: str, value: Any, source: str) -> CapabilityFact:
    return CapabilityFact(name=name, value=value, source=source)


def _unknown(note: str) -> ComponentCapability:
    return ComponentCapability.unknown(note)


def inspect_dense_causal_lm(
    model: nn.Module,
    *,
    source_provider: str = "external",
    structural_evidence: StructuralInspectionEvidence | None = None,
) -> ModelCapabilities:
    """Describe capabilities supported by direct config/module evidence.

    Model-family names are retained as metadata only and never drive semantic
    conclusions. Any capability without direct evidence remains ``unknown``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if structural_evidence is not None and not isinstance(
        structural_evidence, StructuralInspectionEvidence
    ):
        raise TypeError("structural_evidence must be StructuralInspectionEvidence.")
    config = getattr(model, "config", None)
    config_class = type(config).__name__ if config is not None else "unknown"
    if structural_evidence is not None:
        observed_boundary = (
            type(model).__name__,
            config_class,
            source_provider,
        )
        evidence_boundary = (
            structural_evidence.model_class,
            structural_evidence.config_class,
            structural_evidence.source_provider,
        )
        if evidence_boundary != observed_boundary:
            raise ValueError(
                "Structural evidence boundary does not match the inspected model, "
                "configuration, and source provider."
            )
    model_type = str(getattr(config, "model_type", "unknown") or "unknown")
    architectures = tuple(getattr(config, "architectures", ()) or ())
    warnings: list[str] = []

    heads = getattr(config, "num_attention_heads", None)
    kv_heads = getattr(config, "num_key_value_heads", None)
    if isinstance(heads, int) and heads > 0:
        if kv_heads is None:
            kv_heads = heads
        if isinstance(kv_heads, int) and 0 < kv_heads <= heads and heads % kv_heads == 0:
            kind = "mha" if kv_heads == heads else "mqa" if kv_heads == 1 else "gqa"
            attention = ComponentCapability(
                status="known",
                kind=kind,
                facts=(
                    _fact("heads", heads, "config.num_attention_heads"),
                    _fact("kv_heads", kv_heads, "config.num_key_value_heads"),
                ),
                evidence=("attention head-count configuration",),
            )
        else:
            attention = _unknown("Attention head geometry is inconsistent or incomplete.")
    else:
        attention = _unknown("No explicit attention head-count configuration.")

    rope_theta = getattr(config, "rope_theta", None)
    uses_alibi = getattr(config, "alibi", None)
    if isinstance(rope_theta, (int, float)):
        position = ComponentCapability(
            status="known",
            kind="rope",
            facts=(_fact("theta", float(rope_theta), "config.rope_theta"),),
            evidence=("explicit RoPE configuration",),
        )
    elif uses_alibi is True:
        position = ComponentCapability(
            status="known", kind="alibi", evidence=("config.alibi",)
        )
    else:
        position = _unknown("Position semantics are not explicit in public configuration.")

    layer_norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]
    normalization = (
        ComponentCapability(
            status="known",
            kind="layer_norm",
            facts=(_fact("count", len(layer_norms), "module traversal"),),
            evidence=("torch.nn.LayerNorm modules",),
        )
        if layer_norms
        else _unknown("No standard LayerNorm module; custom normalization is not guessed.")
    )
    if structural_evidence is not None and structural_evidence.normalization:
        normalization = structural_evidence.normalization

    activation = getattr(config, "hidden_act", None)
    intermediate = getattr(config, "intermediate_size", None)
    mlp = (
        ComponentCapability(
            status="inferred",
            kind="feed_forward",
            facts=tuple(
                fact
                for fact in (
                    _fact("activation", activation, "config.hidden_act") if isinstance(activation, str) else None,
                    _fact("intermediate_size", intermediate, "config.intermediate_size") if isinstance(intermediate, int) else None,
                )
                if fact is not None
            ),
            evidence=("public feed-forward configuration",),
        )
        if isinstance(activation, str) or isinstance(intermediate, int)
        else _unknown("No public feed-forward configuration.")
    )

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    projections = (
        ComponentCapability(
            status="known",
            kind="separate_qkv",
            evidence=("q_proj/k_proj/v_proj modules",),
        )
        if {"q_proj", "k_proj", "v_proj"} <= module_names
        else _unknown("Projection layout is not exposed as separate Q/K/V modules.")
    )

    input_embedding = getattr(model, "get_input_embeddings", lambda: None)()
    output_embedding = getattr(model, "get_output_embeddings", lambda: None)()
    embedding = (
        ComponentCapability(
            status="known",
            kind="token_embedding",
            facts=(_fact("vocab_size", input_embedding.num_embeddings, "input embedding"),),
            evidence=("get_input_embeddings()",),
        )
        if isinstance(input_embedding, nn.Embedding)
        else _unknown("Model does not expose a standard input embedding.")
    )
    lm_head = (
        ComponentCapability(
            status="known",
            kind="linear",
            facts=(_fact("tied", output_embedding.weight is input_embedding.weight, "parameter alias"),),
            evidence=("get_output_embeddings()",),
        )
        if isinstance(output_embedding, nn.Linear) and input_embedding is not None
        else _unknown("Model does not expose a standard linear output embedding.")
    )

    try:
        parameters = inspect.signature(model.forward).parameters
        forward_fields = tuple(name for name in ("labels", "attention_mask", "position_ids", "use_cache") if name in parameters)
    except (TypeError, ValueError):
        forward_fields = ()
        warnings.append("Forward signature could not be inspected.")
    checkpointing = ComponentCapability(
        status="known",
        kind="gradient_checkpointing",
        facts=(
            _fact("supported", bool(getattr(model, "supports_gradient_checkpointing", False)), "model attribute"),
            _fact("forward_fields", ",".join(forward_fields), "forward signature"),
        ),
        evidence=("model attribute and forward signature",),
    )

    residual = _unknown(
        "Residual topology requires an explicit adapter or graph evidence."
    )
    if structural_evidence is not None and structural_evidence.residual:
        residual = structural_evidence.residual

    return ModelCapabilities(
        schema_version=1,
        model_type=model_type,
        model_class=type(model).__name__,
        config_class=config_class,
        source_provider=source_provider,
        architectures=architectures,
        attention=attention,
        position=position,
        normalization=normalization,
        mlp=mlp,
        residual=residual,
        projections=projections,
        embedding=embedding,
        lm_head=lm_head,
        checkpointing=checkpointing,
        warnings=tuple(warnings),
    )


__all__ = ["StructuralInspectionEvidence", "inspect_dense_causal_lm"]
