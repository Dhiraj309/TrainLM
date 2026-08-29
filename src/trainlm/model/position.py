"""Conservative positional-semantics detection for dense causal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PositionKind = Literal["learned", "rope", "alibi", "unknown"]


@dataclass(frozen=True, slots=True)
class PositionSemantics:
    """Position encoding classification with explicit evidence."""

    kind: PositionKind
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"learned", "rope", "alibi", "unknown"}:
            raise ValueError(f"Unsupported position kind: {self.kind}")
        if not self.evidence or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError("Position semantics require non-empty evidence.")


def detect_position_semantics(
    config: Any,
    model: Any | None = None,
) -> PositionSemantics:
    """Classify positions from explicit config/module evidence only.

    A bare ``max_position_embeddings`` field is intentionally insufficient:
    both learned-position and RoPE models commonly expose it. Ambiguous
    configurations therefore remain ``unknown`` instead of receiving a
    family-name guess.
    """

    if config is None:
        raise TypeError("config is required.")
    values = {
        name: getattr(config, name, None)
        for name in (
            "position_embedding_type",
            "position_encoding",
            "position_embedding",
            "use_alibi",
            "alibi",
            "alibi_bias",
            "rope_theta",
            "rope_parameters",
            "rope_scaling",
        )
    }
    explicit = []
    for name in ("position_embedding_type", "position_encoding", "position_embedding"):
        value = values[name]
        if isinstance(value, str) and value.strip():
            explicit.append((name, value.strip().lower()))

    alibi = any(
        value in {"alibi", "linear_bias", "alibi_bias"}
        for _, value in explicit
    ) or any(values[name] is True for name in ("use_alibi", "alibi", "alibi_bias"))
    rope = any(value in {"rope", "rotary", "rotary_embedding"} for _, value in explicit)
    rope = rope or values["rope_parameters"] is not None
    rope = rope or values["rope_scaling"] is not None
    rope = rope or values["rope_theta"] is not None
    learned = any(
        value in {"learned", "absolute", "absolute_learned"}
        for _, value in explicit
    )

    evidence = [
        f"config.{name}={value}"
        for name, value in explicit
    ]
    if alibi and not rope and not learned:
        return PositionSemantics("alibi", tuple(evidence) or ("explicit ALiBi flag",))
    if rope and not alibi and not learned:
        return PositionSemantics("rope", tuple(evidence) or ("explicit RoPE fields",))
    if learned and not alibi and not rope:
        return PositionSemantics(
            "learned", tuple(evidence) or ("explicit learned-position field",)
        )

    # Module inspection is only a fallback.  Some wrappers expose no
    # ``modules`` method, and contradictory config evidence must stay unknown.
    if model is not None and not (alibi or rope or learned):
        modules = getattr(model, "modules", None)
        if callable(modules):
            module_names = tuple(
                type(module).__name__.lower() for module in modules()
            )
            if any("alibi" in name for name in module_names):
                return PositionSemantics(
                    "alibi", ("model module exposes ALiBi semantics",)
                )
            if any("rotary" in name or "rope" in name for name in module_names):
                return PositionSemantics(
                    "rope", ("model module exposes rotary semantics",)
                )

    if not evidence:
        evidence.append("no explicit positional encoding declaration")
    if alibi or rope or learned:
        evidence.append("positional indicators are contradictory or ambiguous")
    return PositionSemantics("unknown", tuple(evidence))
