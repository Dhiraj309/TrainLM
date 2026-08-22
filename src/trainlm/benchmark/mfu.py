"""Model FLOP and utilization calculations for dense causal LMs."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return value


@dataclass(frozen=True, slots=True)
class CausalLMFlopBreakdown:
    """Training FLOPs per supervised token for a dense decoder-only LM."""

    non_embedding_flops_per_token: float
    output_projection_flops_per_token: float

    def __post_init__(self) -> None:
        _positive(
            "non_embedding_flops_per_token",
            self.non_embedding_flops_per_token,
        )
        if (
            not math.isfinite(self.output_projection_flops_per_token)
            or self.output_projection_flops_per_token < 0
        ):
            raise ValueError(
                "output_projection_flops_per_token must be finite and "
                "non-negative."
            )

    @property
    def logits_inclusive_flops_per_token(self) -> float:
        return (
            self.non_embedding_flops_per_token
            + self.output_projection_flops_per_token
        )


def calculate_causal_lm_flops(
    *,
    parameter_count: int,
    vocabulary_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    sequence_length: int,
    tie_word_embeddings: bool,
) -> CausalLMFlopBreakdown:
    """Calculate dense causal-LM training FLOPs per supervised token.

    The non-embedding estimate uses the common ``6 * parameters`` training
    approximation and adds the quadratic attention work that is not represented
    by parameter count. Input embeddings are excluded from that parameter term.
    The logits-inclusive value adds output projection work even when its weight
    is tied to the input embedding, because tying removes parameters but not the
    projection computation.
    """

    if not isinstance(tie_word_embeddings, bool):
        raise ValueError("tie_word_embeddings must be a boolean.")

    integer_inputs = {
        "parameter_count": parameter_count,
        "vocabulary_size": vocabulary_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "sequence_length": sequence_length,
    }
    for name, value in integer_inputs.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    embedding_parameters = vocabulary_size * hidden_size
    output_head_parameters = 0 if tie_word_embeddings else embedding_parameters
    non_embedding_parameters = (
        parameter_count - embedding_parameters - output_head_parameters
    )
    if non_embedding_parameters <= 0:
        raise ValueError(
            "parameter_count must exceed embedding and untied output-head "
            "parameters."
        )

    parameter_flops = 6 * non_embedding_parameters
    attention_flops = (
        12 * num_hidden_layers * sequence_length * hidden_size
    )
    output_projection_flops = 6 * embedding_parameters

    return CausalLMFlopBreakdown(
        non_embedding_flops_per_token=float(
            parameter_flops + attention_flops
        ),
        output_projection_flops_per_token=float(output_projection_flops),
    )


def calculate_mfu(
    *,
    tokens_per_second: float,
    flops_per_token: float,
    peak_flops_per_device: float,
    device_count: int,
) -> float:
    """Return achieved model FLOPs divided by aggregate theoretical peak."""

    _positive("tokens_per_second", tokens_per_second)
    _positive("flops_per_token", flops_per_token)
    _positive("peak_flops_per_device", peak_flops_per_device)
    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise ValueError("device_count must be a positive integer.")
    if device_count <= 0:
        raise ValueError("device_count must be a positive integer.")

    aggregate_peak = peak_flops_per_device * device_count
    return tokens_per_second * flops_per_token / aggregate_peak
