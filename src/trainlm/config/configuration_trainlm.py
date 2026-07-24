from __future__ import annotations

from transformers import PretrainedConfig


class TrainLMConfig(PretrainedConfig):
    """
    Configuration class for TrainLM.

    The default configuration represents the official TrainLM v1
    reference architecture.
    """

    model_type = "trainlm"

    head_dim: int
    num_key_value_groups: int

    def __init__(
        self,
        # ------------------------------------------------------------------
        # Vocabulary & Embeddings
        # ------------------------------------------------------------------
        vocab_size: int = 32000,
        tie_word_embeddings: bool = True,
        # ------------------------------------------------------------------
        # Core Architecture
        # ------------------------------------------------------------------
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        # ------------------------------------------------------------------
        # Attention
        # ------------------------------------------------------------------
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # ------------------------------------------------------------------
        # Feed-Forward Network
        # ------------------------------------------------------------------
        intermediate_size: int = 3072,
        hidden_act: str = "silu",
        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        rms_norm_eps: float = 1e-6,
        # ------------------------------------------------------------------
        # Positional Encoding
        # ------------------------------------------------------------------
        rope_theta: float = 10000.0,
        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        initializer_range: float = 0.02,
        # ------------------------------------------------------------------
        # Sequence Limits
        # ------------------------------------------------------------------
        max_position_embeddings: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.rms_norm_eps = rms_norm_eps

        self.rope_theta = rope_theta

        self.initializer_range = initializer_range

        self.max_position_embeddings = max_position_embeddings

        self._validate()

        self.head_dim = (
            self.hidden_size //
            self.num_attention_heads
        )

        self.num_key_value_groups = (
            self.num_attention_heads //
            self.num_key_value_heads
        )

    def _validate(self) -> None:
        """Validate configuration values."""

        integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
        }

        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(
                    f"'{name}' must be greater than 0, got {value}."
                )

        if self.rms_norm_eps <= 0:
            raise ValueError(
                "'rms_norm_eps' must be greater than 0."
            )

        if self.rope_theta <= 0:
            raise ValueError(
                "'rope_theta' must be greater than 0."
            )

        if self.initializer_range <= 0:
            raise ValueError(
                "'initializer_range' must be greater than 0."
            )

        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(
                "'attention_dropout' must be between 0.0 and 1.0."
            )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "'hidden_size' must be divisible by "
                "'num_attention_heads'."
            )

        if (
            self.num_attention_heads
            % self.num_key_value_heads
            != 0
        ):
            raise ValueError(
                "'num_attention_heads' must be divisible by "
                "'num_key_value_heads'."
            )

        if not self.tie_word_embeddings:
            raise ValueError(
                "TrainLM v1 requires tied word embeddings."
            )

        if self.attention_bias:
            raise ValueError(
                "TrainLM v1 uses bias-free attention projections."
            )

        if self.hidden_act != "silu":
            raise ValueError(
                "TrainLM v1 requires 'hidden_act=\"silu\"'."
            )
