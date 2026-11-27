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
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
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
        mlp_bias: bool = False,
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
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        #
        # Vocabulary
        #
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        #
        # Architecture
        #
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        #
        # Attention
        #
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        #
        # Feed-forward
        #
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias

        #
        # Normalization
        #
        self.rms_norm_eps = rms_norm_eps

        #
        # RoPE
        #
        self.rope_theta = rope_theta

        #
        # Initialization
        #
        self.initializer_range = initializer_range

        #
        # Sequence length
        #
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

        if self.mlp_bias:
            raise ValueError(
                "TrainLM v1 uses bias-free feed-forward projections."
            )

        if self.hidden_act != "silu":
            raise ValueError(
                "TrainLM v1 requires 'hidden_act=\"silu\"'."
            )


class LaughLMLoopConfig(TrainLMConfig):
    """Configuration for looped-depth Prelude + Recurrent Core + Coda Causal LM."""

    model_type = "trainlm_loop"

    def __init__(
        self,
        *,
        num_prelude_layers: int = 2,
        num_core_layers: int = 4,
        num_coda_layers: int = 2,
        num_loop_iterations: int = 4,
        train_min_loops: int = 2,
        train_max_loops: int = 4,
        truncated_bptt_steps: int = 4,
        use_pallas_recurrent_norm: bool = True,
        **kwargs,
    ):
        self.num_prelude_layers = int(num_prelude_layers)
        self.num_core_layers = int(num_core_layers)
        self.num_coda_layers = int(num_coda_layers)
        self.num_loop_iterations = int(num_loop_iterations)
        self.train_min_loops = int(train_min_loops)
        self.train_max_loops = int(train_max_loops)
        self.truncated_bptt_steps = int(truncated_bptt_steps)
        if not isinstance(use_pallas_recurrent_norm, bool):
            raise ValueError("'use_pallas_recurrent_norm' must be a boolean.")
        self.use_pallas_recurrent_norm = use_pallas_recurrent_norm
        kwargs.setdefault(
            "num_hidden_layers",
            self.num_prelude_layers + self.num_core_layers + self.num_coda_layers,
        )
        super().__init__(**kwargs)

    def _validate(self) -> None:
        super()._validate()
        if self.num_prelude_layers < 0:
            raise ValueError("'num_prelude_layers' must be >= 0.")
        if self.num_core_layers <= 0:
            raise ValueError("'num_core_layers' must be > 0.")
        if self.num_coda_layers < 0:
            raise ValueError("'num_coda_layers' must be >= 0.")
        if self.num_loop_iterations <= 0:
            raise ValueError("'num_loop_iterations' must be > 0.")
        if self.train_min_loops <= 0:
            raise ValueError("'train_min_loops' must be > 0.")
        if self.train_max_loops < self.train_min_loops:
            raise ValueError("'train_max_loops' must be >= 'train_min_loops'.")
        if self.truncated_bptt_steps <= 0:
            raise ValueError("'truncated_bptt_steps' must be > 0.")

    def effective_num_layers(self, active_loops: int | None = None) -> int:
        """Return total executed transformer layers for ``active_loops`` recurrent passes."""
        loops = self.num_loop_iterations if active_loops is None else int(active_loops)
        return self.num_prelude_layers + loops * self.num_core_layers + self.num_coda_layers

    @classmethod
    def laughlm_loop_135m(cls, **overrides) -> "LaughLMLoopConfig":
        """111M stored / 167M non-embedding (211M total) effective-active preset for TPU v5e-1.

        All projection, intermediate, vocab, and head dimensions are multiples of
        128 (``hidden_size=896``, ``intermediate_size=2432``, ``head_dim=128``,
        ``vocab_size=49152``) to saturate v5e-1 MXUs and activate unpadded
        Pallas Splash Attention.
        """
        defaults = {
            "vocab_size": 49152,
            "hidden_size": 896,
            "intermediate_size": 2432,
            "num_attention_heads": 7,
            "num_key_value_heads": 1,
            "num_prelude_layers": 2,
            "num_core_layers": 4,
            "num_coda_layers": 2,
            "num_loop_iterations": 4,
            "train_min_loops": 2,
            "train_max_loops": 4,
            "truncated_bptt_steps": 4,
            "use_pallas_recurrent_norm": True,
            "max_position_embeddings": 2048,
            "rope_theta": 100000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
            "use_cache": False,
        }
        defaults.update(overrides)
        return cls(**defaults)


def _register_laughlm_loop_config() -> None:
    try:
        from transformers import AutoConfig

        AutoConfig.register(LaughLMLoopConfig.model_type, LaughLMLoopConfig, exist_ok=True)
    except Exception:
        pass


_register_laughlm_loop_config()
