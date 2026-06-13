from dataclasses import dataclass, field

@dataclass(slots=True)
class LlamaConfig:
    #
    # Vocabulary
    #
    vocab_size: int = 32000

    #
    # Hidden Dimensions
    #

    hidden_size: int = 4096
    intermediate_size: int = 110008

    #
    # Transformer Depth
    #

    num_hidden_layers: int = 32

    #
    # Attention
    #

    num_attention_heads: int = 32

    num_key_value_heads: int = 32

    attention_bias: bool = False

    attention_dropout: float = 0.0

    #
    # MLP
    #

    hidden_act: str = "silu"
    mlp_bias: bool = False

    #
    # Positional Embeddings
    #

    max_position_embeddings: int = 2048
    rope_parameters: dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000.0,
        }
    )
    #
    # Normalization
    #

    rms_norm_eps: float = 1e-6

    #
    # Initialization
    #

    initializer_range: float = 0.02

    #
    # Embeddings
    #

    tie_word_embeddings: bool = False

    #
    # Derived
    #

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if (
            self.num_attention_heads % self.num_key_value_heads != 0
        ):

            raise ValueError(
                "num_attention_heads must be divisible" "by num_key_value_heads"
            )

        if self.hidden_act != "silu":
            raise ValueError(
                "TrainLM currently supports SwiGLU only" "(hidden_act='silu')"
            )

        rope_type = self.rope_parameters.get(
            "rope_type",
            "default",
        )

        if rope_type != "default":
            raise ValueError(
                "TrainLM currently supports "
                "only defualt RoPE"
            )