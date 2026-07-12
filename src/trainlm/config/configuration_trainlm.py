from transformers import PretrainedConfig


class TrainLMConfig(PretrainedConfig):
    """
    Configuration class for TrainLM.

    The default configuration represents the official TrainLM v1
    reference architecture. Architectural parameters will be added
    incrementally throughout Milestone M1.
    """

    model_type = "trainlm"
    def __init__(self, vocab_size: int = 32000, tie_word_embeddings: bool = True, **kwargs) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings ,**kwargs)

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be greater than 0, got {vocab_size}.")

        if not tie_word_embeddings:
            raise ValueError(f"TrainLM v1 requires 'tie_world_embeddings=True'.")

        self.vocab_size=vocab_size
