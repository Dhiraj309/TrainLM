from transformers import PretrainedConfig


class TrainLMConfig(PretrainedConfig):
    """
    Configuration class for TrainLM.

    The default configuration represents the official TrainLM v1
    reference architecture. Architectural parameters will be added
    incrementally throughout Milestone M1.
    """

    model_type = "trainlm"
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
