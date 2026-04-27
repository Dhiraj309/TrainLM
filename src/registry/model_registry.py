from typing import Type, Dict

# Hugging Face Flax model classes
from transformers import (
    FlaxLlamaForCausalLM,
    FlaxGPT2LMHeadModel,
    FlaxOPTForCausalLM,
)


# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------

# Maps model_type → Flax model class
_MODEL_REGISTRY: Dict[str, Type] = {
    "llama": FlaxLlamaForCausalLM,
    "gpt2": FlaxGPT2LMHeadModel,
    "opt": FlaxOPTForCausalLM,
}


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_model_class(model_type: str) -> Type:
    """
    Retrieve the Flax model class for a given model_type.

    Parameters
    ----------
    model_type : str
        Model identifier (e.g., "llama", "gpt2", "opt")

    Returns
    -------
    Type
        Flax model class

    Raises
    ------
    ValueError
        If model_type is not supported
    """

    if model_type not in _MODEL_REGISTRY:
        supported = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported models: {supported}"
        )

    return _MODEL_REGISTRY[model_type]


def list_supported_models() -> Dict[str, Type]:
    """
    Returns all supported model mappings.
    """

    return _MODEL_REGISTRY.copy()
