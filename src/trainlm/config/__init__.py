"""
Configuration package for TrainLM.

This package contains the model configuration used by Hugging Face
(`TrainLMConfig`) together with the training configuration objects used
by the TrainLM training framework.

Structure
---------
configuration_trainlm.py
    Hugging Face model configuration.

train.py
    Root training configuration.

dataset.py
    Dataset configuration.

runtime.py
    Runtime and execution configuration.

optimizer.py
    Optimizer configuration.

scheduler.py
    Learning-rate scheduler configuration.

trainer.py
    Trainer configuration.

checkpoint.py
    Checkpoint configuration.

logging.py
    Logging configuration.

evaluation.py
    Evaluation configuration.

parallel.py
    Parallelism configuration.
"""

from .configuration_trainlm import TrainLMConfig

__all__ = [
    "TrainLMConfig",
]
