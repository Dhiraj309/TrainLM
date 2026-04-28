from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class ModelConfig(BaseModel):
    """
    HF-aligned model configuration for supported Flax families.
    """

    model_type: Literal["llama", "gpt2", "opt"]

    hidden_size: int = Field(..., gt=0)
    num_hidden_layers: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    intermediate_size: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    max_position_embeddings: int = Field(..., gt=0)

    num_key_value_heads: Optional[int] = Field(default=None, gt=0)
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = True

    @model_validator(mode="after")
    def _validate_model(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={self.hidden_size}, "
                f"num_attention_heads={self.num_attention_heads})."
            )

        if self.num_key_value_heads is not None:
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ValueError(
                    "num_attention_heads must be divisible by num_key_value_heads "
                    f"(got num_attention_heads={self.num_attention_heads}, "
                    f"num_key_value_heads={self.num_key_value_heads})."
                )

            if self.num_key_value_heads > self.num_attention_heads:
                raise ValueError(
                    "num_key_value_heads must be <= num_attention_heads."
                )

        return self


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

class OptimizerConfig(BaseModel):
    """
    Optimizer hyperparameters.
    """

    type: Literal["adamw", "adafactor", "lion", "muon"] = "adamw"

    learning_rate: float = Field(..., gt=0)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    weight_decay: float = 0.1
    grad_clip: float = 1.0


# ------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------

class SchedulerConfig(BaseModel):
    """
    Learning-rate schedule configuration.
    """

    type: Literal["cosine", "linear", "rsqrt", "wsd"] = "cosine"

    warmup_steps: Optional[int] = Field(default=None, ge=0)
    warmup_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    min_lr_ratio: float = Field(default=0.1, ge=0.0)
    stable_fraction: float = Field(default=0.88, gt=0.0, lt=1.0)
    decay_steps: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_scheduler(self):
        if self.type == "wsd":
            if self.warmup_steps is None and self.warmup_fraction is None:
                raise ValueError(
                    "WSD scheduler requires either warmup_steps or warmup_fraction."
                )

        return self


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

class RuntimeConfig(BaseModel):
    """
    Runtime training parameters.
    """

    seq_len: int = Field(..., gt=0)
    micro_batch_per_device: int = Field(..., gt=0)
    gradient_accumulation: int = Field(..., gt=0)

    total_tokens: int = Field(..., gt=0)

    log_interval: int = Field(default=10, gt=0)
    eval_interval: int = Field(default=500, gt=0)

    checkpoint_interval: int = Field(default=1000, gt=0)
    checkpoint_max_to_keep: int = Field(default=3, gt=0)
    checkpoint_dir: str = "checkpoints"


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

class DataConfig(BaseModel):
    """
    Tokenized dataset source config.
    """

    sources: List[str] = Field(default_factory=list)
    packing: bool = True
    eos_between_docs: bool = True
    pad_to_multiple: int = Field(default=1, gt=0)


# ------------------------------------------------------------
# Parallelism
# ------------------------------------------------------------

class ParallelismConfig(BaseModel):
    """
    Runtime precision and parallelism settings.
    """

    strategy: Literal["single", "pmap"] = "pmap"
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    param_dtype: Literal["float32", "bfloat16"] = "float32"


# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

class HardwareConfig(BaseModel):
    """
    Accelerator target.
    """

    accelerator: Literal["tpu", "gpu", "cpu"] = "tpu"


# ------------------------------------------------------------
# Monitoring
# ------------------------------------------------------------

class MonitoringConfig(BaseModel):
    """
    Logging / visualization toggles.
    """

    tensorboard: bool = False
    rich_terminal: bool = True


# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

class TrainConfig(BaseModel):
    """
    Full experiment configuration.
    """

    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig
    data: DataConfig
    parallelism: ParallelismConfig
    hardware: HardwareConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
