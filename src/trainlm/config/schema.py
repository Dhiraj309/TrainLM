from dataclasses import dataclass, field
from trainlm.model.config import LlamaConfig

@dataclass(slots=True)
class TrainingConfig:
    micro_batach_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000

@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 42
    compute_dtype: str = "bfloat16"
    param_dtype: str = "bfloat16"
    log_every: int = 10
    eval_every: int = 1000
    save_every: int = 1000

@dataclass(slots=True)
class DistributedConfig:
    mesh_axis_names: tuple[str, ...] = (
        "data",
    )

    mesh_shape: tuple[int, ...] = (1,)

@dataclass(slots=True)
class DataConfig:
    dataset: str = ""
    sequence_lenght: int = 2048
    streaming: bool = True
    packing: bool = True


@dataclass(slots=True)
class TrainLMConfig:
    model: LlamaConfig = field(
        default_factory=LlamaConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig,
    )

    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig,
    )

    distributed: DistributedConfig = field(
        default_factory=DistributedConfig,
    )

    data: DataConfig = field(
        default_factory=DataConfig,
    )

    def validate(self) -> None:
        self.model.validate()