from dataclasses import asdict
import yaml

from trainlm.config.schema import (
    DataConfig,
    DistributedConfig,
    LlamaConfig,
    RuntimeConfig,
    TrainLMConfig,
    TrainingConfig,
)

def load_config(path: str) -> TrainLMConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = TrainLMConfig(
        model=LlamaConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        distributed=DistributedConfig(**raw.get("distributed", {})),
        data=DataConfig(**raw.get("data", {})),
    )

    config.validate()

    return config

def config_to_dict(config: TrainLMConfig) -> dict:
    return asdict(config)