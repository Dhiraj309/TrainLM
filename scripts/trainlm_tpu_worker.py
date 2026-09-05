"""Launch TrainLM on all TPU cores through the PyTorch/XLA PJRT launcher.

The notebook process is a coordinator only. Model, data, optimizer, and Trainer
objects are constructed inside ``train_fn`` so every PJRT worker owns its local
XLA device and deterministic data partition.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace


def _configure_environment() -> None:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for name in (
        "XRT_TPU_CONFIG",
        "TPU_PROCESS_ADDRESSES",
        "JAX_TPU_PROCESS_ADDRESSES",
        "CLOUD_TPU_TASK_ID",
        "TPU_PROCESS_BOUNDS",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_VISIBLE_CHIPS",
    ):
        os.environ.pop(name, None)


_configure_environment()

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
from torch.utils.data import DataLoader, IterableDataset

from trainlm.config import (
    CheckpointConfig,
    DatasetConfig,
    LossConfig,
    LoggingConfig,
    MonitoringConfig,
    OptimizationConfig,
    OptimizerConfig,
    ParallelismConfig,
    RuntimeConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    ModelSourceConfig,
)
from trainlm.data import (
    ContiguousPackedBatchReader,
    HuggingFacePackedShardSource,
    HuggingFaceShardSourceConfig,
    HuggingFaceShardSpec,
    PackedBinaryShardManifest,
    PartitionedPackedBatchReader,
    plan_packed_batch_partition,
    validate_packed_binary_shard,
)
from trainlm.model import load_huggingface_causal_lm
from trainlm.optimization import create_optimizer
from trainlm.runtime import XlaRuntime
from trainlm.tasks import CausalLMTask
from trainlm.training import Trainer, TrainerCallback, create_scheduler


class BatchIterable(IterableDataset):
    def __init__(self, reader: PartitionedPackedBatchReader) -> None:
        self.reader = reader

    def __iter__(self):
        yield from self.reader

    def __len__(self) -> int:
        return len(self.reader)


class PrintMetrics(TrainerCallback):
    def __init__(self, runtime: XlaRuntime) -> None:
        self.runtime = runtime

    def on_metrics(self, state, control, metrics) -> None:
        if self.runtime.is_primary_process:
            print(dict(metrics), flush=True)


def _source(args: argparse.Namespace) -> ModelSourceConfig:
    if args.model_id:
        return ModelSourceConfig(
            provider="huggingface",
            initialization="pretrained",
            name_or_path=args.model_id,
            revision=args.model_revision or None,
            trust_remote_code=args.trust_remote_code,
            dtype="float32",
            use_safetensors=True,
        )
    return ModelSourceConfig(
        provider="huggingface",
        initialization="config",
        model_type="llama",
        dtype="float32",
        config_overrides={
            "vocab_size": 32064,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "max_position_embeddings": args.sequence_length,
            "tie_word_embeddings": True,
        },
    )


def _local_shards(directory: Path):
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No shard manifests found in {directory}")
    shards = []
    for manifest_path in paths:
        manifest = PackedBinaryShardManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        data_path = manifest_path.parent / Path(manifest.data_path)
        document_path = None
        if manifest.documents.path is not None:
            document_path = manifest_path.parent / Path(manifest.documents.path)
        validation = validate_packed_binary_shard(
            manifest, data_path, document_index_file=document_path
        )
        shards.append(
            SimpleNamespace(
                shard_id=manifest.shard_id,
                data_file=data_path,
                manifest=manifest,
                validation=validation,
            )
        )
    return shards


def _shards(args: argparse.Namespace):
    if args.data_mode == "local":
        return _local_shards(Path(args.manifest_dir))
    if len(args.dataset_revision) != 40:
        raise ValueError("--dataset-revision must be a 40-character commit SHA")
    source = HuggingFacePackedShardSource(
        HuggingFaceShardSourceConfig(
            repo_id=args.dataset_repo,
            revision=args.dataset_revision,
            cache_dir=args.hf_cache_dir,
            shards=tuple(
                HuggingFaceShardSpec(
                    shard_id=f"laughlm-v1_shard_{index:05d}",
                    manifest_path=args.manifest_template.format(
                        root=args.dataset_root, index=index
                    ),
                )
                for index in range(
                    args.shard_start, args.shard_start + args.shard_count
                )
            ),
        )
    )
    return list(source.resolve())


def train_fn(index: int, args: argparse.Namespace) -> None:
    del index
    rank = int(xr.global_ordinal())
    world_size = int(xr.world_size())
    if world_size != 8:
        raise RuntimeError(
            f"Torch/XLA launched {world_size} process(es); expected 8 for TPU v5e-8."
        )

    torch.manual_seed(args.seed)
    device = torch_xla.device()
    shards = _shards(args)
    reader = ContiguousPackedBatchReader(
        shards,
        batch_size=args.micro_batch_per_device,
        sequence_length=args.sequence_length,
    )
    partition = plan_packed_batch_partition(
        reader,
        split="train",
        seed=args.seed,
        epoch=0,
        world_size=world_size,
        rank=rank,
        cross_shard_remainder="drop",
        host_remainder="drop",
    )
    partitioned = PartitionedPackedBatchReader(reader, partition)
    loader = DataLoader(
        BatchIterable(partitioned),
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )
    device_loader = pl.MpDeviceLoader(loader, device)

    source = _source(args)
    loaded = load_huggingface_causal_lm(source)
    runtime = XlaRuntime(
        precision="bf16",
        cache_dir=args.cache_dir,
        compile_training=False,
        collect_diagnostics=True,
    )
    config = TrainConfig(
        model=source,
        dataset=DatasetConfig(
            sequence_length=args.sequence_length,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            packing=True,
        ),
        loss=LossConfig(
            implementation="causal_lm",
            normalization="supervised_tokens",
            z_loss=1e-4,
            logits_chunk_size=4096,
        ),
        runtime=RuntimeConfig(device="xla", precision="bf16"),
        parallelism=ParallelismConfig(data=world_size),
        optimizations=OptimizationConfig(
            policy="auto",
            compile=False,
            allow_fallbacks=True,
            compilation_cache_dir=args.cache_dir,
            accumulation_strategy="microstep",
        ),
        optimizer=OptimizerConfig(
            learning_rate=2e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=False,
            mu_dtype="bfloat16",
            nu_dtype="float32",
        ),
        scheduler=SchedulerConfig(
            name="wsd",
            horizon_tokens=20_000_000_000,
            warmup_fraction=0.01,
            stable_fraction=0.95,
            min_lr_ratio=0.05,
        ),
        trainer=TrainerConfig(
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=1.0,
            seed=args.seed,
        ),
        checkpoint=CheckpointConfig(output_dir=Path(args.output_dir)),
        logging=LoggingConfig(log_every_steps=args.log_every_steps),
        monitoring=MonitoringConfig(
            enabled=True,
            compile_metrics=False,
            memory_metrics=False,
            training_integrity=False,
        ),
    )
    config.validate()
    optimizer = create_optimizer(loaded.model.parameters(), config.optimizer)
    scheduler = create_scheduler(optimizer, config.scheduler)
    task = CausalLMTask(
        ignore_index=config.loss.ignore_index,
        normalization=config.loss.normalization,
        z_loss=config.loss.z_loss,
        loss_implementation=config.loss.implementation,
    )
    trainer = Trainer(
        config=config,
        model=loaded.model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        task=task,
        train_dataloader=device_loader,
        callbacks=[PrintMetrics(runtime)],
    )
    state = trainer.train()
    if runtime.is_primary_process:
        print(
            {
                "phase": state.phase.value,
                "steps": state.step,
                "tokens_seen": state.tokens_seen * world_size,
                "last_loss": state.loss,
                "runtime": runtime.diagnostics().values,
            },
            flush=True,
        )
    reader.close()
    xm.rendezvous("trainlm-finished")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--micro-batch-per-device", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--cache-dir", default="jax_cache/trainlm_v5e8")
    parser.add_argument("--output-dir", default="runs/trainlm_v5e8")
    parser.add_argument("--manifest-dir", default="data/packed/train")
    parser.add_argument("--data-mode", choices=("local", "hf"), default="local")
    parser.add_argument("--dataset-repo", default="LaughTaleAI/LaughLM-Tokenized-Fine")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--dataset-root", default="laughlm-v1")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--manifest-template", default="{root}/laughlm-v1_shard_{index:05d}.manifest.json")
    parser.add_argument("--hf-cache-dir", default="/tmp/laughlm_hf_cache")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import torch_xla

    torch_xla.launch(train_fn, args=(_parse_args(),))
