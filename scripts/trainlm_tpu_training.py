"""Training body imported inside an initialized PJRT worker.

The notebook process is a coordinator only. Model, data, optimizer, and Trainer
objects are constructed inside ``train_fn`` so every PJRT worker owns its local
XLA device and deterministic data partition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


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
    PartitionedPackedBatchReader,
    plan_packed_batch_partition,
)
from trainlm.model import load_huggingface_causal_lm
from trainlm.model.outputs import normalize_causal_lm_output
from trainlm.optimization import create_optimizer
from trainlm.runtime import XlaRuntime
from trainlm.tasks import CausalLMTask
from trainlm.training import Trainer, TrainerCallback, create_scheduler


class BatchIterable(IterableDataset):
    def __init__(self, reader: PartitionedPackedBatchReader) -> None:
        self.reader = reader

    def __iter__(self):
        for batch in self.reader:
            # Packed streams contain no padding or document loss mask. Keep
            # this payload flat so MpDeviceLoader can transfer it directly.
            yield {"input_ids": batch["input_ids"], "labels": batch["labels"]}

    def __len__(self) -> int:
        return len(self.reader)


class PrintMetrics(TrainerCallback):
    def __init__(self, runtime: XlaRuntime, args) -> None:
        self.runtime = runtime
        self.args = args
        self.start_time = None
        self.start_tokens = 0
        self.elapsed = None
        self.measured_tokens = 0

    def on_train_begin(self, state, control):
        if self.args.warmup_steps == 0:
            torch_xla.sync(wait=True)
            self.start_time = time.perf_counter()

    def on_step_end(self, state, control):
        if state.loss is None or not math.isfinite(state.loss):
            raise RuntimeError(f"Non-finite loss on rank {self.runtime.rank}, step {state.step}")
        if state.step == self.args.warmup_steps:
            torch_xla.sync(wait=True)
            self.start_time = time.perf_counter()
            self.start_tokens = state.tokens_seen
        if state.step == self.args.max_steps and self.start_time is not None:
            torch_xla.sync(wait=True)
            self.elapsed = time.perf_counter() - self.start_time
            self.measured_tokens = (state.tokens_seen - self.start_tokens) * self.runtime.world_size

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
            "use_cache": False,
            "_attn_implementation": "sdpa",
        },
    )


def _install_xla_attention(model: torch.nn.Module) -> str | None:
    """Register a TPU-safe HF SDPA adapter without changing model modules.

    Some HF rotary implementations leave Q/K in fp32 while autocast produces
    V in bf16. PyTorch SDPA requires a common Q/K/V dtype, so normalize the
    three tensors immediately before delegating to HF's SDPA implementation.
    """
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "_attn_implementation"):
        return None
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.masking_utils import sdpa_mask
    except ImportError as exc:
        raise RuntimeError(
            "This HF model exposes attention dispatch but the Transformers "
            "attention registry is unavailable."
        ) from exc

    name = "trainlm_xla_sdpa"

    def trainlm_sdpa(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    ):
        target_dtype = value.dtype
        query = query.to(dtype=target_dtype)
        key = key.to(dtype=target_dtype)
        if (
            attention_mask is not None
            and attention_mask.dtype != torch.bool
            and attention_mask.dtype != target_dtype
        ):
            attention_mask = attention_mask.to(dtype=target_dtype)
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )

    AttentionInterface.register(name, trainlm_sdpa)
    # HF requires a matching mask formatter for every custom attention name.
    AttentionMaskInterface.register(name, sdpa_mask)
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter(name)
    else:
        config._attn_implementation = name
    return name


def model_preflight(args: argparse.Namespace) -> None:
    """Load and execute one HF forward on every PJRT rank.

    This deliberately stops before data, optimizer, and Trainer setup. It
    isolates dependency/model/forward/XLA failures from training failures.
    """
    rank = int(xr.global_ordinal())
    device = torch_xla.device()
    runtime = XlaRuntime(device=device, precision="bf16", compile_training=False)
    source = _source(args)
    print({"stage": "model_preflight_load", "rank": rank}, flush=True)
    loaded = load_huggingface_causal_lm(source)
    if hasattr(loaded.model.config, "use_cache"):
        loaded.model.config.use_cache = False
    model = runtime.prepare_model(loaded.model)
    model.train()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    _install_xla_attention(model)
    input_ids = torch.zeros(
        (args.micro_batch_per_device, args.sequence_length),
        dtype=torch.long,
        device=device,
    )
    with runtime.autocast():
        outputs = model(input_ids=input_ids)
        normalized = normalize_causal_lm_output(outputs)
    if normalized.logits is None:
        raise RuntimeError("HF model preflight returned no logits.")
    expected = (args.micro_batch_per_device, args.sequence_length)
    if tuple(normalized.logits.shape[:-1]) != expected:
        raise RuntimeError(
            f"HF model preflight logits shape {tuple(normalized.logits.shape)} "
            f"does not match expected prefix {expected}."
        )
    torch_xla.sync(wait=True)
    print({"stage": "model_preflight_passed", "rank": rank,
           "model_class": type(model).__name__,
           "logits_shape": tuple(normalized.logits.shape)}, flush=True)
    runtime.finalize()




def train_fn(index: int, args: argparse.Namespace, shards) -> None:
    del index
    rank = int(xr.global_ordinal())
    world_size = int(xr.world_size())
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"Torch/XLA launched {world_size} process(es); "
            f"expected {args.expected_world_size}."
        )

    torch.manual_seed(args.seed)
    device = torch_xla.device()
    torch_xla.manual_seed(args.seed, device=device)
    # The entry point already initialized a distinct persistent cache per rank.
    runtime = XlaRuntime(device=device, precision="bf16", compile_training=False,
                         collect_diagnostics=True)
    task = CausalLMTask(
        z_loss=1e-4, loss_implementation="causal_lm",
        assume_all_supervised=True,
    )
    print({"stage": "build_reader", "rank": rank}, flush=True)
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
    if len(partitioned) < args.max_steps * args.gradient_accumulation_steps:
        reader.close()
        raise ValueError("Not enough complete rank-local batches for this run; add shards or reduce steps.")
    loader = DataLoader(
        BatchIterable(partitioned),
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )
    source = _source(args)
    print({"stage": "load_model", "rank": rank}, flush=True)
    loaded = load_huggingface_causal_lm(source)
    if hasattr(loaded.model.config, "use_cache"):
        loaded.model.config.use_cache = False
    input_vocab = loaded.model.get_input_embeddings().weight.shape[0]
    if any(s.manifest.token_id_max >= input_vocab for s in shards):
        raise ValueError("Shard token IDs exceed the selected HF model vocabulary.")
    model = runtime.prepare_model(loaded.model)
    # Device conversion can replace Parameter objects and tied aliases.
    model.tie_weights()
    attention_backend = _install_xla_attention(model)
    if getattr(model.config, "tie_word_embeddings", False):
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise RuntimeError("HF tied embedding aliases were lost during device placement.")
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
            logits_chunk_size=None,
        ),
        runtime=RuntimeConfig(device="xla", precision="bf16"),
        parallelism=ParallelismConfig(data=world_size),
        optimizations=OptimizationConfig(
            policy="auto",
            compile=False,
            allow_fallbacks=True,
            compilation_cache_dir=str(Path(args.cache_dir) / f"rank-{rank}"),
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
    optimizer = create_optimizer(model.parameters(), config.optimizer)
    model_parameter_ids = {id(p) for p in model.parameters()}
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != model_parameter_ids:
        raise RuntimeError("Optimizer references do not match the XLA model.")
    scheduler = create_scheduler(optimizer, config.scheduler)
    metrics = PrintMetrics(runtime, args)
    # Start prefetch after setup succeeds, and close it before closing mappings.
    parallel_loader = pl.ParallelLoader(
        loader, [device], loader_prefetch_size=16, device_prefetch_size=8,
        host_to_device_transfer_threads=1, batches_per_execution=1,
    )
    device_loader = parallel_loader.per_device_loader(device)
    trainer = Trainer(
        config=config,
        model=model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        task=task,
        train_dataloader=device_loader,
        callbacks=[metrics],
    )
    print({"stage": "train_start", "rank": rank,
           "parameters": sum(p.numel() for p in model.parameters()),
           "attention": attention_backend or getattr(model.config, "_attn_implementation", None),
           "loss": "full_logits_causal_ce_z_loss", "max_steps": args.max_steps}, flush=True)
    try:
        state = trainer.train()
        torch_xla.sync(wait=True)
    finally:
        parallel_loader.close()
        reader.close()
    # Rank-local timing alone can overstate DP throughput. Use the slowest
    # replica's synchronized window and report supervised and scheduled tokens.
    elapsed = None
    if metrics.elapsed is not None and metrics.measured_tokens > 0:
        max_elapsed = xm.all_reduce(xm.REDUCE_MAX, torch.tensor(metrics.elapsed, device=device))
        torch_xla.sync(wait=True)
        elapsed = max_elapsed.item()
    summary = {
        "phase": state.phase.value,
        "steps": state.step,
        "global_supervised_tokens": state.tokens_seen * world_size,
        "last_loss_rank0": state.loss,
        "world_size": world_size,
        "expected_world_size": args.expected_world_size,
        "scheduled_tokens_per_update": args.sequence_length * args.micro_batch_per_device
            * args.gradient_accumulation_steps * world_size,
        "measured_global_supervised_tokens": metrics.measured_tokens,
        "measured_seconds_slowest_rank": elapsed,
        "steady_global_supervised_tokens_per_second": (
            metrics.measured_tokens / elapsed if elapsed else None
        ),
        "steady_global_scheduled_tokens_per_second": (
            metrics.measured_tokens * args.sequence_length / (args.sequence_length - 1) / elapsed
            if elapsed else None
        ),
        "warmup_steps": args.warmup_steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model": loaded.metadata.to_dict(),
        "shards": [s.manifest.to_dict() for s in shards],
        "geometry": {"sequence_length": args.sequence_length,
                     "micro_batch_per_device": args.micro_batch_per_device,
                     "gradient_accumulation_steps": args.gradient_accumulation_steps},
        "runtime": dict(runtime.diagnostics().values),
        "launcher_cache": str(Path(args.cache_dir) / f"rank-{rank}"),
        "versions": {"torch": torch.__version__, "torch_xla": torch_xla.__version__},
        "performance_certified": False,
    }
    if runtime.is_primary_process:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        import torch_xla.debug.metrics as xla_metrics
        (output / "xla_metrics.txt").write_text(xla_metrics.metrics_report(), encoding="utf-8")
        print({"stage": "train_finished", "summary": str(output / "summary.json")}, flush=True)
    if args.export_hf:
        xm.rendezvous("trainlm-before-export")
        # All replicas participate in XLA-to-CPU transfer before only rank zero
        # serializes canonical HF weights. This is export, not exact resume.
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if rank == 0:
            model.save_pretrained(Path(args.output_dir) / "hf_export",
                                  state_dict=cpu_state, safe_serialization=True)
    xm.rendezvous("trainlm-finished")
