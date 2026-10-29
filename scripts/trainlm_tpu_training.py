"""Training body imported inside an initialized PJRT worker.

The notebook process is a coordinator only. Model, data, optimizer, and Trainer
objects are constructed inside ``train_fn`` so every PJRT worker owns its local
XLA device and deterministic data partition.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import time


import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
from torch.utils.data import DataLoader

from trainlm.config import (
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
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
from trainlm.training import Trainer, TrainerCallback, TrainerPhase, create_scheduler
from trainlm._tpu_lifecycle import (
    _WorkerBatches, _WorkerCheckpointIO, _WorkerEvaluationTask, _global_stop_steps,
)
from trainlm._tpu_optimization import model_source, prepare_session, apply_worker_profile


class PrintMetrics(TrainerCallback):
    def __init__(self, runtime: XlaRuntime, args) -> None:
        self.runtime = runtime
        self.args = args
        self.start_time = None
        self.start_tokens = 0
        self.elapsed = None
        self.measured_tokens = 0
        self.metrics_path = Path(args.output_dir) / "metrics.jsonl"

    def reset_artifact(self):
        if self.runtime.is_primary_process:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.unlink(missing_ok=True)

    def on_train_begin(self, state, control):
        self.reset_artifact()
        if state.step >= self.args.warmup_steps:
            torch_xla.sync(wait=True)
            self.start_time = time.perf_counter()
            self.start_tokens = state.tokens_seen

    def on_step_end(self, state, control):
        if state.loss is not None and not math.isfinite(state.loss):
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
            snapshot = dict(metrics)
            snapshot.setdefault("step", float(state.step))
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            print(snapshot, flush=True)


class EvaluationLoader:
    """Each evaluation pass owns and closes its own XLA prefetch threads."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        parallel = pl.ParallelLoader(self.loader, [self.device])
        try:
            yield from parallel.per_device_loader(self.device)
        finally:
            parallel.close()


class FinalCheckpoint(TrainerCallback):
    """Retain worker-owned state before the training engine releases runtime resources."""

    def __init__(self, checkpoint_io, destination):
        self.checkpoint_io = checkpoint_io
        self.destination = destination
        self.trainer = None
        self.path = None

    def on_train_end(self, state, control):
        if self.destination is None or state.failure is not None:
            return
        runtime = self.trainer.runtime
        success = False
        try:
            runtime.before_checkpoint("final-state")
            self.path = self.checkpoint_io.save(self.trainer, Path(self.destination).resolve())
            success = True
        finally:
            runtime.after_checkpoint("final-state", success=success)


def _export_model(trainer, checkpoint_io, destination, original_attention):
    """All ranks materialize weights; only rank zero writes the plain HF export."""
    export_model = (checkpoint_io._collective(trainer, checkpoint_io.optimization.canonical_model)
                    if checkpoint_io.optimization is not None else trainer.model)
    cpu_state = checkpoint_io._collective(trainer, lambda: {
        key: value.detach().cpu() for key, value in export_model.state_dict().items()
    })
    def write():
        if not trainer.runtime.is_primary_process:
            return
        config = export_model.config
        attention = getattr(config, "_attn_implementation", None)
        try:
            # A private runtime attention registry is not part of a plain HF model.
            config._attn_implementation = original_attention
            export_model.save_pretrained(
                destination, state_dict=cpu_state, safe_serialization=True,
            )
        finally:
            config._attn_implementation = attention
    checkpoint_io._collective(trainer, write)


def _source(args: argparse.Namespace) -> ModelSourceConfig:
    return model_source(args)


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
    runtime = XlaRuntime(
        device=device,
        precision=args.precision,
        compile_training=False,
    )
    source = _source(args)
    print(json.dumps({"stage": "model_preflight_load", "rank": rank}), flush=True)
    loaded = load_huggingface_causal_lm(source)
    if hasattr(loaded.model.config, "use_cache"):
        loaded.model.config.use_cache = False
    optimization = prepare_session(loaded.model, args)
    model = runtime.prepare_model(loaded.model)
    model.train()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    _install_xla_attention(model)
    apply_worker_profile(optimization, model, get_rng=lambda: xm.get_rng_state(device),
                         set_rng=lambda value: xm.set_rng_state(value, device))
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
    print(json.dumps({
        "stage": "model_preflight_passed",
        "rank": rank,
        "model_class": type(model).__name__,
        "logits_shape": tuple(normalized.logits.shape),
    }), flush=True)
    runtime.finalize()




def train_fn(index: int, args: argparse.Namespace, shards) -> None:
    with ExitStack() as resources:
        _train_fn(index, args, shards, resources)


def _train_fn(index: int, args: argparse.Namespace, shards, resources) -> None:
    del index
    # Derive an absolute stop from requested global tokens without mutating the
    # caller's arguments or truncating the worker's fixed accumulation shape.
    args = argparse.Namespace(**vars(args))
    rank = int(xr.global_ordinal())
    world_size = int(xr.world_size())
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"Torch/XLA launched {world_size} process(es); "
            f"expected {args.expected_world_size}."
        )
    args.max_steps = _global_stop_steps(
        args.max_steps, args.max_tokens, sequence_length=args.sequence_length,
        batch_size=args.micro_batch_per_device,
        accumulation_steps=args.gradient_accumulation_steps, world_size=world_size,
    )

    torch.manual_seed(args.seed)
    device = torch_xla.device()
    torch_xla.manual_seed(args.seed, device=device)
    # The entry point already initialized a distinct persistent cache per rank.
    runtime = XlaRuntime(
        device=device,
        precision=args.precision,
        compile_training=False,
        collect_diagnostics=True,
    )
    task = _WorkerEvaluationTask(
        reduce_sum=lambda value: xm.all_reduce(xm.REDUCE_SUM, value),
        z_loss=args.z_loss, loss_implementation="causal_lm",
        assume_all_supervised=True,
    )
    print({"stage": "build_reader", "rank": rank}, flush=True)
    eval_shards = shards.get("evaluation") if isinstance(shards, dict) else None
    shards = shards["train"] if isinstance(shards, dict) else shards
    reader = resources.enter_context(ContiguousPackedBatchReader(
        shards,
        batch_size=args.micro_batch_per_device,
        sequence_length=args.sequence_length,
    ))
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
    if args.operation == "train" and len(partitioned) < args.max_steps * args.gradient_accumulation_steps:
        raise ValueError("Not enough complete rank-local batches for this run; add shards or reduce steps.")
    batches = _WorkerBatches(partitioned)
    loader = DataLoader(
        batches,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_reader = None
    eval_loader = None
    eval_partition = None
    if eval_shards is not None:
        eval_reader = resources.enter_context(ContiguousPackedBatchReader(
            eval_shards, batch_size=args.eval_batch_size,
            sequence_length=args.sequence_length,
        ))
        eval_partition = plan_packed_batch_partition(
            eval_reader, split="validation", seed=0, epoch=0,
            world_size=world_size, rank=rank,
            cross_shard_remainder="drop", host_remainder="drop",
        )
        evaluation_batches = _WorkerBatches(PartitionedPackedBatchReader(eval_reader, eval_partition))
        if len(evaluation_batches) == 0:
            raise ValueError("Evaluation requires at least one complete batch per rank.")
        eval_loader = EvaluationLoader(DataLoader(
            evaluation_batches, batch_size=None, num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
        ), device)
    source = _source(args)
    print({"stage": "load_model", "rank": rank}, flush=True)
    loaded = load_huggingface_causal_lm(source)
    if hasattr(loaded.model.config, "use_cache"):
        loaded.model.config.use_cache = False
    optimization = prepare_session(loaded.model, args)
    input_vocab = loaded.model.get_input_embeddings().weight.shape[0]
    if any(s.manifest.token_id_max >= input_vocab for s in (*shards, *(eval_shards or ()))):
        raise ValueError("Shard token IDs exceed the selected HF model vocabulary.")
    model = runtime.prepare_model(loaded.model)
    # Device conversion can replace Parameter objects and tied aliases.
    model.tie_weights()
    original_attention = getattr(model.config, "_attn_implementation", None)
    attention_backend = _install_xla_attention(model)
    if attention_backend is not None:
        optimization.runtime_adapters = [{"name": attention_backend, "role": "dtype_compatibility", "performance_certified": False}]
    apply_worker_profile(optimization, model, get_rng=lambda: xm.get_rng_state(device),
                         set_rng=lambda value: xm.set_rng_state(value, device))
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
            z_loss=args.z_loss,
            logits_chunk_size=None,
        ),
        runtime=RuntimeConfig(device="xla", precision=args.precision),
        parallelism=ParallelismConfig(data=world_size),
        optimizations=OptimizationConfig(
            policy=args.optimization_policy,
            compile=False,
            allow_fallbacks=(args.optimization_policy != "required" if args.optimization_fallbacks is None
                             else args.optimization_fallbacks == "allow"),
            compilation_cache_dir=str(Path(args.cache_dir) / f"rank-{rank}"),
            accumulation_strategy="microstep",
        ),
        optimizer=OptimizerConfig(
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            fused=False,
            mu_dtype="bfloat16" if args.precision == "bf16" else "float32",
            nu_dtype="float32",
        ),
        scheduler=SchedulerConfig(
            name=args.scheduler,
            horizon_steps=(
                args.max_steps if args.scheduler in {"linear", "cosine"} else None
            ),
            horizon_tokens=20_000_000_000 if args.scheduler == "wsd" else None,
            warmup_steps=args.warmup_steps,
            warmup_fraction=0.01 if args.scheduler == "wsd" else 0.0,
            stable_fraction=0.95 if args.scheduler == "wsd" else 1.0,
            min_lr_ratio=0.05 if args.scheduler == "wsd" else 0.0,
        ),
        trainer=TrainerConfig(
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            materialize_loss_every_steps=args.log_every_steps,
            max_grad_norm=1.0,
            seed=args.seed,
        ),
        checkpoint=CheckpointConfig(
            output_dir=Path(args.output_dir),
            save_training_every_steps=args.save_steps,
        ),
        evaluation=EvaluationConfig(
            enabled=eval_loader is not None, eval_every_steps=args.eval_steps,
        ),
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
    # Pin semantics that must remain unchanged for same-topology continuation.
    # A constant schedule may extend max_steps; decaying schedules keep their
    # original horizon rather than silently changing the next learning rate.
    context = json.loads(json.dumps({
        "model": asdict(source),
        "model_config": loaded.config.to_dict(),
        "resolved_model_revision": loaded.metadata.resolved_revision,
        "dataset": partition.dataset_fingerprint,
        "schedule": partition.schedule_fingerprint,
        "evaluation_dataset": eval_partition.dataset_fingerprint if eval_partition else None,
        "geometry": [args.sequence_length, args.micro_batch_per_device,
                     args.gradient_accumulation_steps, args.eval_batch_size],
        "precision": args.precision,
        "optimizer": {"lr": args.learning_rate, "betas": [args.beta1, args.beta2],
                      "eps": args.eps, "weight_decay": args.weight_decay},
        "scheduler": {"name": args.scheduler, "warmup_steps": args.warmup_steps,
                      "horizon": args.max_steps if args.scheduler in {"linear", "cosine"} else None},
        "z_loss": args.z_loss,
        "eval_steps": args.eval_steps,
        "versions": {"torch": str(torch.__version__), "torch_xla": str(torch_xla.__version__)},
    }))
    if args.optimization_factory is not None or args.optimization_policy != "auto" or args.optimization_fallbacks is not None:
        context["optimization_plan_id"] = optimization.plan.plan_id
    checkpoint_io = _WorkerCheckpointIO(
        args.output_dir, batches, context,
        get_rng=lambda: xm.get_rng_state(device),
        set_rng=lambda value: xm.set_rng_state(value, device),
        exchange_error=lambda error: xm.mesh_reduce(
            "trainlm-worker-checkpoint-errors", error,
            lambda errors: next((item for item in errors if item is not None), None),
        ),
        optimization=optimization.applied,
    )
    final_state = FinalCheckpoint(checkpoint_io, args.final_checkpoint_dir)
    trainer = Trainer(
        config=config,
        model=model,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        task=task,
        train_dataloader=loader,
        eval_dataloader=eval_loader,
        callbacks=[metrics, final_state],
        checkpoint_saver=checkpoint_io.save,
        checkpoint_loader=checkpoint_io.load,
        model_prepared=True,
    )
    final_state.trainer = trainer
    print({"stage": "train_start", "rank": rank,
           "parameters": sum(p.numel() for p in model.parameters()),
           "attention": attention_backend or getattr(model.config, "_attn_implementation", None),
           "loss": "full_logits_causal_ce_z_loss",
           "materialize_loss_every_steps": args.log_every_steps,
           "max_steps": args.max_steps}, flush=True)
    parallel_loader = None
    evaluation_metrics = None
    export_dir = None
    try:
        if args.resume_from_checkpoint:
            trainer.load_checkpoint(args.resume_from_checkpoint)
        if args.operation == "train":
            # Restore the committed cursor before starting any prefetch threads.
            parallel_loader = pl.ParallelLoader(
                loader, [device], loader_prefetch_size=16, device_prefetch_size=8,
                host_to_device_transfer_threads=1, batches_per_execution=1,
            )
            trainer.train_dataloader = parallel_loader.per_device_loader(device)
            trainer.train()
        else:
            metrics.reset_artifact()
            trainer.prepare()
            if args.operation == "evaluate":
                evaluation_metrics = trainer.evaluate()
            elif args.operation == "save_state":
                final_state.path = trainer.save_checkpoint(Path(args.final_checkpoint_dir).resolve())
        if args.operation == "export" or args.export_hf:
            export_dir = Path(args.export_dir or Path(args.output_dir) / "hf_export").resolve()
            _export_model(trainer, checkpoint_io, export_dir, original_attention)
        torch_xla.sync(wait=True)
    finally:
        if parallel_loader is not None:
            parallel_loader.close()
        if trainer.state.phase != TrainerPhase.FINALIZED:
            trainer.finalize()
    state = trainer.state
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
        "micro_steps": state.micro_step,
        "tokens_seen_rank0": state.tokens_seen,
        "samples_seen_rank0": state.samples_seen,
        "learning_rate": state.learning_rate,
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
                     "gradient_accumulation_steps": args.gradient_accumulation_steps,
                     "materialize_loss_every_steps": args.log_every_steps},
        "runtime": dict(runtime.diagnostics().values),
        "launcher_cache": str(Path(args.cache_dir) / f"rank-{rank}"),
        "versions": {"torch": torch.__version__, "torch_xla": torch_xla.__version__},
        "performance_certified": False,
        "checkpoints": checkpoint_io.saved_paths,
        "resumed_from_checkpoint": args.resume_from_checkpoint,
        "operation": args.operation,
        "final_checkpoint": str(final_state.path) if final_state.path is not None else None,
        "evaluation_metrics": evaluation_metrics,
        "export_dir": str(export_dir) if export_dir is not None else None,
        "max_tokens": args.max_tokens,
        "effective_max_steps": args.max_steps,
        "optimization": optimization.report(),
    }
    if runtime.is_primary_process:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        import torch_xla.debug.metrics as xla_metrics
        (output / "xla_metrics.txt").write_text(xla_metrics.metrics_report(), encoding="utf-8")
        print({"stage": "train_finished", "summary": str(output / "summary.json")}, flush=True)
    xm.rendezvous("trainlm-finished")
