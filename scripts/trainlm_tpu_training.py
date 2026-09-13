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
from trainlm._tpu_checkpoint import (
    load_tpu_worker_checkpoint,
    save_tpu_worker_checkpoint,
)
from trainlm.data import (
    ContiguousPackedBatchReader,
    PartitionedPackedBatchReader,
    plan_packed_batch_partition,
)
from trainlm.model import load_huggingface_causal_lm
from trainlm.model.outputs import normalize_causal_lm_output
from trainlm.optimization import (
    BatchPrefetchGeometry,
    XLAAdamWPolicy,
    create_optimizer,
    inspect_dense_causal_lm,
    materialize_xla_adamw_policy,
)
from trainlm.runtime import XlaRuntime
from trainlm.tasks import CausalLMTask, LinearCausalLMTrainingView, extract_hidden_state
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
        self.metrics_path = Path(args.output_dir) / "metrics.jsonl"
        self.progress_path = Path(args.output_dir) / "progress.md"

    def _is_primary(self) -> bool:
        # PJRT's master-ordinal helper has varied across runtime releases;
        # global ordinal zero is the stable ownership contract for artifacts.
        return self.runtime.rank == 0

    @staticmethod
    def _progress_bar(step: int, maximum: int, width: int = 30) -> str:
        fraction = min(max(step / maximum, 0.0), 1.0) if maximum else 0.0
        filled = int(round(fraction * width))
        return f"[{'#' * filled}{'-' * (width - filled)}] {fraction * 100:.1f}%"

    def _write_progress(self, state, *, phase: str, summary: dict | None = None) -> None:
        """Atomically replace one human-readable progress document.

        The coordinator can safely display this file while a worker is writing
        it because the temporary file is replaced only after the full document
        has been flushed to disk.
        """

        if not self._is_primary():
            return
        now = time.time()
        global_tokens = state.tokens_seen * self.runtime.world_size
        throughput = None
        scheduled_throughput = None
        elapsed = self.elapsed
        completed_steps = max(state.step - self.args.warmup_steps, 0)
        if self.start_time is not None and completed_steps > 0:
            elapsed = elapsed or max(time.perf_counter() - self.start_time, 0.0)
            if elapsed > 0:
                throughput = max(
                    global_tokens - self.start_tokens * self.runtime.world_size, 0
                ) / elapsed
                scheduled_throughput = throughput * self.args.sequence_length / max(
                    self.args.sequence_length - 1, 1
                )
        eta = None
        if throughput and completed_steps > 0:
            eta = max(self.args.max_steps - state.step, 0) * elapsed / completed_steps
        loss = state.loss
        if summary is not None:
            loss = summary.get("last_loss_rank0", loss)
            throughput = summary.get("steady_global_supervised_tokens_per_second", throughput)
            scheduled_throughput = summary.get(
                "steady_global_scheduled_tokens_per_second", scheduled_throughput
            )
            elapsed = summary.get("measured_seconds_slowest_rank", elapsed)
            eta = 0.0 if state.step >= self.args.max_steps else eta
            phase = summary.get("phase", phase)
        perplexity = None
        if loss is not None:
            try:
                perplexity = math.exp(loss)
            except OverflowError:
                perplexity = math.inf

        def value(item) -> str:
            if item is None:
                return "—"
            if isinstance(item, float) and not math.isfinite(item):
                return "∞" if item > 0 else "NaN"
            if isinstance(item, float):
                return f"{item:,.4f}"
            return f"{item:,}" if isinstance(item, int) else str(item)

        updated = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now))
        lines = [
            "# TrainLM training progress",
            "",
            f"**Status:** `{phase}`  ",
            f"**Last updated:** {updated}",
            "",
            "## Current step",
            "",
            f"**{self._progress_bar(state.step, self.args.max_steps)}**",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Step | {state.step:,} / {self.args.max_steps:,} |",
            f"| Loss | {value(loss)} |",
            f"| Perplexity | {value(perplexity)} |",
            f"| Gradient norm | {value(state.grad_norm)} |",
            f"| Learning rate | {value(state.learning_rate)} |",
            "",
            "## Throughput and time",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Supervised tokens/sec (global) | {value(throughput)} |",
            f"| Scheduled tokens/sec (global) | {value(scheduled_throughput)} |",
            f"| Tokens seen (global) | {global_tokens:,} |",
            f"| Tokens/update (global) | {state.global_batch_size * self.runtime.world_size * self.args.sequence_length:,} |",
            f"| Elapsed (seconds) | {value(elapsed)} |",
            f"| Estimated time remaining (seconds) | {value(eta)} |",
            "",
            "## Run configuration",
            "",
            f"- Data parallel: `{self.runtime.world_size}`; model parallel: `1`",
            f"- Sequence length: `{self.args.sequence_length:,}`",
            f"- Micro batch/device: `{self.args.micro_batch_per_device}`",
            f"- Gradient accumulation: `{self.args.gradient_accumulation_steps}`",
            f"- Loss implementation: `{self.args.loss_implementation}`",
            f"- Metrics artifact: `{self.metrics_path.name}`",
        ]
        temporary = self.progress_path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(self.progress_path)

    def on_train_begin(self, state, control):
        if self._is_primary():
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.unlink(missing_ok=True)
            self._write_progress(state, phase="starting")
        if self.args.warmup_steps == 0:
            torch_xla.sync(wait=True)
            self.start_time = time.perf_counter()

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
        if self._is_primary():
            snapshot = {
                **dict(metrics),
                "global_tokens_seen": float(
                    state.tokens_seen * self.runtime.world_size
                ),
                "global_batch_size": float(
                    state.global_batch_size * self.runtime.world_size
                ),
                "world_size": float(self.runtime.world_size),
            }
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            self._write_progress(state, phase="training")
            print(json.dumps(snapshot, sort_keys=True), flush=True)
            if "step" in snapshot and self.args.eval_every_steps is not None:
                step = int(snapshot["step"])
                if step % self.args.eval_every_steps == 0:
                    print(
                        json.dumps(
                            {"stage": "evaluation_start", "step": step},
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    def on_evaluate(self, state, control) -> None:
        if self._is_primary():
            print(
                json.dumps(
                    {"stage": "evaluation_completed", "step": state.step},
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                self.args.save_every_steps is not None
                and state.step % self.args.save_every_steps == 0
            ):
                print(
                    json.dumps(
                        {"stage": "checkpoint_start", "step": state.step},
                        sort_keys=True,
                    ),
                    flush=True,
                )

    def on_save_checkpoint(self, state, control) -> None:
        if self._is_primary():
            print(
                json.dumps(
                    {"stage": "checkpoint_completed", "step": state.step},
                    sort_keys=True,
                ),
                flush=True,
            )

    def finalize(self, state, summary: dict) -> None:
        self._write_progress(state, phase="finalized", summary=summary)


def _source(args: argparse.Namespace) -> ModelSourceConfig:
    if getattr(args, "model_source_json", ""):
        values = json.loads(args.model_source_json)
        if not isinstance(values, dict):
            raise ValueError("--model-source-json must contain a JSON object.")
        try:
            return ModelSourceConfig(**values)
        except TypeError as exc:
            raise ValueError("--model-source-json has unknown model fields.") from exc
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
        if query.dtype != target_dtype:
            query = query.to(dtype=target_dtype)
        if key.dtype != target_dtype:
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
    model = runtime.prepare_model(loaded.model)
    model.train()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    _install_xla_attention(model)
    # Preflight proves that the HF model can execute on XLA; it is not the
    # training-shape compilation benchmark. Compiling the complete training
    # sequence here duplicates the largest compiler workload across every rank
    # and can consume hundreds of GiB of host RAM before training even starts.
    preflight_sequence_length = min(args.sequence_length, 16)
    input_ids = torch.zeros(
        (args.micro_batch_per_device, preflight_sequence_length),
        dtype=torch.long,
        device=device,
    )
    with runtime.autocast():
        outputs = model(input_ids=input_ids)
        normalized = normalize_causal_lm_output(outputs)
    if normalized.logits is None:
        raise RuntimeError("HF model preflight returned no logits.")
    expected = (args.micro_batch_per_device, preflight_sequence_length)
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
        "sequence_length": preflight_sequence_length,
        "logits_shape": tuple(normalized.logits.shape),
    }), flush=True)
    runtime.finalize()




def train_fn(index: int, args: argparse.Namespace, shards, eval_shards=None) -> None:
    del index
    rank = int(xr.global_ordinal())
    world_size = int(xr.world_size())

    torch.manual_seed(args.seed)
    device = torch_xla.device()
    torch_xla.manual_seed(args.seed, device=device)
    # The entry point already initialized a distinct persistent cache per rank.
    cache_dir = Path(args.cache_dir) / f"rank-{rank}"
    runtime = XlaRuntime(
        device=device,
        precision=args.precision,
        cache_dir=cache_dir,
        cache_already_initialized=True,
        compile_training=False,
        collect_diagnostics=True,
    )
    print(json.dumps({"stage": "build_reader", "rank": rank}), flush=True)
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
    eval_reader = None
    eval_loader = None
    if eval_shards is not None:
        eval_reader = ContiguousPackedBatchReader(
            eval_shards,
            batch_size=args.micro_batch_per_device,
            sequence_length=args.sequence_length,
        )
        eval_partition = plan_packed_batch_partition(
            eval_reader,
            split="validation",
            seed=0,
            epoch=0,
            world_size=world_size,
            rank=rank,
            cross_shard_remainder="drop",
            host_remainder="drop",
        )
        # ``BatchPartitionPlan`` exposes the rank-local schedule as
        # ``batch_indices``. The former ``assignments`` name belonged to an
        # older partition contract and only failed once the TPU worker reached
        # evaluation setup.
        if not eval_partition.batch_indices:
            raise ValueError(
                "Evaluation data has no complete batch for the active TPU topology."
            )
        eval_loader = DataLoader(
            BatchIterable(PartitionedPackedBatchReader(eval_reader, eval_partition)),
            batch_size=None,
            num_workers=0,
            pin_memory=False,
        )
    source = _source(args)
    print(json.dumps({"stage": "load_model", "rank": rank}), flush=True)
    loaded = load_huggingface_causal_lm(source)
    if hasattr(loaded.model.config, "use_cache"):
        loaded.model.config.use_cache = False
    input_vocab = loaded.model.get_input_embeddings().weight.shape[0]
    all_shards = [*shards, *(eval_shards or ())]
    if any(s.manifest.token_id_max >= input_vocab for s in all_shards):
        raise ValueError("Shard token IDs exceed the selected HF model vocabulary.")
    model = runtime.prepare_model(loaded.model)
    # Device conversion can replace Parameter objects and tied aliases.
    model.tie_weights()
    attention_backend = _install_xla_attention(model)
    if getattr(model.config, "tie_word_embeddings", False):
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise RuntimeError("HF tied embedding aliases were lost during device placement.")
    training_view = None
    if args.loss_implementation == "chunked_linear":
        capabilities = inspect_dense_causal_lm(model, source_provider="huggingface")
        base_model = getattr(model, "base_model", None)
        if not isinstance(base_model, torch.nn.Module):
            raise RuntimeError(
                "chunked_linear loss requires a public Hugging Face base_model."
            )

        def hidden_state_provider(module, model_inputs):
            body = getattr(module, "base_model", None)
            if not isinstance(body, torch.nn.Module):
                raise RuntimeError("Hugging Face base_model is unavailable.")
            return extract_hidden_state(body(**model_inputs))

        training_view = LinearCausalLMTrainingView(
            model=model,
            capabilities=capabilities,
            hidden_state_provider=hidden_state_provider,
            provider_id="huggingface.base_model",
        )
    task = CausalLMTask(
        z_loss=args.z_loss,
        loss_implementation=args.loss_implementation,
        assume_all_supervised=True,
        training_view=training_view,
        logits_chunk_size=args.logits_chunk_size or 2048,
        rematerialization=(
            "per_chunk" if args.loss_implementation == "chunked_linear" else "disabled"
        ),
    )
    xla_optimizer = materialize_xla_adamw_policy(
        XLAAdamWPolicy(
            first_moment_dtype=(
                "bfloat16" if args.precision == "bf16" else "float32"
            ),
            gradient_clip_norm=1.0,
            gradient_reduction="mean",
        ),
        OptimizerConfig(
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            fused=False,
        ),
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
            implementation=args.loss_implementation,
            normalization="supervised_tokens",
            z_loss=args.z_loss,
            logits_chunk_size=args.logits_chunk_size,
        ),
        runtime=RuntimeConfig(device="xla", precision=args.precision),
        parallelism=ParallelismConfig(data=world_size),
        optimizations=OptimizationConfig(
            policy="auto",
            compile=False,
            allow_fallbacks=True,
            compilation_cache_dir=str(Path(args.cache_dir) / f"rank-{rank}"),
            accumulation_strategy="microstep",
        ),
        optimizer=xla_optimizer.optimizer,
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
            max_grad_norm=xla_optimizer.gradient_clip_norm,
            seed=args.seed,
        ),
        checkpoint=CheckpointConfig(
            output_dir=Path(args.output_dir),
            save_training_every_steps=args.save_every_steps,
        ),
        logging=LoggingConfig(log_every_steps=args.log_every_steps),
        monitoring=MonitoringConfig(
            enabled=True,
            compile_metrics=False,
            memory_metrics=False,
            training_integrity=False,
        ),
        evaluation=EvaluationConfig(
            enabled=eval_loader is not None,
            eval_every_steps=args.eval_every_steps,
            max_batches=args.max_eval_batches,
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
    # Keep one device-loader execution aligned with several microsteps. A
    # value of one flushes the lazy XLA graph for every batch; capping at eight
    # reduces host/mark_step overhead without allowing an unbounded graph.
    batches_per_execution = min(args.gradient_accumulation_steps, 8)
    input_geometry = BatchPrefetchGeometry(
        geometry_id=(
            f"s{args.sequence_length}-mb{args.micro_batch_per_device}-"
            f"ga{args.gradient_accumulation_steps}-dp{world_size}-"
            f"p16-bpe{batches_per_execution}"
        ),
        sequence_length=args.sequence_length,
        micro_batch_per_device=args.micro_batch_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        data_parallel_replicas=world_size,
        prefetch_depth=16,
        device_prefetch_depth=8,
        host_to_device_transfer_threads=1,
        batches_per_execution=batches_per_execution,
    )
    parallel_loader = pl.ParallelLoader(
        loader, [device], **input_geometry.parallel_loader_kwargs()
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
        eval_dataloader=eval_loader,
        callbacks=[metrics],
        checkpoint_saver=lambda engine, destination: save_tpu_worker_checkpoint(
            engine, Path(args.output_dir) / str(destination)
        ),
        checkpoint_loader=lambda engine, source: load_tpu_worker_checkpoint(
            engine, source
        ),
    )
    if args.resume_from_checkpoint is not None:
        trainer.load_checkpoint(Path(args.resume_from_checkpoint))
        # The packed schedule is deterministic. Rebuild its exact position
        # before entering the training loop so the next batch is not repeated.
        trainer._train_iterator = iter(trainer.train_dataloader)
        for _ in range(trainer.state.micro_step):
            try:
                next(trainer._train_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "TPU checkpoint data position exceeds the available rank schedule."
                ) from exc
    print(
        json.dumps(
            {"stage": "train_start", "rank": rank,
             "parameters": sum(p.numel() for p in model.parameters()),
             "attention": attention_backend or getattr(model.config, "_attn_implementation", None),
             "parallelism": {
                 "data_parallel": world_size,
                 "model_parallel": 1,
             },
             "loss": (
                 "chunked_linear_causal_ce_z_loss"
                 if args.loss_implementation == "chunked_linear"
                 else "full_logits_causal_ce_z_loss"
             ),
             "materialize_loss_every_steps": args.log_every_steps,
             "batches_per_execution": min(args.gradient_accumulation_steps, 8),
             "max_steps": args.max_steps},
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        state = trainer.train()
        torch_xla.sync(wait=True)
    finally:
        parallel_loader.close()
        reader.close()
        if eval_reader is not None:
            eval_reader.close()
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
        "global_batch_size": state.global_batch_size * world_size,
        "samples_per_update": state.global_batch_size * world_size,
        "learning_rate": state.learning_rate,
        "global_supervised_tokens": state.tokens_seen * world_size,
        "last_loss_rank0": state.loss,
        "last_grad_norm_rank0": state.grad_norm,
        "world_size": world_size,
        "parallelism": {"data_parallel": world_size, "model_parallel": 1},
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
                     "batches_per_execution": batches_per_execution,
                     "materialize_loss_every_steps": args.log_every_steps},
        "runtime": dict(runtime.diagnostics().values),
        "launcher_cache": str(cache_dir),
        "summary_writer_rank": rank,
        "versions": {"torch": torch.__version__, "torch_xla": torch_xla.__version__},
        "performance_certified": False,
        "resumed_from_checkpoint": args.resume_from_checkpoint,
        "committed_checkpoints": sorted(
            str(path)
            for path in Path(args.output_dir).glob("checkpoint-*")
            if (path / "manifest.json").is_file()
        ),
    }
    summary["progress_document"] = str(Path(args.output_dir) / "progress.md")
    if rank == 0:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        metrics.finalize(state, summary)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        import torch_xla.debug.metrics as xla_metrics
        (output / "xla_metrics.txt").write_text(xla_metrics.metrics_report(), encoding="utf-8")
        print(
            json.dumps(
                {"stage": "train_finished", "summary": str(output / "summary.json")},
                sort_keys=True,
            ),
            flush=True,
        )
    if args.export_hf:
        xm.rendezvous("trainlm-before-export")
        # All replicas participate in XLA-to-CPU transfer before only rank zero
        # serializes canonical HF weights. This is export, not exact resume.
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if rank == 0:
            model.save_pretrained(Path(args.output_dir) / "hf_export",
                                  state_dict=cpu_state, safe_serialization=True)
    xm.rendezvous("trainlm-finished")
