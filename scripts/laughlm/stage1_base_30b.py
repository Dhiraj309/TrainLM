"""Stage 1: From-scratch base pretraining (Dense 30B or LaughLM-Loop 1B/35B, single or multi-chip)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.laughlm.common import (
    DEFAULT_SHARD_REPO,
    DEFAULT_TOKENIZER_ID,
    VOCAB_SIZE,
    atomic_save_model_snapshot,
    configure_preimport_tpu_compile_cache,
    load_or_init_135m_model,
    load_or_init_loop_135m_model,
    load_tokenizer,
    make_synthetic_dataset,
    parse_loop_depth_schedule,
    resolve_loss_rematerialization,
    resolve_shard_range,
)

LOOP_PRESETS: dict[str, dict[str, int | bool]] = {
    "pilot-1b": {
        "seq_len": 2048,
        "max_tokens": 1_000_000_000,
        "save_steps": 250,
        "use_pallas_recurrent_norm": True,
    },
    "trc-35b-v6e8": {
        "seq_len": 2048,
        "max_tokens": 35_000_000_000,
        "save_steps": 2000,
        "use_pallas_recurrent_norm": True,
    },
}


def resolve_grad_accum(
    micro_batch_size: int,
    grad_accum: int | None,
    *,
    world_size: int = 1,
) -> int:
    """Keep global batch at ~524,288 tokens/step across all ``world_size`` ranks."""
    per_step_seqs = max(1, micro_batch_size * max(1, world_size))
    return grad_accum if grad_accum is not None else max(1, 256 // per_step_seqs)


def resolve_global_batch_sequences(
    *,
    micro_batch_size: int,
    grad_accum: int,
    world_size: int,
) -> int:
    """Return total global batch sequences across microbatches and ranks."""
    return max(1, int(micro_batch_size)) * max(1, int(grad_accum)) * max(1, int(world_size))


def _resolve_distributed_coords() -> tuple[int, int]:
    """Return ``(rank, world_size)`` from ``torch.distributed`` or launch env vars."""
    try:
        import torch  # noqa: PLC0415

        dist_mod = getattr(torch, "distributed", None)
        if (
            dist_mod is not None
            and callable(getattr(dist_mod, "is_initialized", None))
            and dist_mod.is_initialized()
        ):
            return int(dist_mod.get_rank()), int(dist_mod.get_world_size())
    except Exception:
        pass
    rank = int(os.environ.get("RANK", "0") or "0")
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")) or "1"
    )
    return max(0, rank), max(1, world_size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Base pretraining (Dense / LaughLM-Loop).")
    parser.add_argument("--repo-id", "--dataset-repo", dest="repo_id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--gcs-shard-dir", default=None, help="Same-region gs://bucket/prefix shard directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional local cache dir for gs:// staging.")
    parser.add_argument(
        "--compile-cache-dir",
        default=None,
        help="XLA/TorchTPU Tier-3 compilation cache directory or gs:// URI exported before PJRT init.",
    )
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--shard-start", type=int, default=0, help="Inclusive start shard index (default: 0).")
    parser.add_argument("--shard-stop", type=int, default=None, help="Exclusive stop shard index.")
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--arch", default=None, choices=("dense", "laughlm_loop"))
    parser.add_argument(
        "--loop-preset",
        default=None,
        choices=tuple(LOOP_PRESETS.keys()),
        help="Apply a validated LaughLM-Loop preset (defaults --arch to laughlm_loop).",
    )
    parser.add_argument(
        "--no-pallas-recurrent-norm",
        action="store_true",
        help="Disable custom Pallas recurrent_inject_rmsnorm kernel for ablations.",
    )
    parser.add_argument(
        "--deep-init-scale",
        action="store_true",
        help="Scale residual o_proj/down_proj weights by 1/sqrt(2*effective_layers) at init.",
    )
    parser.add_argument(
        "--doc-boundary-mask",
        dest="pack_doc_boundary_mask",
        action="store_true",
        help="Mask packed sequences at EOS document boundaries (Splash SegmentIds / SDPA mask).",
    )
    parser.add_argument(
        "--use-value-residual",
        action="store_true",
        help="Enable opt-in V1 value residuals from first layer into later blocks.",
    )
    parser.add_argument(
        "--use-qk-norm",
        action="store_true",
        help="Enable opt-in RMSNorm on query and key heads before RoPE.",
    )
    parser.add_argument(
        "--use-attn-output-gate",
        action="store_true",
        help="Enable opt-in per-head sigmoid attention output gate before o_proj.",
    )
    parser.add_argument(
        "--loop-depth-schedule",
        default=None,
        help="Optional comma-separated host-side loop cycle (e.g. '2,4,4,6').",
    )
    parser.add_argument("--distributed-strategy", default="auto", choices=("auto", "ddp", "fsdp2", "none"))
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=None, help="Defaults to max(1, 256 // (micro_batch_size * world_size)).")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap; defaults to None so --max-tokens governs.")
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--logits-chunk-size", type=int, default=16384)
    parser.add_argument("--loss-rematerialization", default="auto")
    parser.add_argument("--torch-compile-mode", default="default", choices=("default",))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
    parser.add_argument("--output-dir", default=None)

    ns = parser.parse_args(argv)
    preset = LOOP_PRESETS.get(ns.loop_preset) if ns.loop_preset else None
    if ns.arch is None:
        ns.arch = "laughlm_loop" if ns.loop_preset is not None else "dense"
    if ns.seq_len is None:
        ns.seq_len = int(preset["seq_len"]) if preset else 2048
    if ns.max_tokens is None:
        ns.max_tokens = int(preset["max_tokens"]) if preset else 30_000_000_000
    if ns.save_steps is None:
        ns.save_steps = int(preset["save_steps"]) if preset else 2000
    if ns.output_dir is None:
        ns.output_dir = (
            f"runs/stage1_{ns.loop_preset}" if ns.loop_preset else "runs/stage1_base_30b"
        )
    ns.use_pallas_recurrent_norm = (
        False if ns.no_pallas_recurrent_norm else bool(preset["use_pallas_recurrent_norm"] if preset else True)
    )
    ns.shard_range = resolve_shard_range(ns.shard_start, ns.shard_stop, ns.max_shards)
    ns.loop_depth_schedule = parse_loop_depth_schedule(ns.loop_depth_schedule)
    return ns


def build_training_arguments(
    args: argparse.Namespace,
    *,
    accel_kind: str,
    rank: int,
    grad_accum: int,
    remat: str,
    optim_name: str,
    chunk_size: int,
) -> Any:
    """Build ``TrainLMTrainingArguments`` honoring ``--max-tokens`` without a 1,000-step cap."""
    from trainlm import TrainLMTrainingArguments  # noqa: PLC0415

    return TrainLMTrainingArguments(
        output_dir=args.output_dir,
        accelerator="native_tpu" if accel_kind == "tpu" else ("cuda" if accel_kind == "cuda" else "cpu"),
        distributed_strategy=args.distributed_strategy,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        sequence_length=args.seq_len,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        warmup_steps=args.warmup_steps,
        learning_rate=args.peak_lr,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="wsd",
        wsd_stable_fraction=1.0,
        save_strategy="steps" if rank == 0 else "no",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        torch_compile=not args.no_compile,
        torch_compile_mode=args.torch_compile_mode,
        tpu_compile_cache_dir=args.compile_cache_dir or "auto",
        attn_implementation="sdpa" if accel_kind != "tpu" else "auto",
        loss_implementation="chunked_linear",
        logits_chunk_size=chunk_size,
        loss_rematerialization=remat,
        optim=optim_name,
        loop_depth_schedule=args.loop_depth_schedule,
        report_to="none",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        configure_preimport_tpu_compile_cache(args.compile_cache_dir)

    from trainlm import TrainLMTrainer, TrainLMTrainingArguments

    if args.dry_run:
        ds = make_synthetic_dataset(8, 64, vocab_size=VOCAB_SIZE)
        with tempfile.TemporaryDirectory() as tmp:
            if args.arch == "laughlm_loop":
                from trainlm.config import LaughLMLoopConfig
                from trainlm.model import LaughLMLoopForCausalLM

                cfg = LaughLMLoopConfig.laughlm_loop_135m(
                    vocab_size=VOCAB_SIZE,
                    hidden_size=128,
                    intermediate_size=256,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    num_prelude_layers=1,
                    num_core_layers=1,
                    num_coda_layers=1,
                    max_position_embeddings=64,
                    use_pallas_recurrent_norm=args.use_pallas_recurrent_norm,
                    deep_init_scale=args.deep_init_scale,
                    loop_depth_schedule=args.loop_depth_schedule,
                    pack_doc_boundary_mask=args.pack_doc_boundary_mask,
                    use_value_residual=args.use_value_residual,
                    use_qk_norm=args.use_qk_norm,
                    use_attn_output_gate=args.use_attn_output_gate,
                )
                model = LaughLMLoopForCausalLM(cfg)
            else:
                model = load_or_init_135m_model(max_position_embeddings=64)
            t_args = TrainLMTrainingArguments(
                output_dir=tmp,
                accelerator="cpu",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                sequence_length=64,
                max_steps=2,
                logging_steps=1,
                save_strategy="steps",
                save_steps=2,
                bf16=False,
                torch_compile=False,
                loss_implementation="chunked_linear",
                logits_chunk_size=64,
                optim="adamw_torch",
                lr_scheduler_type="wsd",
                wsd_stable_fraction=1.0,
                warmup_steps=1,
                loop_depth_schedule=args.loop_depth_schedule,
                report_to="none",
            )
            trainer = TrainLMTrainer(model=model, args=t_args, train_dataset=ds)
            state = trainer.train()
            snap = atomic_save_model_snapshot(trainer.model, None, Path(tmp) / "final_model")
            assert snap.is_dir() and state.step == 2
            print(f"[Stage 1 Dry-Run] OK | arch={args.arch} | step={state.step}", flush=True)
        return 0

    from scripts.laughlm.common import (
        detect_accelerator_kind,
        resolve_logits_chunk_size,
        resolve_optim,
    )
    from trainlm.data import PackedBinDataset

    accel_kind = detect_accelerator_kind()
    rank, world_size = _resolve_distributed_coords()
    grad_accum = resolve_grad_accum(
        args.micro_batch_size, args.grad_accum, world_size=world_size
    )
    global_batch_seqs = resolve_global_batch_sequences(
        micro_batch_size=args.micro_batch_size,
        grad_accum=grad_accum,
        world_size=world_size,
    )
    remat = resolve_loss_rematerialization(args.loss_rematerialization, accel_kind=accel_kind)
    optim_name = resolve_optim(accel_kind)
    chunk_size = resolve_logits_chunk_size(args.logits_chunk_size, accel_kind=accel_kind)
    tokenizer = load_tokenizer(args.tokenizer_id)
    shard_kwargs: dict[str, object] = {}
    if args.shard_range is not None:
        shard_kwargs["shard_range"] = args.shard_range
    elif args.max_shards is not None:
        shard_kwargs["max_shards"] = args.max_shards
    if args.gcs_shard_dir:
        dataset = PackedBinDataset.from_gcs(
            args.gcs_shard_dir,
            cache_dir=args.cache_dir,
            sequence_length=args.seq_len,
            header_bytes=0,
            vocab_size=VOCAB_SIZE,
            rank=rank,
            world_size=world_size,
            **shard_kwargs,
        )
    else:
        dataset = PackedBinDataset.from_hub(
            args.repo_id,
            subfolder=args.subfolder,
            sequence_length=args.seq_len,
            header_bytes=0,
            vocab_size=VOCAB_SIZE,
            rank=rank,
            world_size=world_size,
            **shard_kwargs,
        )
    if args.arch == "laughlm_loop":
        model = load_or_init_loop_135m_model(
            max_position_embeddings=max(4096, args.seq_len),
            use_pallas_recurrent_norm=args.use_pallas_recurrent_norm,
            deep_init_scale=args.deep_init_scale,
            loop_depth_schedule=args.loop_depth_schedule,
            pack_doc_boundary_mask=args.pack_doc_boundary_mask,
            use_value_residual=args.use_value_residual,
            use_qk_norm=args.use_qk_norm,
            use_attn_output_gate=args.use_attn_output_gate,
        )
    else:
        model = load_or_init_135m_model(max_position_embeddings=max(4096, args.seq_len))
    t_args = build_training_arguments(
        args,
        accel_kind=accel_kind,
        rank=rank,
        grad_accum=grad_accum,
        remat=remat,
        optim_name=optim_name,
        chunk_size=chunk_size,
    )
    trainer = TrainLMTrainer(
        model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
    )
    state = trainer.train(resume_from_checkpoint=args.resume)
    is_primary = getattr(getattr(trainer, "runtime", None), "is_primary_process", rank == 0)
    final_dir = Path(args.output_dir) / "final_model"
    if is_primary:
        final_dir = atomic_save_model_snapshot(trainer.model, tokenizer, final_dir)
    if trainer.runtime is not None and getattr(trainer.runtime, "is_distributed", False):
        trainer.runtime.barrier("stage1-final-export")
    if is_primary:
        print(
            f"[Stage 1] Complete | arch={args.arch} preset={args.loop_preset} "
            f"world_size={world_size} global_batch={global_batch_seqs} "
            f"step={state.step} loss={state.loss} ga={grad_accum} | export={final_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
