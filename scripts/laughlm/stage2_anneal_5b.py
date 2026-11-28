"""Stage 2: 5B-token 4K context-extension annealing with WSD decay (3e-3 -> 0.0)."""

from __future__ import annotations

import argparse
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
    load_tokenizer,
    make_synthetic_dataset,
    parse_loop_depth_schedule,
    resolve_loss_rematerialization,
    resolve_shard_range,
)


def resolve_grad_accum(micro_batch_size: int, grad_accum: int | None) -> int:
    """Keep global batch at ~524,288 tokens/step for seq_len=4096."""
    return grad_accum if grad_accum is not None else max(1, 128 // max(1, micro_batch_size))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: 5B-token 4K annealing.")
    parser.add_argument("--init-checkpoint", required=True, help="Final Stage 1 checkpoint.")
    parser.add_argument("--repo-id", "--dataset-repo", dest="repo_id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--gcs-shard-dir", default=None, help="Same-region gs://bucket/prefix shard directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional local cache dir for gs:// staging.")
    parser.add_argument(
        "--compile-cache-dir",
        default=None,
        help="XLA/TorchTPU Tier-3 compilation cache directory or gs:// URI exported before PJRT init.",
    )
    parser.add_argument("--subfolder", default="stage2")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--shard-start", type=int, default=0, help="Inclusive start shard index (default: 0).")
    parser.add_argument("--shard-stop", type=int, default=None, help="Exclusive stop shard index.")
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--rope-theta", type=float, default=100000.0)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=None, help="Defaults to max(1, 128 // micro_batch_size).")
    parser.add_argument("--max-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap; defaults to None so --max-tokens governs.")
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--logits-chunk-size", type=int, default=16384)
    parser.add_argument("--loss-rematerialization", default="auto")
    parser.add_argument("--torch-compile-mode", default="default", choices=("default",))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--deep-init-scale", action="store_true")
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
    parser.add_argument("--loop-depth-schedule", default=None)
    parser.add_argument("--resume", default=None, help="Stage 2 checkpoint path or 'latest'.")
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
    parser.add_argument("--output-dir", default="runs/stage2_anneal_5b")
    ns = parser.parse_args(argv)
    ns.shard_range = resolve_shard_range(ns.shard_start, ns.shard_stop, ns.max_shards)
    ns.loop_depth_schedule = parse_loop_depth_schedule(ns.loop_depth_schedule)
    return ns


def build_training_arguments(
    args: argparse.Namespace,
    *,
    accel_kind: str,
    grad_accum: int,
    remat: str,
    optim_name: str,
    chunk_size: int,
) -> "TrainLMTrainingArguments":
    from trainlm import TrainLMTrainingArguments  # noqa: PLC0415

    return TrainLMTrainingArguments(
        output_dir=args.output_dir,
        accelerator="native_tpu" if accel_kind == "tpu" else ("cuda" if accel_kind == "cuda" else "cpu"),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        sequence_length=args.seq_len,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        warmup_steps=0,
        learning_rate=args.peak_lr,
        min_lr_ratio=0.0,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="wsd",
        wsd_stable_fraction=0.0,
        save_strategy="steps",
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
        ds = make_synthetic_dataset(8, 128, vocab_size=VOCAB_SIZE)
        with tempfile.TemporaryDirectory() as tmp:
            init_arg = None if args.init_checkpoint == "synthetic" else args.init_checkpoint
            if init_arg is None and any(
                (
                    args.pack_doc_boundary_mask,
                    args.use_value_residual,
                    args.use_qk_norm,
                    args.use_attn_output_gate,
                )
            ):
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
                    max_position_embeddings=128,
                    use_pallas_recurrent_norm=False,
                    deep_init_scale=args.deep_init_scale,
                    loop_depth_schedule=args.loop_depth_schedule,
                    pack_doc_boundary_mask=args.pack_doc_boundary_mask,
                    use_value_residual=args.use_value_residual,
                    use_qk_norm=args.use_qk_norm,
                    use_attn_output_gate=args.use_attn_output_gate,
                )
                model = LaughLMLoopForCausalLM(cfg)
            else:
                model = load_or_init_135m_model(
                    init_checkpoint=init_arg,
                    max_position_embeddings=128,
                    rope_theta=args.rope_theta,
                )
                if args.pack_doc_boundary_mask:
                    model.config.pack_doc_boundary_mask = True
            t_args = TrainLMTrainingArguments(
                output_dir=tmp,
                accelerator="cpu",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                sequence_length=128,
                max_steps=2,
                logging_steps=1,
                save_strategy="no",
                bf16=False,
                torch_compile=False,
                loss_implementation="chunked_linear",
                logits_chunk_size=128,
                optim="adamw_torch",
                lr_scheduler_type="wsd",
                wsd_stable_fraction=0.0,
                min_lr_ratio=0.0,
                warmup_steps=0,
                loop_depth_schedule=args.loop_depth_schedule,
                report_to="none",
            )
            trainer = TrainLMTrainer(model=model, args=t_args, train_dataset=ds)
            state = trainer.train()
            assert state.step == 2
            print(f"[Stage 2 Dry-Run] OK | step={state.step}", flush=True)
        return 0

    from scripts.laughlm.common import (
        detect_accelerator_kind,
        resolve_logits_chunk_size,
        resolve_optim,
    )
    from trainlm.data import PackedBinDataset

    accel_kind = detect_accelerator_kind()
    grad_accum = resolve_grad_accum(args.micro_batch_size, args.grad_accum)
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
            **shard_kwargs,
        )
    else:
        dataset = PackedBinDataset.from_hub(
            args.repo_id,
            subfolder=args.subfolder,
            sequence_length=args.seq_len,
            header_bytes=0,
            vocab_size=VOCAB_SIZE,
            **shard_kwargs,
        )
    model = load_or_init_135m_model(
        init_checkpoint=args.init_checkpoint,
        max_position_embeddings=args.seq_len,
        rope_theta=args.rope_theta,
    )
    if args.pack_doc_boundary_mask:
        model.config.pack_doc_boundary_mask = True
    t_args = build_training_arguments(
        args,
        accel_kind=accel_kind,
        grad_accum=grad_accum,
        remat=remat,
        optim_name=optim_name,
        chunk_size=chunk_size,
    )
    trainer = TrainLMTrainer(
        model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
    )
    state = trainer.train(resume_from_checkpoint=args.resume)
    final_dir = atomic_save_model_snapshot(
        trainer.model, tokenizer, Path(args.output_dir) / "final_model"
    )
    print(
        f"[Stage 2] Complete | step={state.step} loss={state.loss} ga={grad_accum} | export={final_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
