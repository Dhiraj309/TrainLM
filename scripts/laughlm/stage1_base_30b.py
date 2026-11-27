"""Stage 1: 30B-token from-scratch base pretraining at seq_len=2048."""

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
    load_or_init_135m_model,
    load_tokenizer,
    make_synthetic_dataset,
    resolve_loss_rematerialization,
)


def resolve_grad_accum(micro_batch_size: int, grad_accum: int | None) -> int:
    """Keep global batch at ~524,288 tokens/step for seq_len=2048."""
    return grad_accum if grad_accum is not None else max(1, 256 // max(1, micro_batch_size))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: 30B-token base pretraining.")
    parser.add_argument("--repo-id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=None, help="Defaults to max(1, 256 // micro_batch_size).")
    parser.add_argument("--max-tokens", type=int, default=30_000_000_000)
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--logits-chunk-size", type=int, default=16384)
    parser.add_argument("--loss-rematerialization", default="auto")
    parser.add_argument("--torch-compile-mode", default="default", choices=("default",))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
    parser.add_argument("--output-dir", default="runs/stage1_base_30b")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from trainlm import TrainLMTrainer, TrainLMTrainingArguments

    if args.dry_run:
        ds = make_synthetic_dataset(8, 64, vocab_size=VOCAB_SIZE)
        with tempfile.TemporaryDirectory() as tmp:
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
                report_to="none",
            )
            trainer = TrainLMTrainer(model=model, args=t_args, train_dataset=ds)
            state = trainer.train()
            snap = atomic_save_model_snapshot(trainer.model, None, Path(tmp) / "final_model")
            assert snap.is_dir() and state.step == 2
            print(f"[Stage 1 Dry-Run] OK | step={state.step}", flush=True)
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
    dataset = PackedBinDataset.from_hub(
        args.repo_id,
        subfolder=args.subfolder,
        max_shards=args.max_shards,
        sequence_length=args.seq_len,
        header_bytes=0,
        vocab_size=VOCAB_SIZE,
    )
    model = load_or_init_135m_model(max_position_embeddings=max(4096, args.seq_len))
    t_args = TrainLMTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        sequence_length=args.seq_len,
        max_tokens=args.max_tokens,
        warmup_steps=args.warmup_steps,
        learning_rate=args.peak_lr,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="wsd",
        wsd_stable_fraction=1.0,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        torch_compile=not args.no_compile,
        torch_compile_mode=args.torch_compile_mode,
        attn_implementation="sdpa" if accel_kind != "tpu" else "auto",
        loss_implementation="chunked_linear",
        logits_chunk_size=chunk_size,
        loss_rematerialization=remat,
        optim=optim_name,
        report_to="none",
    )
    trainer = TrainLMTrainer(
        model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
    )
    state = trainer.train(resume_from_checkpoint=args.resume)
    final_dir = atomic_save_model_snapshot(
        trainer.model, tokenizer, Path(args.output_dir) / "final_model"
    )
    print(
        f"[Stage 1] Complete | step={state.step} loss={state.loss} ga={grad_accum} | export={final_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
