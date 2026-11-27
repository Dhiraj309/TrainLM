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
    load_or_init_135m_model,
    load_tokenizer,
    make_synthetic_dataset,
    resolve_loss_rematerialization,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: 5B-token 4K annealing.")
    parser.add_argument("--init-checkpoint", required=True, help="Final Stage 1 checkpoint.")
    parser.add_argument("--repo-id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--subfolder", default="stage2")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--rope-theta", type=float, default=100000.0)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--logits-chunk-size", type=int, default=16384)
    parser.add_argument("--loss-rematerialization", default="auto")
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", default=None, help="Stage 2 checkpoint path or 'latest'.")
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
    parser.add_argument("--output-dir", default="runs/stage2_anneal_5b")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from trainlm import TrainLMTrainer, TrainLMTrainingArguments

    if args.dry_run:
        ds = make_synthetic_dataset(4, 128, vocab_size=VOCAB_SIZE)
        with tempfile.TemporaryDirectory() as tmp:
            init_arg = None if args.init_checkpoint == "synthetic" else args.init_checkpoint
            model = load_or_init_135m_model(
                init_checkpoint=init_arg,
                max_position_embeddings=128,
                rope_theta=args.rope_theta,
            )
            t_args = TrainLMTrainingArguments(
                output_dir=tmp,
                accelerator="cpu",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
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
                report_to="none",
            )
            trainer = TrainLMTrainer(model=model, args=t_args, train_dataset=ds)
            state = trainer.train()
            assert state.step == 2
            print(f"[Stage 2 Dry-Run] OK | step={state.step}", flush=True)
        return 0

    from trainlm.data import PackedBinDataset

    cuda_avail = torch.cuda.is_available()
    remat = resolve_loss_rematerialization(args.loss_rematerialization, cuda_available=cuda_avail)
    tokenizer = load_tokenizer(args.tokenizer_id)
    dataset = PackedBinDataset.from_hub(
        args.repo_id,
        subfolder=args.subfolder,
        max_shards=args.max_shards,
        sequence_length=args.seq_len,
        header_bytes=0,
        vocab_size=VOCAB_SIZE,
    )
    model = load_or_init_135m_model(
        init_checkpoint=args.init_checkpoint,
        max_position_embeddings=args.seq_len,
        rope_theta=args.rope_theta,
    )
    t_args = TrainLMTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        sequence_length=args.seq_len,
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
        attn_implementation="sdpa",
        loss_implementation="chunked_linear",
        logits_chunk_size=args.logits_chunk_size,
        loss_rematerialization=remat,
        optim="adamw_torch",
        report_to="none",
    )
    trainer = TrainLMTrainer(
        model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
    )
    state = trainer.train(resume_from_checkpoint=args.resume)
    final_dir = atomic_save_model_snapshot(
        trainer.model, tokenizer, Path(args.output_dir) / "final_model"
    )
    print(f"[Stage 2] Complete | step={state.step} loss={state.loss} | export={final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
