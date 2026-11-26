"""Stage 1: 30B-token from-scratch base pretraining at seq_len=2048."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: 30B-token base pretraining.")
    parser.add_argument("--repo-id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=30_000_000_000)
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--output-dir", default="runs/stage1_base_30b")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from trainlm import TrainLMTrainer, TrainLMTrainingArguments
    from trainlm.data import PackedBinDataset

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
        gradient_accumulation_steps=args.grad_accum,
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
        attn_implementation="sdpa",
        loss_implementation="chunked_linear",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
    )
    trainer = TrainLMTrainer(
        model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
    )
    result = trainer.train(resume_from_checkpoint=args.resume)
    final_dir = atomic_save_model_snapshot(
        trainer.model, tokenizer, Path(args.output_dir) / "final_model"
    )
    print(f"[Stage 1] Complete | metrics={result.metrics} | export={final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
