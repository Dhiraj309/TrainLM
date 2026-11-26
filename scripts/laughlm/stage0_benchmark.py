"""Stage 0: MI300X micro-batch / compile sweep + optional WSD stability check."""

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
    load_or_init_135m_model,
    load_tokenizer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: MI300X MFU sweep on 1 real shard.")
    parser.add_argument("--repo-id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-sizes", default="32,64,128,256")
    parser.add_argument("--target-batch-tokens", type=int, default=524_288)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--stability-steps", type=int, default=0)
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output-dir", default="runs/stage0_benchmark")
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
    print(
        f"[Stage 0] Loaded {len(dataset.shards)} shard(s) from {args.repo_id}/{args.subfolder} "
        f"({sum(s.manifest.token_count for s in dataset.shards):,} tokens)",
        flush=True,
    )

    mbs_list = [int(x.strip()) for x in args.micro_batch_sizes.split(",") if x.strip()]
    compile_modes = (False,) if args.no_compile else (False, True)
    total_steps = args.warmup_steps + args.measure_steps

    for compiled in compile_modes:
        for mbs in mbs_list:
            grad_accum = max(1, args.target_batch_tokens // (mbs * args.seq_len))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = load_or_init_135m_model(max_position_embeddings=max(4096, args.seq_len))
            t_args = TrainLMTrainingArguments(
                output_dir=str(Path(args.output_dir) / f"mbs{mbs}_compile_{compiled}"),
                per_device_train_batch_size=mbs,
                gradient_accumulation_steps=grad_accum,
                sequence_length=args.seq_len,
                max_steps=total_steps,
                logging_steps=args.warmup_steps,
                save_strategy="no",
                bf16=True,
                torch_compile=compiled,
                attn_implementation="sdpa",
                loss_implementation="chunked_linear",
                optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
                learning_rate=args.peak_lr,
                max_grad_norm=args.grad_clip,
                lr_scheduler_type="wsd",
                warmup_steps=args.warmup_steps,
                report_to="none",
            )
            trainer = TrainLMTrainer(
                model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
            )
            result = trainer.train()
            perf = result.performance.get("steady_state") or result.performance
            peak_gb = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            mfu = (perf.get("non_embedding_mfu") or 0.0) * 100.0
            tps = perf.get("tokens_per_second", 0.0)
            print(
                f"[Stage 0] compile={compiled} mbs={mbs} grad_accum={grad_accum} | "
                f"{tps:,.1f} tok/s | MFU={mfu:.2f}% | peak_vram={peak_gb:.2f} GiB",
                flush=True,
            )

    if args.stability_steps > 0:
        mbs = mbs_list[-1]
        grad_accum = max(1, args.target_batch_tokens // (mbs * args.seq_len))
        model = load_or_init_135m_model(max_position_embeddings=max(4096, args.seq_len))
        t_args = TrainLMTrainingArguments(
            output_dir=str(Path(args.output_dir) / "stability"),
            per_device_train_batch_size=mbs,
            gradient_accumulation_steps=grad_accum,
            sequence_length=args.seq_len,
            max_steps=args.stability_steps,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            torch_compile=not args.no_compile,
            attn_implementation="sdpa",
            loss_implementation="chunked_linear",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            learning_rate=args.peak_lr,
            max_grad_norm=args.grad_clip,
            lr_scheduler_type="wsd",
            warmup_steps=min(50, args.stability_steps),
            report_to="none",
        )
        trainer = TrainLMTrainer(
            model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
        )
        res = trainer.train()
        print(f"[Stage 0 Stability] final_loss={res.metrics.get('train_loss')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
