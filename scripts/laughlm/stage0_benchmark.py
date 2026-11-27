"""Stage 0: MI300X micro-batch / compile sweep + optional WSD stability check."""

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
    parser = argparse.ArgumentParser(description="Stage 0: MI300X MFU sweep on 1 real shard.")
    parser.add_argument("--repo-id", default=DEFAULT_SHARD_REPO)
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-sizes", default="32,64,128,256")
    parser.add_argument("--target-batch-tokens", type=int, default=524_288)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument("--stability-steps", type=int, default=0)
    parser.add_argument("--peak-lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--logits-chunk-size", type=int, default=16384)
    parser.add_argument("--loss-rematerialization", default="auto")
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
    parser.add_argument("--output-dir", default="runs/stage0_benchmark")
    return parser.parse_args(argv)


def _run_dry_run(output_dir: Path) -> int:
    from trainlm import TrainLMTrainer, TrainLMTrainingArguments

    ds = make_synthetic_dataset(4, 64, vocab_size=VOCAB_SIZE)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / output_dir.name
        model = load_or_init_135m_model(max_position_embeddings=64)
        args = TrainLMTrainingArguments(
            output_dir=str(out),
            accelerator="cpu",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
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
            warmup_steps=1,
            report_to="none",
        )
        trainer = TrainLMTrainer(model=model, args=args, train_dataset=ds)
        state = trainer.train()
        summary = trainer.performance_summary()
        assert state.step == 2
        snap = atomic_save_model_snapshot(trainer.model, None, out / "final_model")
        reloaded = load_or_init_135m_model(init_checkpoint=snap, max_position_embeddings=128)
        assert reloaded.config.vocab_size == VOCAB_SIZE
        print(
            f"[Stage 0 Dry-Run] OK | step={state.step} | "
            f"status={summary.get('status')} | tok/s={summary.get('tokens_per_second', 0):.1f}",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return _run_dry_run(Path(args.output_dir))

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

    cuda_avail = torch.cuda.is_available()
    remat = resolve_loss_rematerialization(args.loss_rematerialization, cuda_available=cuda_avail)
    mbs_list = [int(x.strip()) for x in args.micro_batch_sizes.split(",") if x.strip()]
    compile_modes = (False,) if args.no_compile else (False, True)
    total_steps = args.warmup_steps + args.measure_steps

    best_cfg: tuple[bool, int, int, float] | None = None
    for compiled in compile_modes:
        for mbs in mbs_list:
            grad_accum = max(1, args.target_batch_tokens // (mbs * args.seq_len))
            if cuda_avail:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = load_or_init_135m_model(max_position_embeddings=max(4096, args.seq_len))
            t_args = TrainLMTrainingArguments(
                output_dir=str(Path(args.output_dir) / f"mbs{mbs}_compile_{compiled}"),
                per_device_train_batch_size=mbs,
                gradient_accumulation_steps=grad_accum,
                sequence_length=args.seq_len,
                max_steps=total_steps,
                logging_steps=max(1, args.warmup_steps),
                save_strategy="no",
                bf16=True,
                torch_compile=compiled,
                torch_compile_mode=args.torch_compile_mode,
                attn_implementation="sdpa",
                loss_implementation="chunked_linear",
                logits_chunk_size=args.logits_chunk_size,
                loss_rematerialization=remat,
                optim="adamw_torch",
                learning_rate=args.peak_lr,
                max_grad_norm=args.grad_clip,
                lr_scheduler_type="wsd",
                warmup_steps=args.warmup_steps,
                report_to="none",
            )
            try:
                trainer = TrainLMTrainer(
                    model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
                )
                trainer.train()
                summary = trainer.performance_summary()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        f"[Stage 0] compile={compiled} mbs={mbs} grad_accum={grad_accum} | OOM — skipped",
                        flush=True,
                    )
                    del model
                    if cuda_avail:
                        torch.cuda.empty_cache()
                    continue
                raise
            perf = summary.get("steady_state") or summary
            peak_gb = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if cuda_avail
                else 0.0
            )
            mfu = (perf.get("non_embedding_mfu") or 0.0) * 100.0
            tps = float(perf.get("tokens_per_second") or 0.0)
            print(
                f"[Stage 0] compile={compiled} mbs={mbs} grad_accum={grad_accum} | "
                f"{tps:,.1f} tok/s | MFU={mfu:.2f}% | peak_vram={peak_gb:.2f} GiB",
                flush=True,
            )
            if best_cfg is None or tps > best_cfg[3]:
                best_cfg = (compiled, mbs, grad_accum, tps)

    if args.stability_steps > 0 and best_cfg is not None:
        win_compiled, win_mbs, win_ga, win_tps = best_cfg
        print(
            f"[Stage 0 Stability] Running {args.stability_steps} steps on winner "
            f"compile={win_compiled} mbs={win_mbs} grad_accum={win_ga} ({win_tps:,.1f} tok/s)",
            flush=True,
        )
        if cuda_avail:
            torch.cuda.empty_cache()
        model = load_or_init_135m_model(max_position_embeddings=max(4096, args.seq_len))
        t_args = TrainLMTrainingArguments(
            output_dir=str(Path(args.output_dir) / "stability"),
            per_device_train_batch_size=win_mbs,
            gradient_accumulation_steps=win_ga,
            sequence_length=args.seq_len,
            max_steps=args.stability_steps,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            torch_compile=win_compiled,
            torch_compile_mode=args.torch_compile_mode,
            attn_implementation="sdpa",
            loss_implementation="chunked_linear",
            logits_chunk_size=args.logits_chunk_size,
            loss_rematerialization=remat,
            optim="adamw_torch",
            learning_rate=args.peak_lr,
            max_grad_norm=args.grad_clip,
            lr_scheduler_type="wsd",
            warmup_steps=min(50, args.stability_steps),
            report_to="none",
        )
        trainer = TrainLMTrainer(
            model=model, args=t_args, train_dataset=dataset, processing_class=tokenizer
        )
        state = trainer.train()
        print(f"[Stage 0 Stability] step={state.step} final_loss={state.loss}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
