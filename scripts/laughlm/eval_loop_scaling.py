"""Evaluate a LaughLM-Loop checkpoint across recurrent loop depths R in [1, 2, 4, 6, 8]."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate validation loss, perplexity, and tok/s across LaughLM-Loop depths."
    )
    parser.add_argument("--checkpoint", default=None, help="Path or HF repo for LaughLMLoopForCausalLM.")
    parser.add_argument("--repo-id", "--dataset-repo", dest="repo_id", default=None, help="Optional shard Hub dataset repo for eval.")
    parser.add_argument("--gcs-shard-dir", default=None, help="Optional same-region gs://bucket/prefix validation shard directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional local cache dir for gs:// staging.")
    parser.add_argument("--subfolder", default="stage1")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--loops", default="1,2,4,6,8")
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic CPU loop-scaling check.")
    return parser.parse_args(argv)


def _evaluate_loop_depths(
    model,
    batches: list,
    *,
    loops_list: list[int],
    device,
) -> list[dict[str, float]]:
    import torch
    import torch.nn.functional as F

    model.eval()
    results = []
    vocab_size = int(model.config.vocab_size)
    for num_loops in loops_list:
        total_loss = 0.0
        total_tokens = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in batches:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, num_loops=num_loops)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                batch_tokens = int(shift_labels.ne(-100).sum().item())
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += batch_tokens
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "tpu":
            sync = getattr(getattr(torch, "tpu", None), "synchronize", None)
            if callable(sync):
                sync()
        elapsed = max(1e-9, time.perf_counter() - t0)
        mean_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(mean_loss, 20.0))
        tok_s = total_tokens / elapsed
        results.append(
            {
                "loops": float(num_loops),
                "loss": mean_loss,
                "ppl": ppl,
                "tokens_per_second": tok_s,
            }
        )
    return results


def _run_dry_run(loops_list: list[int]) -> int:
    import torch
    from trainlm.config import LaughLMLoopConfig
    from trainlm.model import LaughLMLoopForCausalLM

    with tempfile.TemporaryDirectory():
        cfg = LaughLMLoopConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_prelude_layers=1,
            num_core_layers=1,
            num_coda_layers=1,
            num_loop_iterations=4,
            train_min_loops=2,
            train_max_loops=4,
            truncated_bptt_steps=4,
            use_pallas_recurrent_norm=False,
            max_position_embeddings=64,
        )
        model = LaughLMLoopForCausalLM(cfg)
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(0, cfg.vocab_size, (4, 32), generator=gen)
        batches = [{"input_ids": ids[:2], "labels": ids[:2]}, {"input_ids": ids[2:], "labels": ids[2:]}]
        rows = _evaluate_loop_depths(
            model,
            batches,
            loops_list=loops_list,
            device=torch.device("cpu"),
        )
        for row in rows:
            print(
                f"[Eval Dry-Run] R={int(row['loops'])} | loss={row['loss']:.4f} | "
                f"ppl={row['ppl']:.2f} | tok/s={row['tokens_per_second']:.1f}",
                flush=True,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loops_list = [int(x.strip()) for x in args.loops.split(",") if x.strip()]
    if args.dry_run:
        return _run_dry_run(loops_list)
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --dry-run is set.")

    import torch
    from trainlm.data import PackedBinDataset
    from trainlm.model import LaughLMLoopForCausalLM
    from scripts.laughlm.common import VOCAB_SIZE, detect_accelerator_kind

    accel = detect_accelerator_kind()
    device = torch.device("cuda" if accel == "cuda" else ("tpu" if accel == "tpu" else "cpu"))
    dtype = torch.bfloat16 if device.type in {"cuda", "tpu"} else torch.float32
    model = LaughLMLoopForCausalLM.from_pretrained(args.checkpoint).to(device=device, dtype=dtype)

    if args.gcs_shard_dir:
        ds = PackedBinDataset.from_gcs(
            args.gcs_shard_dir,
            cache_dir=args.cache_dir,
            max_shards=args.max_shards,
            sequence_length=args.seq_len,
            header_bytes=0,
            vocab_size=int(model.config.vocab_size),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
        batches = []
        for idx, batch in enumerate(loader):
            batches.append(batch)
            if idx + 1 >= args.max_eval_batches:
                break
    elif args.repo_id:
        ds = PackedBinDataset.from_hub(
            args.repo_id,
            subfolder=args.subfolder,
            max_shards=args.max_shards,
            sequence_length=args.seq_len,
            header_bytes=0,
            vocab_size=int(model.config.vocab_size),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
        batches = []
        for idx, batch in enumerate(loader):
            batches.append(batch)
            if idx + 1 >= args.max_eval_batches:
                break
    else:
        gen = torch.Generator().manual_seed(123)
        rows = args.batch_size * max(1, args.max_eval_batches)
        ids = torch.randint(0, int(model.config.vocab_size), (rows, args.seq_len), generator=gen)
        batches = [
            {
                "input_ids": ids[i : i + args.batch_size],
                "labels": ids[i : i + args.batch_size],
            }
            for i in range(0, rows, args.batch_size)
        ]

    del VOCAB_SIZE
    rows = _evaluate_loop_depths(model, batches, loops_list=loops_list, device=device)
    print(f"=== LaughLM-Loop Depth Scaling ({args.checkpoint}, seq={args.seq_len}, bs={args.batch_size}) ===")
    for row in rows:
        print(
            f"  R={int(row['loops']):2d} | loss={row['loss']:.4f} | "
            f"ppl={row['ppl']:.2f} | {row['tokens_per_second']:8.1f} tok/s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
