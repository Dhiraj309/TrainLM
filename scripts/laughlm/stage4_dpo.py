"""Stage 4: Reference-policy DPO (beta=0.1) on smoltalk2 Preference no_think split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.laughlm.common import (
    DEFAULT_TOKENIZER_ID,
    EOS_TOKEN_ID,
    load_or_init_135m_model,
    load_tokenizer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: DPO on smoltalk2 Preference.")
    parser.add_argument("--init-checkpoint", required=True, help="Final Stage 3 SFT checkpoint.")
    parser.add_argument("--dataset-id", default="HuggingFaceTB/smoltalk2")
    parser.add_argument("--dataset-config", default="Preference")
    parser.add_argument(
        "--split",
        default="llama_3.1_tulu_3_8b_preference_mixture_no_think",
    )
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
    parser.add_argument("--output-dir", default="runs/stage4_dpo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from trainlm.dpo import DPOConfig, DPOTrainer, format_dpo_pair

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            init_arg = None if args.init_checkpoint == "synthetic" else args.init_checkpoint
            policy_model = load_or_init_135m_model(init_checkpoint=init_arg, max_position_embeddings=64)
            ref_model = load_or_init_135m_model(init_checkpoint=init_arg, max_position_embeddings=64)
            pairs = [
                {
                    "chosen_input_ids": [10, 11, 12, 13, EOS_TOKEN_ID],
                    "chosen_labels": [-100, -100, 12, 13, EOS_TOKEN_ID],
                    "rejected_input_ids": [10, 11, 20, 21, EOS_TOKEN_ID],
                    "rejected_labels": [-100, -100, 20, 21, EOS_TOKEN_ID],
                }
                for _ in range(4)
            ]
            cfg = DPOConfig(beta=args.beta, max_length=64, bf16=False, gradient_accumulation_steps=1)
            trainer = DPOTrainer(
                model=policy_model,
                ref_model=ref_model,
                tokenizer=type("_T", (), {"pad_token_id": EOS_TOKEN_ID, "eos_token_id": EOS_TOKEN_ID})(),
                train_dataset=pairs,
                output_dir=tmp,
                config=cfg,
                per_device_batch_size=2,
                max_steps=2,
                save_steps=2,
                logging_steps=1,
                device=torch.device("cpu"),
            )
            res = trainer.train()
            assert res["step"] == 2
            print(f"[Stage 4 Dry-Run] OK | {res}", flush=True)
        return 0

    from datasets import load_dataset

    tokenizer = load_tokenizer(args.tokenizer_id)
    raw_ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split)
    if args.max_examples is not None:
        raw_ds = raw_ds.select(range(min(args.max_examples, len(raw_ds))))

    pairs = []
    for row in raw_ds:
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        prompt = row.get("prompt", [])
        if chosen is None or rejected is None:
            continue
        if isinstance(chosen, list) and not prompt and len(chosen) > 1 and isinstance(rejected, list):
            prompt = chosen[:-1]
            chosen = chosen[-1:]
            rejected = rejected[-1:]
        pairs.append(
            format_dpo_pair(
                prompt,
                chosen,
                rejected,
                tokenizer,
                max_length=args.seq_len,
                eos_token_id=EOS_TOKEN_ID,
            )
        )
    print(f"[Stage 4] Prepared {len(pairs):,} DPO pairs.", flush=True)

    policy_model = load_or_init_135m_model(
        init_checkpoint=args.init_checkpoint,
        max_position_embeddings=args.seq_len,
    )
    ref_model = load_or_init_135m_model(
        init_checkpoint=args.init_checkpoint,
        max_position_embeddings=args.seq_len,
    )
    cfg = DPOConfig(
        beta=args.beta,
        max_length=args.seq_len,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.grad_accum,
    )
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=pairs,
        output_dir=args.output_dir,
        config=cfg,
        per_device_batch_size=args.micro_batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    res = trainer.train(resume_from_checkpoint=args.resume)
    print(f"[Stage 4] Complete | {res}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
