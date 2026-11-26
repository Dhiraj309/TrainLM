"""Stage 3: SFT on HuggingFaceTB/smoltalk2 'SFT' *_no_think splits with assistant loss masking."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.laughlm.common import (
    DEFAULT_TOKENIZER_ID,
    EOS_TOKEN_ID,
    atomic_save_model_snapshot,
    load_or_init_135m_model,
    load_tokenizer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: SFT on smoltalk2 no_think.")
    parser.add_argument("--init-checkpoint", required=True, help="Final Stage 2 checkpoint.")
    parser.add_argument("--dataset-id", default="HuggingFaceTB/smoltalk2")
    parser.add_argument("--dataset-config", default="SFT")
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default="runs/stage3_sft")
    return parser.parse_args(argv)


def _tokenize_sft_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, list[int]] | None:
    input_ids: list[int] = []
    loss_mask: list[int] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        prefix = f"<|im_start|>{role}\n"
        body = f"{content}<|im_end|>\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        body_ids = tokenizer.encode(body, add_special_tokens=False)
        is_assistant = 1 if role == "assistant" else 0
        input_ids.extend(prefix_ids)
        loss_mask.extend([0] * len(prefix_ids))
        input_ids.extend(body_ids)
        loss_mask.extend([is_assistant] * len(body_ids))
    if not input_ids or sum(loss_mask) == 0:
        return None
    if input_ids[-1] != EOS_TOKEN_ID:
        input_ids.append(EOS_TOKEN_ID)
        loss_mask.append(1)
    input_ids = input_ids[:max_length]
    loss_mask = loss_mask[:max_length]
    labels = [
        tok if mask == 1 else -100
        for tok, mask in zip(input_ids, loss_mask, strict=True)
    ]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "loss_mask": loss_mask,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from datasets import concatenate_datasets, get_dataset_split_names, load_dataset
    from trainlm import TrainLMTrainer, TrainLMTrainingArguments

    tokenizer = load_tokenizer(args.tokenizer_id)
    split_names = [
        s
        for s in get_dataset_split_names(args.dataset_id, args.dataset_config)
        if s.endswith("_no_think")
    ]
    if not split_names:
        raise ValueError(
            f"No *_no_think splits found in {args.dataset_id} ({args.dataset_config})."
        )
    print(f"[Stage 3] Loading {len(split_names)} *_no_think splits: {split_names}", flush=True)
    parts = [load_dataset(args.dataset_id, args.dataset_config, split=s) for s in split_names]
    raw_ds = concatenate_datasets(parts)
    if args.max_examples is not None:
        raw_ds = raw_ds.select(range(min(args.max_examples, len(raw_ds))))

    examples: list[dict[str, list[int]]] = []
    for row in raw_ds:
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            continue
        item = _tokenize_sft_messages(msgs, tokenizer, max_length=args.seq_len)
        if item is not None:
            examples.append(item)
    print(f"[Stage 3] Prepared {len(examples):,} SFT examples.", flush=True)

    model = load_or_init_135m_model(
        init_checkpoint=args.init_checkpoint,
        max_position_embeddings=args.seq_len,
    )
    t_args = TrainLMTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        sequence_length=args.seq_len,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
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
        model=model, args=t_args, train_dataset=examples, processing_class=tokenizer
    )
    result = trainer.train(resume_from_checkpoint=args.resume)
    final_dir = atomic_save_model_snapshot(
        trainer.model, tokenizer, Path(args.output_dir) / "final_model"
    )
    print(f"[Stage 3] Complete | metrics={result.metrics} | export={final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
