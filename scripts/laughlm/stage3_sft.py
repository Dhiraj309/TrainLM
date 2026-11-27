"""Stage 3: SFT on HuggingFaceTB/smoltalk2 'SFT' *_no_think splits with assistant loss masking."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.laughlm.common import (
    DEFAULT_TOKENIZER_ID,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
    atomic_save_model_snapshot,
    load_or_init_135m_model,
    load_tokenizer,
    make_synthetic_dataset,
    resolve_loss_rematerialization,
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
    parser.add_argument("--logits-chunk-size", type=int, default=16384)
    parser.add_argument("--loss-rematerialization", default="auto")
    parser.add_argument("--torch-compile-mode", default="default", choices=("default",))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run a 2-step CPU/synthetic preflight check.")
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
    from trainlm import TrainLMTrainer, TrainLMTrainingArguments

    if args.dry_run:
        ds = make_synthetic_dataset(4, 64, vocab_size=VOCAB_SIZE)
        for ex in ds:
            ex["loss_mask"] = torch.tensor([0] * 16 + [1] * 48, dtype=torch.long)
            ex["attention_mask"] = torch.ones(64, dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            init_arg = None if args.init_checkpoint == "synthetic" else args.init_checkpoint
            model = load_or_init_135m_model(init_checkpoint=init_arg, max_position_embeddings=64)
            t_args = TrainLMTrainingArguments(
                output_dir=tmp,
                accelerator="cpu",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                sequence_length=64,
                max_steps=2,
                logging_steps=1,
                save_strategy="no",
                bf16=False,
                torch_compile=False,
                loss_implementation="chunked_linear",
                logits_chunk_size=64,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                warmup_steps=1,
                report_to="none",
            )
            trainer = TrainLMTrainer(model=model, args=t_args, train_dataset=ds)
            state = trainer.train()
            assert state.step == 2
            print(f"[Stage 3 Dry-Run] OK | step={state.step}", flush=True)
        return 0

    from datasets import concatenate_datasets, get_dataset_split_names, load_dataset
    from scripts.laughlm.common import (
        detect_accelerator_kind,
        resolve_logits_chunk_size,
        resolve_optim,
    )

    accel_kind = detect_accelerator_kind()
    remat = resolve_loss_rematerialization(args.loss_rematerialization, accel_kind=accel_kind)
    optim_name = resolve_optim(accel_kind)
    chunk_size = resolve_logits_chunk_size(args.logits_chunk_size, accel_kind=accel_kind)
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
        torch_compile_mode=args.torch_compile_mode,
        attn_implementation="sdpa" if accel_kind != "tpu" else "auto",
        loss_implementation="chunked_linear",
        logits_chunk_size=chunk_size,
        loss_rematerialization=remat,
        optim=optim_name,
        report_to="none",
    )
    trainer = TrainLMTrainer(
        model=model, args=t_args, train_dataset=examples, processing_class=tokenizer
    )
    state = trainer.train(resume_from_checkpoint=args.resume)
    final_dir = atomic_save_model_snapshot(
        trainer.model, tokenizer, Path(args.output_dir) / "final_model"
    )
    print(f"[Stage 3] Complete | step={state.step} loss={state.loss} | export={final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
