#!/usr/bin/env python3
"""Bounded-by-steps native TrainLM trainer smoke; run in an isolated child."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from importlib import metadata
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
sys.path.insert(0, str(ROOT / "src"))


def _token_documents(
    dataset, tokenizer, *, max_examples: int, batch_size: int = 64,
    text_field: str = "text",
):
    """Yield EOS-separated tokenized documents without materializing a split."""
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise RuntimeError(
            "The selected tokenizer must define an integer EOS token."
        )

    pending: list[str] = []
    seen = 0

    def encode(texts: list[str]):
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        for token_ids in encoded["input_ids"]:
            yield [int(token) for token in token_ids] + [eos_token_id]

    for row in dataset:
        if seen >= max_examples:
            break
        text = row.get(text_field) if isinstance(row, dict) else None
        if not isinstance(text, str):
            raise ValueError(
                f"Training rows must contain a string {text_field!r} field."
            )
        pending.append(text)
        seen += 1
        if len(pending) == batch_size:
            yield from encode(pending)
            pending.clear()
    if pending:
        yield from encode(pending)
    if seen == 0:
        raise ValueError("The selected dataset split contains no text examples.")


def _train_byte_level_tokenizer(
    dataset, *, max_examples: int, vocab_size: int, text_field: str = "text",
):
    """Train a compact tokenizer from the training text without caching it."""
    if max_examples < 1:
        raise ValueError("Tokenizer training examples must be positive.")
    if vocab_size < 260:
        raise ValueError("Byte-level BPE vocab_size must be at least 260.")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    seen = 0

    def text_documents():
        nonlocal seen
        for row in dataset:
            if seen >= max_examples:
                break
            text = row.get(text_field) if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise ValueError(
                    f"Tokenizer rows must contain a string {text_field!r} field."
                )
            seen += 1
            yield text
        if seen == 0:
            raise ValueError("The tokenizer training split contains no text examples.")

    core = Tokenizer(models.BPE(unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.train_from_iterator(
        text_documents(),
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "<|pad|>", "<|bos|>", "<|eos|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    core.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )
    if not all(
        isinstance(token_id, int)
        for token_id in (
            tokenizer.unk_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        )
    ):
        raise RuntimeError("The trained tokenizer is missing a special token ID.")
    return tokenizer, seen


def _validate_scratch_dataset_source(
    *, dataset_id: str, dataset_revision: str, train_split: str,
    validation_split: str, text_field: str,
) -> None:
    if not isinstance(dataset_id, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}", dataset_id
    ):
        raise ValueError("dataset_id must be a Hugging Face owner/name ID.")
    if not isinstance(dataset_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", dataset_revision
    ):
        raise ValueError("dataset_revision must be an immutable 40-character commit SHA.")
    if (
        not isinstance(train_split, str)
        or not train_split.strip()
        or not re.fullmatch(r"[A-Za-z0-9._+-]{1,96}", train_split)
        or not isinstance(validation_split, str)
        or not validation_split.strip()
        or not re.fullmatch(r"[A-Za-z0-9._+-]{1,96}", validation_split)
        or train_split == validation_split
    ):
        raise ValueError("Use distinct non-empty training and validation split names.")
    if not isinstance(text_field, str) or not re.fullmatch(
        r"[A-Za-z0-9._+-]{1,96}", text_field
    ):
        raise ValueError("text_field must name a dataset text column.")


def _scratch_gpt2_config(
    *, vocab_size: int, sequence_length: int, embedding_size: int,
    layers: int, attention_heads: int, pad_token_id: int,
    bos_token_id: int, eos_token_id: int,
):
    """Build GPT-2 architecture locally, without downloading a model config."""
    if min(vocab_size, sequence_length, embedding_size, layers, attention_heads) < 1:
        raise ValueError("Scratch GPT-2 dimensions and vocabulary must be positive.")
    if embedding_size % attention_heads:
        raise ValueError("model embedding size must be divisible by attention heads.")

    from transformers import GPT2Config

    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=sequence_length,
        n_ctx=sequence_length,
        n_embd=embedding_size,
        n_layer=layers,
        n_head=attention_heads,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def run_smoke(
    output_dir: Path, *, compile_enabled: bool, resume: bool,
    from_scratch: bool = False, max_steps: int = 3, train_examples: int = 16,
    sequence_length: int = 64, save_steps: int | None = None,
    eval_examples: int = 128, max_eval_batches: int = 20,
    gradient_accumulation_steps: int = 1, tokenizer_vocab_size: int = 16000,
    model_embedding_size: int = 768, model_layers: int = 6,
    model_attention_heads: int = 12,
    bf16: bool = False,
    learning_rate: float | None = None,
    weight_decay: float = 0.1,
    warmup_steps: int | None = None,
    seed: int = 42,
    dataset_id: str = DATASET_ID, dataset_revision: str = DATASET_REVISION,
    train_split: str = "train", validation_split: str = "validation",
    text_field: str = "text",
) -> None:
    if max_steps < 1 or sequence_length < 1 or gradient_accumulation_steps < 1:
        raise ValueError("Training steps, sequence length, and accumulation must be positive.")
    if from_scratch and min(train_examples, eval_examples, max_eval_batches) < 1:
        raise ValueError("Scratch training/evaluation dataset sizes must be positive.")
    if from_scratch and tokenizer_vocab_size < 260:
        raise ValueError("Byte-level BPE vocab_size must be at least 260.")
    if from_scratch and min(
        model_embedding_size, model_layers, model_attention_heads
    ) < 1:
        raise ValueError("Scratch GPT-2 model dimensions must be positive.")
    if from_scratch and model_embedding_size % model_attention_heads:
        raise ValueError("model embedding size must be divisible by attention heads.")
    if not isinstance(bf16, bool):
        raise ValueError("bf16 must be a boolean.")
    resolved_learning_rate = (
        learning_rate if learning_rate is not None
        else 3e-4 if from_scratch else 5e-5
    )
    if (isinstance(resolved_learning_rate, bool)
            or not isinstance(resolved_learning_rate, (int, float))
            or not math.isfinite(resolved_learning_rate)
            or resolved_learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive.")
    if (isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("weight_decay must be finite and non-negative.")
    resolved_warmup_steps = (
        warmup_steps if warmup_steps is not None
        else min(max_steps // 10, 100) if from_scratch else 0
    )
    if (isinstance(resolved_warmup_steps, bool)
            or not isinstance(resolved_warmup_steps, int)
            or resolved_warmup_steps < 0):
        raise ValueError("warmup_steps must be a non-negative integer.")
    if resolved_warmup_steps > max_steps:
        raise ValueError("warmup_steps must not exceed max_steps.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if from_scratch:
        _validate_scratch_dataset_source(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            train_split=train_split,
            validation_split=validation_split,
            text_field=text_field,
        )
    if save_steps is not None and save_steps < 1:
        raise ValueError("save_steps must be positive when configured.")
    if resume and compile_enabled:
        raise ValueError("Run checkpoint/resume after the eager baseline.")
    if resume and max_steps != 3:
        raise ValueError("The resume comparison requires a three-step baseline.")

    if from_scratch:
        from datasets import load_dataset

    import torch_tpu  # noqa: F401 - registers the native PyTorch TPU device
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trainlm import PackedBinDataset, TrainLMTrainer, TrainLMTrainingArguments
    from trainlm.data.writer import write_packed_binary_shard
    from trainlm.runtime.torchtpu_discovery import discover_tpu, normalize_worker_report

    discovery = discover_tpu(torch, verify_allocation=True, include_properties=True)
    if discovery["status"] != "ready" or discovery["device_count_visible"] != 1:
        raise RuntimeError("This smoke requires one allocatable native TPU device.")
    print(f"device={discovery['device_name'] or 'tpu'}", flush=True)

    model_id = MODEL_ID
    dataset_info = None
    if from_scratch:
        data_root = output_dir / "packed-data"
        train_source = load_dataset(
            dataset_id, revision=dataset_revision,
            split=train_split, streaming=True,
        )
        tokenizer, tokenizer_training_documents = _train_byte_level_tokenizer(
            train_source,
            max_examples=train_examples,
            vocab_size=tokenizer_vocab_size,
            text_field=text_field,
        )
        tokenizer.save_pretrained(output_dir / "trained-tokenizer")
        # Re-open the deterministic stream after consuming it for tokenizer fit.
        train_source = load_dataset(
            dataset_id, revision=dataset_revision,
            split=train_split, streaming=True,
        )
        validation_source = load_dataset(
            dataset_id, revision=dataset_revision,
            split=validation_split, streaming=True,
        )
        train_manifest_path = write_packed_binary_shard(
            _token_documents(
                train_source, tokenizer, max_examples=train_examples,
                text_field=text_field,
            ),
            data_root / "train", shard_id="scratch-train",
            vocab_size=len(tokenizer),
        )
        validation_manifest_path = write_packed_binary_shard(
            _token_documents(
                validation_source, tokenizer, max_examples=eval_examples,
                text_field=text_field,
            ),
            data_root / "validation", shard_id="scratch-validation",
            vocab_size=len(tokenizer),
        )
        rows = PackedBinDataset.from_directory(
            data_root / "train", sequence_length=sequence_length,
            split="train", seed=seed,
        )
        eval_rows = PackedBinDataset.from_directory(
            data_root / "validation", sequence_length=sequence_length,
            split="validation",
        )
        required_train_batches = max_steps * gradient_accumulation_steps
        if len(rows) < required_train_batches:
            available_train_batches = len(rows)
            rows.close()
            eval_rows.close()
            raise ValueError(
                f"Packed train split has {available_train_batches} sequences; "
                f"{required_train_batches} are required for the requested run."
            )
        model_config = _scratch_gpt2_config(
            vocab_size=len(tokenizer),
            sequence_length=sequence_length,
            embedding_size=model_embedding_size,
            layers=model_layers,
            attention_heads=model_attention_heads,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(model_config)
        dataset_info = {
            "id": dataset_id, "revision": dataset_revision,
            "train_split": train_split,
            "validation_split": validation_split,
            "text_field": text_field,
            "train_examples": len(rows), "validation_examples": len(eval_rows),
            "tokenizer": {
                "kind": "byte_level_bpe_from_training_split",
                "training_documents": tokenizer_training_documents,
                "vocab_size": len(tokenizer),
            },
        }
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        validation_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
        dataset_info.update({
            "train_tokens": train_manifest["token_count"],
            "validation_tokens": validation_manifest["token_count"],
            "train_documents": train_manifest["documents"]["document_count"],
            "validation_documents": validation_manifest["documents"]["document_count"],
            "train_shard_sha256": train_manifest["sha256"],
            "validation_shard_sha256": validation_manifest["sha256"],
        })
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=MODEL_REVISION,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=MODEL_REVISION,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        texts = [
            "TrainLM trains causal language models with PyTorch.",
            "A TPU can accelerate training when configured correctly.",
            "This example checks the user-facing training path.",
            "Real experiments need representative datasets and settings.",
        ] * 4
        rows = []
        eval_rows = None
        for text in texts:
            encoded = tokenizer(
                text, max_length=sequence_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            labels = encoded["input_ids"][0].clone()
            labels[encoded["attention_mask"][0] == 0] = -100
            rows.append({
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": labels,
            })

    args = TrainLMTrainingArguments(
        output_dir=output_dir,
        accelerator="native_tpu",
        max_steps=max_steps,
        per_device_train_batch_size=1,
        sequence_length=sequence_length,
        learning_rate=resolved_learning_rate,
        lr_scheduler_type="cosine" if from_scratch else "constant",
        warmup_steps=resolved_warmup_steps,
        logging_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        eval_steps=max(1, max_steps // 10) if from_scratch else None,
        max_eval_batches=max_eval_batches if from_scratch else None,
        bf16=bf16,
        weight_decay=weight_decay,
        seed=seed,
        torch_compile=compile_enabled,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainer = TrainLMTrainer(
        model=model, tokenizer=tokenizer, train_dataset=rows,
        eval_dataset=eval_rows, args=args,
    )
    state = trainer.train()
    if state.step != max_steps or state.loss is None or not math.isfinite(state.loss):
        raise RuntimeError("Training did not finish with finite loss at the requested step.")
    export = trainer.save_model(output_dir / "baseline-export")
    _check_export(export)
    evaluation = trainer.evaluate() if from_scratch else {}
    eval_loss = evaluation.get("eval_loss") if evaluation else None
    if from_scratch and (eval_loss is None or not math.isfinite(eval_loss)):
        raise RuntimeError("Scratch pretraining evaluation did not return finite loss.")
    print("train=PASS", flush=True)
    print("export=PASS", flush=True)
    if compile_enabled:
        print("compile=PASS", flush=True)

    if resume:
        checkpoint = trainer.save_state(output_dir / "resume-checkpoint")
        if not checkpoint.is_file():
            raise RuntimeError("Training state was not saved.")
        del trainer, model
        gc.collect()

        if from_scratch:
            torch.manual_seed(seed)
            resumed_model = AutoModelForCausalLM.from_config(model_config)
        else:
            resumed_model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=MODEL_REVISION,
            )
        resumed_model.config.pad_token_id = tokenizer.pad_token_id
        resumed_trainer = TrainLMTrainer(
            model=resumed_model, tokenizer=tokenizer, train_dataset=rows,
            eval_dataset=eval_rows,
            args=replace(args, max_steps=4),
        )
        resumed_state = resumed_trainer.train(resume_from_checkpoint=checkpoint)
        if (resumed_state.step != 4 or resumed_state.loss is None
                or not math.isfinite(resumed_state.loss)):
            raise RuntimeError("Resume did not finish one finite step.")
        _check_export(resumed_trainer.save_model(output_dir / "resumed-export"))
        resumed_weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in resumed_trainer.model.state_dict().items()
        }
        del resumed_trainer, resumed_model
        gc.collect()

        if from_scratch:
            torch.manual_seed(seed)
            reference_model = AutoModelForCausalLM.from_config(model_config)
        else:
            reference_model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=MODEL_REVISION,
            )
        reference_model.config.pad_token_id = tokenizer.pad_token_id
        reference_trainer = TrainLMTrainer(
            model=reference_model, tokenizer=tokenizer, train_dataset=rows,
            eval_dataset=eval_rows,
            args=replace(args, max_steps=4, output_dir=output_dir / "uninterrupted"),
        )
        reference_state = reference_trainer.train()
        if (reference_state.step != 4 or reference_state.loss is None
                or not math.isfinite(reference_state.loss)):
            raise RuntimeError("Uninterrupted four-step run did not finish.")
        reference_weights = reference_trainer.model.state_dict()
        if set(resumed_weights) != set(reference_weights):
            raise RuntimeError("Resumed and reference tensor names differ.")
        mismatches = sum(
            not torch.equal(resumed_weights[name], tensor.detach().cpu())
            for name, tensor in reference_weights.items()
        )
        if mismatches:
            print(f"mismatched_tensors={mismatches}", flush=True)
            raise RuntimeError("Resumed weights differ from uninterrupted weights.")
        print("resume=PASS", flush=True)
        print("exact_weight_match=PASS", flush=True)

    if from_scratch:
        rows.close()
        eval_rows.close()

    packages = {}
    for name in ("torch", "torch_tpu", "libtpu", "transformers", "tokenizers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    if from_scratch:
        model_summary = {
            "id": "gpt2",
            "revision": None,
            "initialization": "random_from_config",
            "model_type": "gpt2",
            "config_source": "explicit_gpt2_config",
            "config": {
                "vocab_size": len(tokenizer),
                "n_positions": sequence_length,
                "n_embd": model_embedding_size,
                "n_layer": model_layers,
                "n_head": model_attention_heads,
            },
            "parameter_count": parameter_count,
        }
    else:
        model_summary = {
            "id": model_id, "revision": MODEL_REVISION,
            "initialization": "pretrained",
        }
    print("trainer_summary=" + json.dumps({
        "schema_version": 1,
        "precision": "bf16" if bf16 else "fp32",
        "training": {
            "max_steps": max_steps,
            "learning_rate": resolved_learning_rate,
            "weight_decay": weight_decay,
            "lr_scheduler_type": "cosine" if from_scratch else "constant",
            "warmup_steps": resolved_warmup_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "seed": seed,
        },
        "model": model_summary,
        "dataset": dataset_info,
        "metrics": {
            "final_train_loss": float(state.loss),
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
            "eval_perplexity": (
                float(evaluation["eval_perplexity"]) if evaluation else None
            ),
        },
        "device": normalize_worker_report(discovery),
        "packages": packages,
        "checks": {
            "train": "passed", "export": "passed",
            "compile": "passed" if compile_enabled else "not_run",
            "resume": "passed" if resume else "not_run",
            "exact_weight_match": "passed" if resume else "not_run",
        },
    }, sort_keys=True), flush=True)
    print("result=trainer_smoke_passed", flush=True)


def _check_export(path: Path) -> None:
    required = ("config.json", "model.safetensors", "tokenizer_config.json")
    if not all((path / name).is_file() for name in required):
        raise RuntimeError("Model/tokenizer export is incomplete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--train-examples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=16000)
    parser.add_argument("--model-embedding-size", type=int, default=768)
    parser.add_argument("--model-layers", type=int, default=6)
    parser.add_argument("--model-attention-heads", type=int, default=12)
    parser.add_argument(
        "--bf16", action="store_true",
        help="opt into bfloat16 autocast (experimental; off by default)",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()
    run_smoke(
        args.output_dir, compile_enabled=args.compile, resume=args.resume,
        from_scratch=args.from_scratch, max_steps=args.max_steps,
        train_examples=args.train_examples, sequence_length=args.sequence_length,
        save_steps=args.save_steps, eval_examples=args.eval_examples,
        max_eval_batches=args.max_eval_batches,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        model_embedding_size=args.model_embedding_size,
        model_layers=args.model_layers,
        model_attention_heads=args.model_attention_heads,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        train_split=args.train_split,
        validation_split=args.validation_split,
        text_field=args.text_field,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
