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
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
sys.path.insert(0, str(ROOT / "src"))


def run_smoke(
    output_dir: Path, *, compile_enabled: bool, resume: bool,
    from_scratch: bool = False, max_steps: int = 3, train_examples: int = 16,
    sequence_length: int = 64, save_steps: int | None = None,
    eval_examples: int = 128, max_eval_batches: int = 20,
    gradient_accumulation_steps: int = 1,
) -> None:
    if max_steps < 1 or sequence_length < 1 or gradient_accumulation_steps < 1:
        raise ValueError("Training steps, sequence length, and accumulation must be positive.")
    if from_scratch and min(train_examples, eval_examples, max_eval_batches) < 1:
        raise ValueError("Scratch training/evaluation dataset sizes must be positive.")
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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from trainlm import TrainLMTrainer, TrainLMTrainingArguments
    from trainlm.runtime.torchtpu_discovery import discover_tpu, normalize_worker_report

    discovery = discover_tpu(torch, verify_allocation=True, include_properties=True)
    if discovery["status"] != "ready" or discovery["device_count_visible"] != 1:
        raise RuntimeError("This smoke requires one allocatable native TPU device.")
    print(f"device={discovery['device_name'] or 'tpu'}", flush=True)

    model_id = MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_info = None
    if from_scratch:
        raw_dataset = load_dataset(
            DATASET_ID, revision=DATASET_REVISION,
            split=f"train[:{train_examples}]",
        )

        def encode_batch(batch):
            encoded = tokenizer(
                batch["text"], max_length=sequence_length, truncation=True,
                padding="max_length",
            )
            encoded["labels"] = [
                [token if mask else -100 for token, mask in zip(ids, attention)]
                for ids, attention in zip(
                    encoded["input_ids"], encoded["attention_mask"]
                )
            ]
            return encoded

        rows = raw_dataset.map(
            encode_batch, batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing pinned TinyStories training examples",
        )
        rows.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "labels",
        ])
        raw_eval = load_dataset(
            DATASET_ID, revision=DATASET_REVISION,
            split=f"validation[:{eval_examples}]",
        )
        eval_rows = raw_eval.map(
            encode_batch, batched=True,
            remove_columns=raw_eval.column_names,
            desc="Tokenizing pinned TinyStories validation examples",
        )
        eval_rows.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "labels",
        ])
        model_config = AutoConfig.from_pretrained(
            model_id, revision=MODEL_REVISION,
        )
        model_config.n_positions = sequence_length
        if hasattr(model_config, "n_ctx"):
            model_config.n_ctx = sequence_length
        model_config.vocab_size = len(tokenizer)
        model_config.pad_token_id = tokenizer.pad_token_id
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(model_config)
        dataset_info = {
            "id": DATASET_ID, "revision": DATASET_REVISION,
            "train_examples": len(rows), "validation_examples": len(eval_rows),
        }
    else:
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
        learning_rate=3e-4 if from_scratch else 5e-5,
        lr_scheduler_type="cosine" if from_scratch else "constant",
        warmup_steps=min(max_steps // 10, 100) if from_scratch else 0,
        logging_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        eval_steps=max(1, max_steps // 10) if from_scratch else None,
        max_eval_batches=max_eval_batches if from_scratch else None,
        bf16=False,
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
            torch.manual_seed(42)
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
            torch.manual_seed(42)
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

    packages = {}
    for name in ("torch", "torch_tpu", "libtpu", "transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    model_summary = {
        "id": model_id, "revision": MODEL_REVISION,
        "initialization": "random_from_config" if from_scratch else "pretrained",
    }
    if from_scratch:
        model_summary["parameter_count"] = parameter_count
    print("trainer_summary=" + json.dumps({
        "schema_version": 1,
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
    args = parser.parse_args()
    run_smoke(
        args.output_dir, compile_enabled=args.compile, resume=args.resume,
        from_scratch=args.from_scratch, max_steps=args.max_steps,
        train_examples=args.train_examples, sequence_length=args.sequence_length,
        save_steps=args.save_steps, eval_examples=args.eval_examples,
        max_eval_batches=args.max_eval_batches,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
