#!/usr/bin/env python3
"""Bounded-by-steps native TrainLM trainer smoke; run in an isolated child."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_smoke(output_dir: Path, *, compile_enabled: bool, resume: bool) -> None:
    if resume and compile_enabled:
        raise ValueError("Run checkpoint/resume after the eager baseline.")

    import torch_tpu  # noqa: F401 - registers the native PyTorch TPU device
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trainlm import TrainLMTrainer, TrainLMTrainingArguments
    from trainlm.runtime.torchtpu_discovery import discover_tpu

    discovery = discover_tpu(torch, verify_allocation=True, include_properties=True)
    if discovery["status"] != "ready" or discovery["device_count_visible"] != 1:
        raise RuntimeError("This smoke requires one allocatable native TPU device.")
    print(f"device={discovery['device_name'] or 'tpu'}", flush=True)

    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    texts = [
        "TrainLM trains causal language models with PyTorch.",
        "A TPU can accelerate training when configured correctly.",
        "This example checks the user-facing training path.",
        "Real experiments need representative datasets and settings.",
    ] * 4
    rows = []
    for text in texts:
        encoded = tokenizer(
            text, max_length=64, truncation=True, padding="max_length",
            return_tensors="pt",
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
        max_steps=3,
        per_device_train_batch_size=1,
        sequence_length=64,
        learning_rate=5e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        bf16=False,
        torch_compile=compile_enabled,
    )
    trainer = TrainLMTrainer(
        model=model, tokenizer=tokenizer, train_dataset=rows, args=args,
    )
    state = trainer.train()
    if state.step != 3 or state.loss is None or not math.isfinite(state.loss):
        raise RuntimeError("Three-step training did not finish with finite loss.")
    export = trainer.save_model(output_dir / "baseline-export")
    _check_export(export)
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

        resumed_model = AutoModelForCausalLM.from_pretrained(model_id)
        resumed_model.config.pad_token_id = tokenizer.pad_token_id
        resumed_trainer = TrainLMTrainer(
            model=resumed_model, tokenizer=tokenizer, train_dataset=rows,
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

        reference_model = AutoModelForCausalLM.from_pretrained(model_id)
        reference_model.config.pad_token_id = tokenizer.pad_token_id
        reference_trainer = TrainLMTrainer(
            model=reference_model, tokenizer=tokenizer, train_dataset=rows,
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
    args = parser.parse_args()
    run_smoke(args.output_dir, compile_enabled=args.compile, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
