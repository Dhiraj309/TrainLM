"""Run the four single-device HF pretraining workflows on a tiny corpus."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from trainlm import TrainLMPretrainingConfig, run_pretraining


def validate(output_dir: Path, model: str, revision: str | None, accelerator: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    if not Path(model).is_dir():
        revision = HfApi().model_info(model, revision=revision).sha
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    dataset_id = "lhoestq/demo1"
    dataset_revision = HfApi().dataset_info(dataset_id).sha
    splits = {"train": "train", "validation": "test"}
    for role, split in splits.items():
        source = load_dataset(dataset_id, revision=dataset_revision, split=split, streaming=True)
        row = next(iter(source))
        ids = tokenizer(row["review"], add_special_tokens=False)["input_ids"]
        (output_dir / f"{role}.jsonl").write_text(json.dumps({"input_ids": ids}) + "\n")

    evidence = {
        "model": model, "model_revision": revision,
        "dataset": dataset_id, "dataset_revision": dataset_revision, "cases": {},
    }
    for initialization in ("random", "pretrained"):
        for data_format in ("text", "tokens"):
            case = f"{initialization}-{data_format}"
            sources = {
                "dataset_id": dataset_id, "dataset_revision": dataset_revision,
                "train_split": splits["train"], "validation_split": splits["validation"],
                "text_field": "review",
            }
            if data_format == "tokens":
                sources = {
                    "train_file": output_dir / "train.jsonl",
                    "validation_file": output_dir / "validation.jsonl",
                }
            config = TrainLMPretrainingConfig(
                output_dir=output_dir / case, model_name_or_path=model, model_revision=revision,
                model_initialization=initialization, data_format=data_format,
                accelerator=accelerator, max_tokens=30, sequence_length=16,
                max_train_examples=1, max_eval_examples=1, **sources,
            )
            result = run_pretraining(config)
            expected_backend = "torchtpu" if accelerator == "native_tpu" else "pytorch"
            backend = result.report["accelerator"]["backend"]
            if backend != expected_backend or not math.isfinite(result.metrics["eval_loss"]):
                raise RuntimeError(f"{case}: unexpected backend or nonfinite evaluation loss")
            if result.training["supervised_tokens"] < config.max_tokens:
                raise RuntimeError(f"{case}: token budget incomplete")
            if result.checkpoint_path is None or not result.checkpoint_path.is_dir():
                raise RuntimeError(f"{case}: missing checkpoint")
            if result.report["accelerator"]["world_size"] != 1:
                raise RuntimeError(f"{case}: expected a single training process")
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve().with_name("infer_local_export.py")),
                 str(result.export_path), "--max-new-tokens", "1"], check=True,
            )
            evidence["cases"][case] = {
                "status": result.status, "backend": backend, "eval_loss": result.metrics["eval_loss"],
                "report": str(result.report_path), "transformers_reload_and_generation": "passed",
                "accelerator": result.report["accelerator"], "runtime": result.report["runtime"],
                "supervised_tokens": result.training["supervised_tokens"],
            }
            (output_dir / "matrix.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(f"{case}: passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="A new directory for all four runs.")
    parser.add_argument("--accelerator", choices=["cpu", "native_tpu"], required=True)
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--model-revision", help="Hub revision, resolved once for all four runs.")
    args = parser.parse_args()
    validate(args.output_dir, args.model, args.model_revision, args.accelerator)
