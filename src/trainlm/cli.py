"""Public command-line entry point backed by the TrainLM trainer facade."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .api import (
    TrainLMTrainer,
    TrainLMTrainingArguments,
    _load_public_config,
)
from .data import PackedBinDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainlm")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train from a public YAML config.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-manifest-dir", type=Path, required=True)
    train.add_argument("--eval-manifest-dir", type=Path)
    train.add_argument("--resume-from-checkpoint", type=Path)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the trainer explanation without training.",
    )
    return parser


def _result_payload(result: Any) -> Any:
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return result.to_dict()
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    raise TypeError("Trainer result is not serializable by the public CLI.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a public TrainLM command without exposing private worker arguments."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "train":  # pragma: no cover - argparse owns this guard
        raise AssertionError(f"Unexpected command: {args.command}")
    if args.dry_run and args.resume_from_checkpoint is not None:
        parser.error("--dry-run cannot be combined with --resume-from-checkpoint")
    config = _load_public_config(args.config)
    training_values = config.get("training_args", {})
    if not isinstance(training_values, dict):
        raise TypeError("'training_args' must be a mapping.")
    training_args = TrainLMTrainingArguments(**training_values)
    train_dataset = PackedBinDataset.from_directory(
        args.train_manifest_dir,
        sequence_length=training_args.sequence_length,
        split="train",
        seed=training_args.seed,
    )
    eval_dataset = (
        PackedBinDataset.from_directory(
            args.eval_manifest_dir,
            sequence_length=training_args.sequence_length,
            split="validation",
        )
        if args.eval_manifest_dir is not None
        else None
    )
    trainer = TrainLMTrainer.from_config(
        config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    result = (
        trainer.explain(format="dict")
        if args.dry_run
        else trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    )
    print(json.dumps(_result_payload(result), sort_keys=True, default=str))
    return 0


__all__ = ["main"]
