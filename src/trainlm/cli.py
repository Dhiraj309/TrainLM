"""Public command-line entry point backed by the TrainLM trainer facade."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .api import (
    TrainLMTrainer,
    TrainLMTrainingArguments,
    _load_public_config,
)
from .data import PackedBinDataset
from .runtime.diagnostics_cli import collect_diagnostics
from .finetuning import TrainLMFineTuningConfig, run_finetuning
from .pretraining import TrainLMPretrainingConfig
from .pretraining_run import run_pretraining


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainlm")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train from a public YAML config.")
    train.add_argument("--config", type=Path, required=True)
    data = train.add_mutually_exclusive_group(required=True)
    data.add_argument("--train-manifest-dir", type=Path)
    data.add_argument("--dataset-repo")
    train.add_argument("--eval-manifest-dir", type=Path)
    train.add_argument("--dataset-revision", default="main")
    train.add_argument("--train-shard-start", type=int, default=0)
    train.add_argument("--train-shard-stop", type=int)
    train.add_argument("--eval-shard-start", type=int)
    train.add_argument("--eval-shard-stop", type=int)
    train.add_argument("--resume-from-checkpoint", type=Path)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the trainer explanation without training.",
    )
    for name, help_text in (
        ("pretrain", "Run scratch or continued pretraining from a YAML config."),
        ("finetune", "Run causal language-model fine-tuning from a YAML config."),
    ):
        workflow = commands.add_parser(name, help=help_text)
        workflow.add_argument("--config", type=Path, required=True)
        workflow.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the configuration-only training plan without starting a run.",
        )
        workflow.add_argument("--resume-from-checkpoint", type=Path)
        workflow.add_argument(
            "--push-to-hub",
            metavar="REPO_ID",
            help="Publish the completed model export to this Hugging Face repository.",
        )
        workflow.add_argument(
            "--public",
            action="store_true",
            help="Create or use a public Hub repository (requires --push-to-hub).",
        )
        workflow.add_argument(
            "--hub-commit-message",
            help="Commit message for the model upload (requires --push-to-hub).",
        )
        if name == "pretrain":
            workflow.add_argument("--extend-training-budget", action="store_true")
    commands.add_parser(
        "doctor", help="Report package compatibility and available TPU runtime facts."
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
    if args.command == "doctor":
        report = collect_diagnostics()
        print(json.dumps(report, sort_keys=True))
        return 1 if report["status"] != "diagnostics_complete" else 0
    if args.command in {"pretrain", "finetune"}:
        if args.dry_run and (
            args.resume_from_checkpoint is not None
            or args.push_to_hub is not None
            or args.public
            or args.hub_commit_message is not None
        ):
            parser.error("--dry-run cannot be combined with run-only options")
        if args.public and args.push_to_hub is None:
            parser.error("--public requires --push-to-hub REPO_ID")
        if args.hub_commit_message is not None and args.push_to_hub is None:
            parser.error("--hub-commit-message requires --push-to-hub REPO_ID")
        values = _load_workflow_config(args.config)
        if args.command == "pretrain":
            config = TrainLMPretrainingConfig(**values)
            if args.dry_run:
                result = config.training_plan()
            else:
                result = run_pretraining(
                    config,
                    resume_from_checkpoint=args.resume_from_checkpoint,
                    extend_training_budget=args.extend_training_budget,
                )
                if result.status != "passed":
                    print(result.summary())
                    return 1
                _print_run_result(result, args)
                return 0 if result.status == "passed" else 1
        else:
            training_values = values.pop("args", {})
            if not isinstance(training_values, dict):
                raise TypeError("'args' must be a mapping.")
            config = TrainLMFineTuningConfig(
                args=TrainLMTrainingArguments(**training_values), **values
            )
            if args.dry_run:
                result = config.training_plan()
            else:
                result = run_finetuning(
                    config, resume_from_checkpoint=args.resume_from_checkpoint
                )
                _print_run_result(result, args)
                return 0
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    if args.command != "train":  # pragma: no cover - argparse owns this guard
        raise AssertionError(f"Unexpected command: {args.command}")
    if args.dry_run and args.resume_from_checkpoint is not None:
        parser.error("--dry-run cannot be combined with --resume-from-checkpoint")
    config = _load_public_config(args.config)
    training_values = config.get("training_args", {})
    if not isinstance(training_values, dict):
        raise TypeError("'training_args' must be a mapping.")
    training_args = TrainLMTrainingArguments(**training_values)
    if args.dataset_repo is not None:
        if args.train_shard_stop is None:
            parser.error("--dataset-repo requires --train-shard-stop")
        if (args.eval_shard_start is None) != (args.eval_shard_stop is None):
            parser.error(
                "--eval-shard-start and --eval-shard-stop must be used together"
            )
        if args.eval_manifest_dir is not None:
            parser.error("--eval-manifest-dir cannot be combined with --dataset-repo")
        train_dataset = PackedBinDataset.from_hub(
            args.dataset_repo,
            revision=args.dataset_revision,
            shard_range=(args.train_shard_start, args.train_shard_stop),
            sequence_length=training_args.sequence_length,
            split="train",
            seed=training_args.seed,
        )
        eval_dataset = (
            PackedBinDataset.from_hub(
                args.dataset_repo,
                revision=train_dataset.hub_revision,
                shard_range=(args.eval_shard_start, args.eval_shard_stop),
                sequence_length=training_args.sequence_length,
                split="validation",
            )
            if args.eval_shard_start is not None
            else None
        )
    else:
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


def _load_workflow_config(path: Path) -> dict[str, Any]:
    """Load a user workflow YAML mapping without normalizing its schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Training configuration root must be a YAML mapping.")
    return dict(values)


def _print_run_result(result: Any, args: argparse.Namespace) -> None:
    """Print a completed run summary and, only when requested, its Hub receipt."""
    summary = result.summary()
    if args.push_to_hub is None:
        if isinstance(summary, str):
            print(summary)
        else:
            print(json.dumps(summary, sort_keys=True, default=str))
        return
    published = result.push_to_hub(
        args.push_to_hub,
        private=not args.public,
        commit_message=args.hub_commit_message,
    )
    print(json.dumps(
        {"training": summary, "hub": asdict(published)},
        sort_keys=True,
        default=str,
    ))


__all__ = ["main"]
