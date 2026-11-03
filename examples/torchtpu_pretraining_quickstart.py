#!/usr/bin/env python3
"""Small, user-facing entry point for the native TorchTPU scratch notebook.

The detailed worker command and runtime checks stay here so notebook users can
focus on their data source, token budget, and sequence length.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import runpy
import sys
from typing import Any


DEFAULT_DATASET_ID = "roneneldan/TinyStories"
DEFAULT_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
_DATASET_ID = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_FIELD = re.compile(r"[A-Za-z0-9._+-]{1,96}\Z")


def build_training_command(
    *,
    repo_root: Path,
    output_dir: Path,
    target_supervised_tokens: int,
    sequence_length: int,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    train_split: str = "train",
    validation_split: str = "validation",
    text_field: str = "text",
) -> list[str]:
    """Build the bounded worker command behind the beginner-facing API."""
    if (isinstance(target_supervised_tokens, bool)
            or not isinstance(target_supervised_tokens, int)
            or target_supervised_tokens < 1):
        raise ValueError("target_supervised_tokens must be a positive integer.")
    if (isinstance(sequence_length, bool) or not isinstance(sequence_length, int)
            or sequence_length < 2):
        raise ValueError("sequence_length must be an integer of at least two.")
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset_id must be a Hugging Face owner/name ID.")
    if not isinstance(dataset_revision, str) or not _REVISION.fullmatch(dataset_revision):
        raise ValueError("dataset_revision must be an immutable 40-character commit SHA.")
    if not all(isinstance(value, str) and _FIELD.fullmatch(value) for value in (
        train_split, validation_split, text_field,
    )):
        raise ValueError("Dataset splits and text_field must use simple column names.")
    if train_split == validation_split:
        raise ValueError("Training and validation split names must differ.")

    # Defaults below are intentionally maintained in one place. Beginner users
    # only need the corpus, token budget, sequence length, and output location.
    updates = math.ceil(target_supervised_tokens / (sequence_length - 1))
    save_steps = max(1, updates // 10)
    return [
        sys.executable,
        str(Path(repo_root) / "examples/torchtpu_native_trainer_smoke.py"),
        "--output-dir", str(output_dir),
        "--from-scratch",
        "--target-supervised-tokens", str(target_supervised_tokens),
        "--train-examples", "2000",
        "--tokenizer-vocab-size", "16000",
        "--model-embedding-size", "768",
        "--model-layers", "6",
        "--model-attention-heads", "12",
        "--dataset-id", dataset_id,
        "--dataset-revision", dataset_revision,
        "--train-split", train_split,
        "--validation-split", validation_split,
        "--text-field", text_field,
        "--eval-examples", "200",
        "--sequence-length", str(sequence_length),
        "--gradient-accumulation-steps", "1",
        "--per-device-train-batch-size", "1",
        "--per-device-eval-batch-size", "1",
        "--max-eval-batches", "20",
        "--logging-steps", "10",
        "--save-steps", str(save_steps),
        "--learning-rate", "0.0003",
        "--weight-decay", "0.1",
        "--seed", "42",
    ]


def run_pretraining(
    *,
    repo_root: Path,
    output_dir: Path,
    report_path: Path,
    source_revision: str,
    target_supervised_tokens: int = 10_000,
    sequence_length: int = 128,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    train_split: str = "train",
    validation_split: str = "validation",
    text_field: str = "text",
) -> dict[str, Any]:
    """Run one small native-TPU pretraining job and save its evidence report."""
    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_training_command(
        repo_root=repo_root,
        output_dir=output_dir,
        target_supervised_tokens=target_supervised_tokens,
        sequence_length=sequence_length,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        train_split=train_split,
        validation_split=validation_split,
        text_field=text_field,
    )

    inspect_profile = runpy.run_path(
        str(repo_root / "examples/torchtpu_t1_install_preflight.py")
    )["inspect_profile"]
    assess_profile = runpy.run_path(
        str(repo_root / "src/trainlm/runtime/torchtpu_trainer_preflight.py")
    )["assess_trainer_profile"]
    preflight = assess_profile(inspect_profile(repo_root), require_datasets=True)
    print("package_preflight=" + json.dumps(preflight, sort_keys=True), flush=True)
    # Keep the child-process boundary: TPU runtime registration and teardown
    # should not contaminate the notebook kernel between runs.
    run_process = runpy.run_path(
        str(repo_root / "src/trainlm/runtime/torchtpu_probe_process.py")
    )["run_probe_process"]
    if not preflight["proceed"]:
        report = {
            "schema_version": 1,
            "scope": "single_device_native_scratch_pretraining",
            "status": "blocked",
            "source_revision": source_revision,
            "preflight": preflight,
            "checks": {
                "train": "not_run", "export": "not_run",
                "reload_generate": "not_run", "compile": "not_run",
                "resume": "not_run", "exact_weight_match": "not_run",
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"Preflight blocked the run; report saved to {report_path}.")

    print(
        "training_started; the isolated TPU run may take time; "
        f"local logs will be saved under {report_path.parent}",
        flush=True,
    )
    result = run_process(command, cwd=repo_root, timeout=3600)
    run_dir = report_path.parent
    (run_dir / "stdout.log").write_text(result["stdout"], encoding="utf-8")
    (run_dir / "stderr.log").write_text(result["stderr"], encoding="utf-8")
    build_trainer_report = runpy.run_path(
        str(repo_root / "src/trainlm/runtime/torchtpu_trainer_evidence.py")
    )["build_trainer_report"]
    report = build_trainer_report(
        result, source_revision=source_revision,
        compile_requested=False, resume_requested=False,
    )
    report["scope"] = "single_device_native_scratch_pretraining"
    report["preflight"] = preflight
    report["limitations"] = ["experimental_single_device", "does_not_certify_performance"]
    if (report["status"] == "passed" and
            (report.get("model") is None
             or report["model"].get("initialization") != "random_from_config"
             or report.get("dataset") is None)):
        report["status"] = "failed"
        report["issues"] = sorted(set(report.get("issues", []) + [
            "scratch_provenance_missing"
        ]))
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"shareable_report={report_path}")
    print(f"local_artifacts_and_logs={run_dir}")
    if report["status"] != "passed":
        raise RuntimeError(f"Pretraining failed; inspect {run_dir} before retrying.")
    return report
