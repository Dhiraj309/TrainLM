"""Sanitize evidence from the isolated native trainer smoke process.

This module has no Torch import and can be loaded by a notebook before TPU
initialization. Raw child logs are deliberately excluded from its result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import runpy
from typing import Any, Mapping


_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9._+\-]{1,96}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}\Z")
_REPO_ID = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}\Z")
_CHECKS = ("train", "export", "compile", "resume", "exact_weight_match")
_MARKERS = {
    "train": "train=PASS", "export": "export=PASS",
    "compile": "compile=PASS", "resume": "resume=PASS",
    "exact_weight_match": "exact_weight_match=PASS",
}
_PACKAGES = ("torch", "torch_tpu", "libtpu", "transformers", "tokenizers")
_FATAL_MARKERS = {"core_dump", "shutdown_hang", "python_fatal"}


def build_trainer_report(
    process: Mapping[str, Any], *, source_revision: str,
    compile_requested: bool, resume_requested: bool,
) -> dict[str, Any]:
    """Require matching process, phase, and device evidence before a pass."""
    if not _SHA.fullmatch(source_revision):
        raise ValueError("source_revision must be a full Git SHA")
    if type(compile_requested) is not bool or type(resume_requested) is not bool:
        raise ValueError("requested modes must be booleans")
    if compile_requested and resume_requested:
        raise ValueError("compile and resume checks must run separately")

    raw_status = process.get("status")
    process_status = raw_status if isinstance(raw_status, str) and raw_status in {
        "completed", "failed", "timeout", "interrupted",
    } else "failed"
    exit_code = process.get("exit_code")
    if type(exit_code) is not int:
        exit_code = None
    error_type = process.get("error_type")
    if not isinstance(error_type, str) or not _SAFE_TEXT.fullmatch(error_type):
        error_type = None
    raw_fatal_markers = process.get("fatal_markers")
    invalid_fatal_markers = (not isinstance(raw_fatal_markers, list)
                             or any(not isinstance(marker, str)
                                    or marker not in _FATAL_MARKERS
                                    for marker in raw_fatal_markers))
    fatal_markers = (sorted(set(raw_fatal_markers))
                     if not invalid_fatal_markers else [])
    cleanup_completed = process.get("cleanup_completed")
    if type(cleanup_completed) is not bool:
        cleanup_completed = None
    stdout = process.get("stdout")
    if not isinstance(stdout, str):
        stdout = ""
    lines = stdout.splitlines()
    summaries = [line[len("trainer_summary="):] for line in lines
                 if line.startswith("trainer_summary=")]
    issues = []
    summary = None
    if len(summaries) != 1 or len(summaries[0]) > 8192:
        issues.append("missing_or_duplicate_summary")
    else:
        try:
            summary = json.loads(summaries[0])
        except (ValueError, TypeError):
            issues.append("invalid_summary")

    checks = {name: "unknown" for name in _CHECKS}
    device = None
    precision = None
    training = None
    model = None
    dataset = None
    metrics = None
    packages = {name: None for name in _PACKAGES}
    if (isinstance(summary, dict)
            and type(summary.get("schema_version")) is int
            and summary["schema_version"] == 1):
        raw_precision = summary.get("precision", "fp32")
        if isinstance(raw_precision, str) and raw_precision in {"fp32", "bf16"}:
            precision = raw_precision
        else:
            issues.append("invalid_precision")
        raw_training = summary.get("training")
        if raw_training is not None:
            if isinstance(raw_training, dict):
                lr = raw_training.get("learning_rate")
                decay = raw_training.get("weight_decay")
                schedule = raw_training.get("lr_scheduler_type")
                gradient_checkpointing = raw_training.get(
                    "gradient_checkpointing", False
                )
            else:
                lr = decay = schedule = gradient_checkpointing = None
            integer_fields = (
                "max_steps", "warmup_steps", "gradient_accumulation_steps", "seed"
            )
            batch_fields = (
                "per_device_train_batch_size", "per_device_eval_batch_size",
                "sequence_length",
            )
            if isinstance(raw_training, dict):
                batch_fields_present = [name in raw_training for name in batch_fields]
            else:
                batch_fields_present = []
            logging_steps_present = (
                isinstance(raw_training, dict) and "logging_steps" in raw_training
            )
            logging_steps_valid = (
                not logging_steps_present
                or (type(raw_training.get("logging_steps")) is int
                    and raw_training["logging_steps"] > 0)
            )
            batch_fields_valid = (
                not any(batch_fields_present)
                or (
                    all(batch_fields_present)
                    and all(
                        type(raw_training.get(name)) is int
                        and raw_training[name] > 0
                        for name in batch_fields
                    )
                )
            )
            integers_valid = isinstance(raw_training, dict) and all(
                type(raw_training.get(name)) is int
                and raw_training[name] >= (
                    0 if name in {"warmup_steps", "seed"} else 1
                )
                for name in integer_fields
            )
            try:
                safe_lr = float(lr)
                safe_decay = float(decay)
                numbers_valid = (
                    type(lr) in {int, float}
                    and math.isfinite(safe_lr)
                    and safe_lr > 0
                    and type(decay) in {int, float}
                    and math.isfinite(safe_decay)
                    and safe_decay >= 0
                )
            except (OverflowError, TypeError, ValueError):
                safe_lr = safe_decay = 0.0
                numbers_valid = False
            nominal_tokens_per_update = None
            nominal_supervised_tokens_per_update = None
            if batch_fields_valid and integers_valid and all(batch_fields_present):
                nominal_tokens_per_update = (
                    raw_training["sequence_length"]
                    * raw_training["per_device_train_batch_size"]
                    * raw_training["gradient_accumulation_steps"]
                )
                nominal_supervised_tokens_per_update = (
                    (raw_training["sequence_length"] - 1)
                    * raw_training["per_device_train_batch_size"]
                    * raw_training["gradient_accumulation_steps"]
                )
                reported_nominal = raw_training.get("nominal_tokens_per_update")
                if (reported_nominal is not None
                        and (type(reported_nominal) is not int
                             or reported_nominal != nominal_tokens_per_update)):
                    batch_fields_valid = False
                reported_supervised_nominal = raw_training.get(
                    "nominal_supervised_tokens_per_update"
                )
                if (reported_supervised_nominal is not None
                        and (type(reported_supervised_nominal) is not int
                             or reported_supervised_nominal
                             != nominal_supervised_tokens_per_update)):
                    batch_fields_valid = False
            if isinstance(raw_training, dict):
                supervised_tokens_seen = raw_training.get("supervised_tokens_seen")
                target_supervised_tokens = raw_training.get(
                    "target_supervised_tokens"
                )
            else:
                supervised_tokens_seen = target_supervised_tokens = None
            if (supervised_tokens_seen is not None
                    and (type(supervised_tokens_seen) is not int
                         or supervised_tokens_seen < 1)):
                batch_fields_valid = False
            if target_supervised_tokens is not None:
                microbatch_tokens = (
                    (raw_training["sequence_length"] - 1)
                    * raw_training["per_device_train_batch_size"]
                    if all(batch_fields_present) and batch_fields_valid else 0
                )
                if (type(target_supervised_tokens) is not int
                        or target_supervised_tokens < 1
                        or type(supervised_tokens_seen) is not int
                        or supervised_tokens_seen < target_supervised_tokens
                        or microbatch_tokens < 1
                        or supervised_tokens_seen - target_supervised_tokens
                        >= microbatch_tokens):
                    batch_fields_valid = False
            if (not integers_valid or not numbers_valid
                    or not batch_fields_valid
                    or not logging_steps_valid
                    or not isinstance(schedule, str)
                    or schedule not in {"constant", "cosine"}
                    or type(gradient_checkpointing) is not bool):
                issues.append("invalid_training_config")
            else:
                training = {name: raw_training[name] for name in integer_fields}
                training.update({
                    "learning_rate": safe_lr,
                    "weight_decay": safe_decay,
                    "lr_scheduler_type": schedule,
                    "gradient_checkpointing": gradient_checkpointing,
                })
                if nominal_tokens_per_update is not None:
                    training.update({
                        name: raw_training[name] for name in batch_fields
                    })
                    training["nominal_tokens_per_update"] = nominal_tokens_per_update
                    training["nominal_supervised_tokens_per_update"] = (
                        nominal_supervised_tokens_per_update
                    )
                if supervised_tokens_seen is not None:
                    training["supervised_tokens_seen"] = supervised_tokens_seen
                if target_supervised_tokens is not None:
                    training["target_supervised_tokens"] = target_supervised_tokens
                if logging_steps_present:
                    training["logging_steps"] = raw_training["logging_steps"]
        normalize = runpy.run_path(str(Path(__file__).with_name(
            "torchtpu_discovery.py")))["normalize_worker_report"]
        device = normalize(summary.get("device"))
        if (device.get("status") != "ready"
                or device.get("device_type") != "tpu"
                or device.get("device_count_visible") != 1):
            issues.append("invalid_device")
            device = None
        raw_model = summary.get("model")
        if raw_model is not None:
            if not isinstance(raw_model, dict):
                issues.append("invalid_model_revision")
            else:
                initialization = raw_model.get("initialization")
                model_type = raw_model.get("model_type")
                config_only = (
                    initialization == "random_from_config"
                    and raw_model.get("revision") is None
                    and isinstance(model_type, str)
                    and _SAFE_TEXT.fullmatch(model_type)
                    and raw_model.get("id") == model_type
                )
                if config_only:
                    model = {
                        "id": model_type,
                        "revision": None,
                        "initialization": initialization,
                        "model_type": model_type,
                    }
                    config = raw_model.get("config")
                    config_fields = (
                        "vocab_size", "n_positions", "n_embd", "n_layer", "n_head"
                    )
                    if (
                        not isinstance(config, dict)
                        or any(
                            type(config.get(name)) is not int or config[name] < 1
                            for name in config_fields
                        )
                        or config["n_embd"] % config["n_head"]
                    ):
                        issues.append("invalid_model_config")
                        model = None
                    else:
                        model["config"] = {
                            name: config[name] for name in config_fields
                        }
                        source = raw_model.get("config_source")
                        if isinstance(source, str) and _SAFE_TEXT.fullmatch(source):
                            model["config_source"] = source
                        else:
                            issues.append("invalid_model_config")
                            model = None
                elif (
                    not isinstance(raw_model.get("id"), str)
                    or not _MODEL_ID.fullmatch(raw_model["id"])
                    or not isinstance(raw_model.get("revision"), str)
                    or not _SHA.fullmatch(raw_model["revision"])
                ):
                    issues.append("invalid_model_revision")
                else:
                    model = {
                        "id": raw_model["id"],
                        "revision": raw_model["revision"],
                    }
                    if initialization is not None:
                        if initialization not in {"pretrained", "random_from_config"}:
                            issues.append("invalid_model_initialization")
                            model = None
                        else:
                            model["initialization"] = initialization
                parameter_count = raw_model.get("parameter_count")
                if parameter_count is not None:
                    if type(parameter_count) is not int or parameter_count < 1:
                        issues.append("invalid_model_parameter_count")
                        model = None
                    elif model is not None:
                        model["parameter_count"] = parameter_count
        raw_dataset = summary.get("dataset")
        if raw_dataset is not None:
            if (not isinstance(raw_dataset, dict)
                    or not isinstance(raw_dataset.get("id"), str)
                    or not _REPO_ID.fullmatch(raw_dataset["id"])
                    or not isinstance(raw_dataset.get("revision"), str)
                    or not _SHA.fullmatch(raw_dataset["revision"])
                    or type(raw_dataset.get("train_examples")) is not int
                    or raw_dataset["train_examples"] < 1):
                issues.append("invalid_dataset_provenance")
            else:
                dataset = {
                    "id": raw_dataset["id"],
                    "revision": raw_dataset["revision"],
                    "train_examples": raw_dataset["train_examples"],
                }
                for name in ("train_split", "validation_split", "text_field"):
                    value = raw_dataset.get(name)
                    if value is not None:
                        if not isinstance(value, str) or not _SAFE_TEXT.fullmatch(value):
                            issues.append("invalid_dataset_provenance")
                            dataset = None
                        elif dataset is not None:
                            dataset[name] = value
                validation_examples = raw_dataset.get("validation_examples")
                if validation_examples is not None:
                    if type(validation_examples) is not int or validation_examples < 1:
                        issues.append("invalid_dataset_provenance")
                        dataset = None
                    elif dataset is not None:
                        dataset["validation_examples"] = validation_examples
                for name in (
                    "train_tokens", "validation_tokens",
                    "train_documents", "validation_documents",
                ):
                    value = raw_dataset.get(name)
                    if value is not None:
                        if type(value) is not int or value < 1:
                            issues.append("invalid_dataset_provenance")
                            dataset = None
                        elif dataset is not None:
                            dataset[name] = value
                for name in ("train_shard_sha256", "validation_shard_sha256"):
                    value = raw_dataset.get(name)
                    if value is not None:
                        if not isinstance(value, str) or not _SHA256.fullmatch(value):
                            issues.append("invalid_dataset_provenance")
                            dataset = None
                        elif dataset is not None:
                            dataset[name] = value
                tokenizer = raw_dataset.get("tokenizer")
                if tokenizer is not None:
                    if (
                        not isinstance(tokenizer, dict)
                        or tokenizer.get("kind") != "byte_level_bpe_from_training_split"
                        or type(tokenizer.get("training_documents")) is not int
                        or tokenizer["training_documents"] < 1
                        or type(tokenizer.get("vocab_size")) is not int
                        or tokenizer["vocab_size"] < 260
                    ):
                        issues.append("invalid_dataset_provenance")
                        dataset = None
                    elif dataset is not None:
                        dataset["tokenizer"] = {
                            "kind": tokenizer["kind"],
                            "training_documents": tokenizer["training_documents"],
                            "vocab_size": tokenizer["vocab_size"],
                        }
        raw_metrics = summary.get("metrics")
        if raw_metrics is not None:
            if not isinstance(raw_metrics, dict):
                issues.append("invalid_metrics")
            else:
                safe_metrics = {}
                for name in ("final_train_loss", "eval_loss", "eval_perplexity"):
                    value = raw_metrics.get(name)
                    if value is None:
                        continue
                    if (type(value) not in {int, float}
                            or not math.isfinite(value)):
                        issues.append("invalid_metrics")
                        safe_metrics = {}
                        break
                    safe_metrics[name] = float(value)
                if safe_metrics or not raw_metrics:
                    metrics = safe_metrics
        raw_packages = summary.get("packages")
        if not isinstance(raw_packages, dict):
            issues.append("invalid_packages")
        else:
            for name in _PACKAGES:
                version = raw_packages.get(name)
                if not isinstance(version, str) or not _SAFE_TEXT.fullmatch(version):
                    issues.append("invalid_packages")
                    break
                packages[name] = version
        raw_checks = summary.get("checks")
        if not isinstance(raw_checks, dict):
            issues.append("invalid_checks")
        else:
            for name in _CHECKS:
                value = raw_checks.get(name)
                if value not in {"passed", "not_run"}:
                    issues.append("invalid_checks")
                    break
                checks[name] = value
    elif summary is not None:
        issues.append("invalid_summary")

    expected = {
        "train": "passed", "export": "passed",
        "compile": "passed" if compile_requested else "not_run",
        "resume": "passed" if resume_requested else "not_run",
        "exact_weight_match": "passed" if resume_requested else "not_run",
    }
    if checks != expected:
        issues.append("phase_mismatch")
    required_markers = {_MARKERS[name] for name in _CHECKS
                        if expected[name] == "passed"}
    required_markers.add("result=trainer_smoke_passed")
    if not required_markers.issubset(set(lines)):
        issues.append("missing_markers")
    if (process_status != "completed" or exit_code != 0 or fatal_markers
            or invalid_fatal_markers):
        issues.append("process_failed")
    return {
        "schema_version": 1,
        "scope": "single_device_native_trainer_smoke",
        "status": "passed" if not issues else "failed",
        "source_revision": source_revision,
        "requested": {"compile": compile_requested, "resume": resume_requested},
        "process": {"status": process_status, "exit_code": exit_code,
                    "error_type": error_type, "fatal_markers": fatal_markers,
                    "cleanup_completed": cleanup_completed},
        "device": device,
        "precision": precision,
        "training": training,
        "model": model,
        "dataset": dataset,
        "metrics": metrics,
        "packages": packages,
        "checks": checks,
        "issues": sorted(set(issues)),
        "limitations": ["smoke_only", "does_not_certify_performance"],
    }
