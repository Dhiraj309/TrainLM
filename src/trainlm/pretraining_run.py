"""High-level from-scratch training workflow for user applications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from typing import Any
import zipfile

from ._notebook_resume import build_resume_report
from .pretraining import TrainLMPretrainingConfig


_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class TrainLMPretrainingError(RuntimeError):
    """A failed high-level run whose machine-readable report was saved."""

    def __init__(
        self, report_path: Path, error_type: str, report: dict[str, Any]
    ) -> None:
        self.report_path = report_path
        self.error_type = error_type
        self.report = report
        details = self._package_preflight_details(report)
        prefix = f"Pretraining failed ({error_type})"
        if details:
            prefix += f": {details}"
        super().__init__(
            f"{prefix}. Full run report: {report_path}."
        )

    @staticmethod
    def _package_preflight_details(report: dict[str, Any]) -> str | None:
        """Summarize safe package facts from a blocked preflight report."""
        preflight = report.get("preflight")
        profile = (
            preflight.get("package_profile")
            if isinstance(preflight, dict)
            else None
        )
        requirements = (
            profile.get("requirements") if isinstance(profile, dict) else None
        )
        if not isinstance(requirements, list):
            return None

        failures = []
        for item in requirements:
            if not isinstance(item, dict) or item.get("status") == "matched":
                continue
            package = item.get("package")
            required = item.get("required")
            installed = item.get("installed")
            status = item.get("status")
            if not all(
                isinstance(value, str) and value
                for value in (package, required, status)
            ):
                continue
            observed = (
                installed
                if isinstance(installed, str) and installed
                else "not installed"
            )
            failures.append(
                f"{package} requires {required}; found {observed} ({status})"
            )
        return "; ".join(failures) if failures else None


@dataclass(frozen=True, slots=True)
class TrainLMPretrainingResult:
    """Paths and summary data produced by :func:`run_pretraining`."""

    report: dict[str, Any]
    report_path: Path
    export_path: Path | None

    @property
    def status(self) -> str:
        return str(self.report.get("status", "unknown"))

    @property
    def training(self) -> dict[str, Any]:
        value = self.report.get("training", {})
        return value if isinstance(value, dict) else {}

    @property
    def checkpoint_path(self) -> Path | None:
        value = (self.report.get("checkpoint") or {}).get("latest_path")
        return Path(value) if isinstance(value, str) and value else None

    @property
    def parameter_count(self) -> int | None:
        """Return the instantiated model's total parameter count, if recorded."""
        model = self.report.get("model", {})
        value = model.get("total_parameters") if isinstance(model, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @property
    def metrics(self) -> dict[str, float]:
        training = self.training
        return {
            key: float(training[key])
            for key in ("train_loss", "eval_loss")
            if isinstance(training.get(key), (int, float))
        }

    @property
    def best_model_selection(self) -> dict[str, Any] | None:
        """Return the selected best checkpoint details when configured."""
        value = self.report.get("best_model_selection")
        return value if isinstance(value, dict) else None

    def summary(self) -> str:
        training = self.training
        model = self.report.get("model", {})
        accelerator = self.report.get("accelerator", {})
        parameters = (
            f"{self.parameter_count:,}"
            if self.parameter_count is not None
            else "unknown"
        )
        best = self.best_model_selection
        best_step = best.get("step") if best is not None else None
        best_summary = f" best_step={best_step}" if best_step is not None else ""
        stop_summary = (
            " early_stopping_triggered=True"
            if training.get("early_stopping_triggered")
            else ""
        )
        return (
            f"status={self.status} "
            f"model={model.get('model_class') or model.get('requested_model_type', 'unknown')} "
            f"parameters={parameters} "
            f"steps={training.get('step', 'not_run')} "
            f"tokens={training.get('supervised_tokens', 'not_run')}/"
            f"{training.get('target_supervised_tokens', 'unknown')} "
            f"eval_loss={training.get('eval_loss', 'not_run')} "
            f"accelerator={accelerator.get('backend', accelerator.get('requested', 'unknown'))}"
            f"{best_summary}"
            f"{stop_summary}"
        )


def run_pretraining(
    config: TrainLMPretrainingConfig,
    *,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
    resume_from_checkpoint: str | Path | None = None,
    extend_training_budget: bool = False,
    show_progress: bool = False,
    source_revision: str | None = None,
    report_path: str | Path | None = None,
    export_path: str | Path | None = None,
) -> TrainLMPretrainingResult:
    """Prepare, train, evaluate, export, and report one scratch run.

    This is the convenient end-to-end path. Users who need a custom lifecycle
    can continue to construct :class:`TrainLMTrainer` directly and call its
    ``train``, ``evaluate``, and ``save_model`` methods. For raw HF Dataset
    objects, set ``dataset_source="provided"`` and pass both splits here. Set
    ``data_format="tokens"`` for ``input_ids`` plus an optional
    ``attention_mask``; TrainLM generates causal labels from the IDs.

    The report is written even when model preparation, training, evaluation,
    or export fails. In that case :class:`TrainLMPretrainingError` identifies
    the report path and preserves the original exception as its cause.
    Set ``show_progress=True`` to print one step, token, loss, and learning
    rate snapshot per optimizer step when its loss is available, at the
    configured logging cadence.
    """
    if not isinstance(config, TrainLMPretrainingConfig):
        raise TypeError("config must be a TrainLMPretrainingConfig instance.")
    if not isinstance(extend_training_budget, bool):
        raise TypeError("extend_training_budget must be a boolean.")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean.")
    if source_revision is not None and (
        not isinstance(source_revision, str)
        or _SOURCE_REVISION.fullmatch(source_revision) is None
    ):
        raise ValueError("source_revision must be a lowercase 40-character commit SHA.")

    output_dir = Path(config.output_dir)
    report_file = Path(report_path) if report_path is not None else output_dir / "report.json"
    export_dir = Path(export_path) if export_path is not None else output_dir / "baseline-export"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report = _initial_report(config, source_revision=source_revision)
    trainer = None
    completed_export: Path | None = None

    try:
        # Import lazily so configuration and report inspection remain usable
        # without importing the training engine.
        from .api import TrainLMTrainer

        trainer_options: dict[str, Any] = {
            "extend_training_budget": extend_training_budget,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
        }
        if show_progress:
            trainer_options["callbacks"] = [_make_training_progress_callback(config)]
        trainer = TrainLMTrainer.from_scratch(config, **trainer_options)
        _capture_local_text_sources(report, config)
        _capture_effective_configuration(report, trainer)
        _capture_runtime(report, trainer)

        state = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        tokens_seen = int(state.tokens_seen)
        checkpoint = _checkpoint_report(trainer)
        best_model_selection = trainer.best_model_selection
        early_stopping = (
            best_model_selection.get("early_stopping")
            if isinstance(best_model_selection, Mapping)
            else None
        )
        early_stopping_triggered = (
            tokens_seen < config.max_tokens
            and config.early_stopping_patience is not None
            and isinstance(early_stopping, Mapping)
            and early_stopping.get("evaluations_without_improvement", 0)
            >= config.early_stopping_patience
        )
        evaluation = trainer.evaluate()
        completed_export = Path(trainer.save_model(export_dir))
        _capture_export(report, completed_export)

        train_loss = float(state.loss) if state.loss is not None else None
        eval_loss = float(evaluation["eval_loss"])
        report["training"] = {
            "extend_training_budget": extend_training_budget,
            "resume_from_checkpoint": (
                str(resume_from_checkpoint)
                if resume_from_checkpoint is not None else None
            ),
            "step": int(state.step),
            "supervised_tokens": tokens_seen,
            "target_supervised_tokens": config.max_tokens,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "early_stopping_triggered": early_stopping_triggered,
        }
        report["best_model_selection"] = best_model_selection
        report["checkpoint"] = checkpoint
        report["model_export"] = str(completed_export)
        report["status"] = (
            "passed"
            if (tokens_seen >= config.max_tokens or early_stopping_triggered)
            and train_loss is not None
            and math.isfinite(train_loss)
            and math.isfinite(eval_loss)
            else "failed"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        package_profile = getattr(exc, "profile", None)
        if (isinstance(package_profile, dict)
                and package_profile.get("scope") == "installed_metadata_only"):
            report["preflight"] = {"package_profile": package_profile}
        report["checkpoint"] = _checkpoint_report(trainer)
        report["best_model_selection"] = (
            trainer.best_model_selection if trainer is not None else None
        )
        _capture_resume(report, config, trainer)
        _write_report(report_file, report)
        raise TrainLMPretrainingError(
            report_file, type(exc).__name__, report
        ) from exc

    _capture_resume(report, config, trainer)
    _write_report(report_file, report)
    return TrainLMPretrainingResult(
        report=report,
        report_path=report_file,
        export_path=completed_export,
    )


def _make_training_progress_callback(
    config: TrainLMPretrainingConfig,
) -> Any:
    """Print one sparse console snapshot per step with a materialized loss."""
    from .training.callback import TrainerCallback

    tokens_per_step = (
        (config.sequence_length - 1)
        * config.per_device_train_batch_size
        * config.gradient_accumulation_steps
    )
    total_steps = math.ceil(config.max_tokens / tokens_per_step)
    reported_steps: set[int] = set()

    class _ProgressCallback(TrainerCallback):
        def on_metrics(
            self,
            state: Any,
            control: Any,
            metrics: Mapping[str, float],
        ) -> None:
            step = int(metrics.get("step", state.step))
            tokens_seen = int(metrics.get("tokens_seen", state.tokens_seen))
            loss = metrics.get("loss")
            if loss is None or step in reported_steps:
                return
            reported_steps.add(step)
            learning_rate = metrics.get("learning_rate", state.learning_rate)
            print(
                f"train step {step}/{total_steps} | "
                f"tokens {tokens_seen:,}/{config.max_tokens:,} | "
                f"loss {loss:.4f} | lr {learning_rate:.3g}"
            )

    return _ProgressCallback()


def _initial_report(
    config: TrainLMPretrainingConfig, *, source_revision: str | None
) -> dict[str, Any]:
    packages = {}
    for distribution in ("torch", "transformers", "torch-tpu", "libtpu", "datasets", "tokenizers"):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None

    uses_hf_tokenizer = config.tokenizer_mode == "auto" and (
        config.tokenizer_name_or_path is not None
        or config.model_name_or_path is not None
    )
    return {
        "schema_version": 6,
        "scope": (
            "experimental_single_device_tokenized_pretraining"
            if config.data_format == "tokens"
            else "experimental_single_device_raw_text_pretraining"
        ),
        "status": "not_started",
        "source_revision": source_revision,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "dataset_config_name": config.dataset_config_name,
        "device": config.accelerator,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "data": {
            "source_type": (
                "huggingface_dataset_objects"
                if config.dataset_source == "provided"
                else (
                    "local_text_files"
                    if config.train_file is not None
                    else "huggingface_hub"
                )
            ),
            "dataset_source": (
                "local_files"
                if config.train_file is not None
                else config.dataset_source
            ),
            "data_format": config.data_format,
            "causal_target_policy": (
                "TrainLM copies input_ids to labels and applies one next-token shift."
            ),
            "tokenizer_dataset_match": (
                "not_verified_by_trainlm"
                if config.data_format == "tokens"
                else None
            ),
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "dataset_revision_requested": config.dataset_revision,
            "dataset_revision_resolved": None,
            "dataset_config_name": config.dataset_config_name,
            "train_split": config.train_split,
            "validation_split": config.validation_split,
            "text_field": config.text_field,
            "max_train_examples": config.max_train_examples,
            "max_eval_examples": config.max_eval_examples,
            "local_files": (
                {
                    "train": str(config.train_file),
                    "validation": str(config.validation_file),
                }
                if config.train_file is not None
                else None
            ),
        },
        "model": {
            "requested_model_type": config.model_type,
            "requested_config_overrides": dict(config.model_config_overrides),
            "requested_source": (
                str(config.model_name_or_path)
                if config.model_name_or_path is not None
                else None
            ),
            "requested_revision": config.model_revision,
            "requested_initialization": config.model_initialization,
        },
        "tokenizer": {
            "algorithm": (
                "huggingface"
                if uses_hf_tokenizer
                else "byte_level_bpe"
            ),
            "requested_source": (
                str(config.tokenizer_name_or_path or config.model_name_or_path)
                if uses_hf_tokenizer
                else None
            ),
            "requested_revision": (
                (
                    config.tokenizer_revision
                    if config.tokenizer_revision is not None
                    else (
                        config.model_revision
                        if config.tokenizer_name_or_path is None
                        else None
                    )
                )
                if uses_hf_tokenizer
                else None
            ),
            "training_split": config.train_split,
            "requested_vocab_size": (
                config.tokenizer_vocab_size
                if not uses_hf_tokenizer
                else None
            ),
        },
        "training_config": {
            "requested": {
                "target_supervised_tokens": config.max_tokens,
                "sequence_length": config.sequence_length,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "per_device_eval_batch_size": config.per_device_eval_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "betas": [float(value) for value in config.betas],
                "eps": config.eps,
                "max_grad_norm": config.max_grad_norm,
                "lr_scheduler_type": config.lr_scheduler_type,
                "warmup_steps": config.warmup_steps,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "bf16": config.bf16,
                "torch_compile": config.torch_compile,
                "logging_steps": config.logging_steps,
                "save_steps": config.save_steps,
                "eval_steps": config.eval_steps,
                "max_eval_batches": config.max_eval_batches,
                "load_best_model_at_end": config.load_best_model_at_end,
                "early_stopping_patience": config.early_stopping_patience,
                "early_stopping_threshold": config.early_stopping_threshold,
                "seed": config.seed,
                "save_total_limit": config.save_total_limit,
            }
        },
        "accelerator": {"requested": config.accelerator},
        "checkpoint": {"latest_path": None},
        "best_model_selection": None,
        "limitations": ["does_not_certify_performance"],
    }


def _capture_local_text_sources(
    report: dict[str, Any], config: TrainLMPretrainingConfig
) -> None:
    """Record immutable Hub identity or local hashes from the prepared cache."""
    cache_path = Path(config.output_dir) / "preparation_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    inputs = cache.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise RuntimeError("The prepared-data cache has no source metadata.")
    if config.dataset_id is not None:
        report["data"]["dataset_revision_requested"] = config.dataset_revision
        report["data"]["dataset_revision_resolved"] = inputs.get(
            "dataset_revision"
        )
        report["data"]["dataset_config_name"] = inputs.get(
            "dataset_config_name", config.dataset_config_name
        )
    if config.dataset_source == "provided":
        datasets = inputs.get("provided_datasets")
        if not isinstance(datasets, Mapping) or set(datasets) != {
            "train", "validation"
        }:
            raise RuntimeError(
                "The prepared-data cache does not record both provided Dataset inputs."
            )
        report["data"]["provided_datasets"] = datasets
        if config.data_format == "tokens":
            tokenizer_source = inputs.get("tokenizer_source")
            if not isinstance(tokenizer_source, Mapping):
                raise RuntimeError(
                    "The prepared token Dataset cache does not record tokenizer provenance."
                )
            report["data"]["tokenizer_provenance"] = {
                name: tokenizer_source.get(name)
                for name in (
                    "requested_source",
                    "requested_revision",
                    "resolved_revision",
                    "local_source",
                    "mapping_sha256",
                )
            }
    if config.train_file is not None and config.validation_file is not None:
        files = inputs.get("local_text_files")
        if not isinstance(files, Mapping) or set(files) != {"train", "validation"}:
            raise RuntimeError(
                "The prepared-data cache does not record both local text files."
            )
        report["data"]["local_files"] = files


def _capture_effective_configuration(report: dict[str, Any], trainer: Any) -> None:
    model = trainer.model
    tokenizer = trainer.processing_class
    parameters = list(model.parameters())
    report["model"].update({
        "model_type": model.config.model_type,
        "model_class": type(model).__name__,
        "vocab_size": int(model.config.vocab_size),
        "tokenizer_vocab_size": len(tokenizer),
        "total_parameters": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameters": sum(
            int(parameter.numel())
            for parameter in parameters
            if parameter.requires_grad
        ),
    })
    model_source = getattr(model, "_trainlm_model_source_metadata", None)
    if isinstance(model_source, Mapping):
        report["model"]["source"] = model_source
        report["model"]["initialization"] = report["model"][
            "requested_initialization"
        ]
    report["tokenizer"].update({
        "class": type(tokenizer).__name__,
        "base_vocab_size": int(tokenizer.vocab_size),
        "vocab_size": len(tokenizer),
        "special_tokens": {
            name: {
                "token": (
                    str(getattr(tokenizer, f"{name}_token"))
                    if getattr(tokenizer, f"{name}_token", None) is not None
                    else None
                ),
                "id": (
                    int(getattr(tokenizer, f"{name}_token_id"))
                    if getattr(tokenizer, f"{name}_token_id", None) is not None
                    else None
                ),
            }
            for name in ("unk", "pad", "bos", "eos")
        },
    })
    tokenizer_source = getattr(
        model, "_trainlm_tokenizer_source_metadata", None
    )
    if isinstance(tokenizer_source, Mapping):
        report["tokenizer"]["source"] = tokenizer_source
        report["tokenizer"]["token_id_capacity"] = max(
            len(tokenizer), max(tokenizer.get_vocab().values()) + 1
        )

    args = trainer.args
    optimizer = getattr(trainer, "optimizer", None)
    optimizer_report = None
    if optimizer is not None:
        optimizer_report = {
            "class": type(optimizer).__name__,
            "parameter_groups": [
                {
                    "learning_rate": float(group["lr"]),
                    "betas": [float(value) for value in group["betas"]],
                    "eps": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "fused": bool(group.get("fused", False)),
                    "parameter_count": sum(
                        int(parameter.numel()) for parameter in group["params"]
                    ),
                }
                for group in optimizer.param_groups
            ],
        }
    scheduler = getattr(trainer, "scheduler", None)
    report["training_config"]["effective"] = {
        "accelerator": args.accelerator,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "sequence_length": args.sequence_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "betas": [float(value) for value in args.betas],
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "torch_compile": args.torch_compile,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": (
            (args.metric_for_best_model or "eval_loss")
            if args.load_best_model_at_end
            else None
        ),
        "greater_is_better": (
            (
                args.greater_is_better
                if args.greater_is_better is not None
                else False
            )
            if args.load_best_model_at_end
            else None
        ),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "max_eval_batches": args.max_eval_batches,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "seed": args.seed,
        "report_to": args.report_to if isinstance(args.report_to, str) else list(args.report_to),
        "logging_verbosity": args.logging_verbosity,
        "loss_implementation": args.loss_implementation,
        "logits_chunk_size": args.logits_chunk_size,
        "max_grad_norm": args.max_grad_norm,
        "optimizer": optimizer_report,
        "scheduler_class": type(scheduler).__name__ if scheduler is not None else None,
    }


def _capture_runtime(report: dict[str, Any], trainer: Any) -> None:
    accelerator = report["accelerator"]
    try:
        diagnostics = trainer.runtime.diagnostics()
        values = diagnostics.values
        accelerator.update({
            "backend": diagnostics.backend,
            "device_type": diagnostics.device_type,
            "precision": diagnostics.precision,
            "world_size": diagnostics.world_size,
            "rank": diagnostics.rank,
            "local_rank": diagnostics.local_rank,
            "runtime_status": values.get("native_tpu_status"),
            "device_count_visible": values.get("device_count_visible"),
            "torch_compile_enabled": values.get("compile_enabled"),
            "torch_tpu_version": values.get("torch_tpu_version"),
            "libtpu_version": values.get("libtpu_version"),
        })
    except Exception as exc:
        accelerator["diagnostics_error_type"] = type(exc).__name__

    try:
        import torch
        from .runtime.torchtpu_discovery import discover_tpu, normalize_worker_report

        hardware = normalize_worker_report(
            discover_tpu(torch, verify_allocation=False, include_properties=True)
        )
    except Exception as exc:
        hardware = {"status": "query_failed", "error_type": type(exc).__name__}
    accelerator.update({
        "discovery_error_type": hardware.get("error_type"),
        "properties_source": hardware.get("source"),
        "properties_query_status": hardware.get("status"),
        "properties_allocation": hardware.get("allocation"),
        "device_name": hardware.get("device_name"),
        "topology": hardware.get("topology"),
        "hbm_bytes_per_device": hardware.get("hbm_bytes_per_device"),
        "hbm_bytes_per_chip": hardware.get("hbm_bytes_per_chip"),
        "devices_per_chip": hardware.get("devices_per_chip"),
        "properties_error_type": hardware.get("properties_error_type"),
    })
    if accelerator.get("backend") == "torchtpu":
        report["limitations"].append("experimental_single_device")


def _capture_export(report: dict[str, Any], export_dir: Path) -> None:
    model_config = export_dir / "config.json"
    if model_config.is_file():
        report["model"]["effective_config"] = json.loads(
            model_config.read_text(encoding="utf-8")
        )
    tokenizer_config = export_dir / "tokenizer_config.json"
    if tokenizer_config.is_file():
        report["tokenizer"]["effective_config"] = json.loads(
            tokenizer_config.read_text(encoding="utf-8")
        )
    tokenizer_file = export_dir / "tokenizer.json"
    if tokenizer_file.is_file():
        report["tokenizer"]["tokenizer_json_sha256"] = hashlib.sha256(
            tokenizer_file.read_bytes()
        ).hexdigest()


def _checkpoint_report(trainer: Any | None) -> dict[str, Any]:
    if trainer is None:
        return {"latest_path": None}
    try:
        latest_checkpoint = trainer.latest_checkpoint
    except Exception as exc:
        return {"latest_path": None, "lookup_error_type": type(exc).__name__}
    return {"latest_path": str(latest_checkpoint) if latest_checkpoint else None}


def _capture_resume(
    report: dict[str, Any],
    config: TrainLMPretrainingConfig,
    trainer: Any | None,
) -> None:
    try:
        report["resume"] = build_resume_report(
            config,
            trainer=trainer,
            checkpoint=report.get("checkpoint", {"latest_path": None}),
        )
        resolved_dataset_revision = report.get("data", {}).get(
            "dataset_revision_resolved"
        )
        resume_config = report["resume"].get("pretraining_config")
        if (
            isinstance(resolved_dataset_revision, str)
            and isinstance(resume_config, dict)
        ):
            resume_config["dataset_revision"] = resolved_dataset_revision
    except Exception as exc:
        report["resume"] = {
            "status": "unavailable",
            "reason": "report_generation_failed",
            "error_type": type(exc).__name__,
            "code": None,
        }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class TrainLMPretrainingResume:
    """Validated configuration and checkpoint restored from a run report."""

    config: TrainLMPretrainingConfig
    checkpoint_path: Path
    extend_training_budget: bool


def restore_pretraining_report(
    report_path: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> TrainLMPretrainingResume:
    """Validate a saved run report and prepare a safe checkpoint resume."""
    path = Path(report_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Resume report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the resume report at {path}.") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 6:
        raise ValueError("Resume needs a schema-v6 TrainLM report.")

    resume = report.get("resume")
    if not isinstance(resume, dict) or resume.get("status") != "available":
        raise ValueError(
            "This report has no published checkpoint to resume. "
            "Complete a training update first."
        )
    package_mismatches = []
    saved_packages = report.get("runtime", {}).get("packages", {})
    if isinstance(saved_packages, dict):
        for distribution, expected in saved_packages.items():
            if expected is None:
                continue
            try:
                installed = version(distribution)
            except PackageNotFoundError:
                installed = None
            if installed != expected:
                package_mismatches.append((distribution, expected, installed))
    if package_mismatches:
        details = "\n".join(
            f"  - {name}: expected {expected}, found {actual or 'not installed'}"
            for name, expected, actual in package_mismatches
        )
        raise RuntimeError(
            "The current runtime's package profile does not match report.json. "
            "Use a fresh runtime with the matching accelerator/package profile; "
            "do not replace Torch/TorchTPU packages in a live session.\n" + details
        )

    settings = resume.get("pretraining_config")
    if not isinstance(settings, dict):
        raise ValueError("The report does not contain the saved pretraining config.")
    config = TrainLMPretrainingConfig(**settings)
    output_dir = Path(config.output_dir).resolve()
    session_dir = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None else output_dir.parent
    )
    if output_dir.parent != session_dir:
        raise RuntimeError(
            "Set run_dir to the original run folder recorded by the saved config."
        )
    if path.resolve().parent not in {session_dir, output_dir}:
        raise RuntimeError("The report is outside the original run directory.")

    checkpoint_value = resume.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("The report does not contain a checkpoint path.")
    checkpoint_input = Path(checkpoint_value).expanduser()
    checkpoint = checkpoint_input.resolve()
    required_files = (
        output_dir / "run_config.json",
        output_dir / "preparation_cache.json",
    )
    if (
        checkpoint_input.is_symlink()
        or not checkpoint.is_relative_to(output_dir)
        or not checkpoint.is_dir()
        or any(not file.is_file() for file in required_files)
    ):
        raise FileNotFoundError(
            "The original checkpoint, run config, or preparation cache is missing. "
            "Restore the complete run directory before resuming."
        )
    extension = resume.get("extend_training_budget", False)
    if not isinstance(extension, bool):
        raise ValueError("The report's budget-extension flag must be a boolean.")
    return TrainLMPretrainingResume(config, checkpoint, extension)


def bundle_pretraining_artifacts(
    report_path: str | Path,
    *,
    bundle_path: str | Path | None = None,
) -> Path:
    """Bundle the run report, resume example, and valid model export."""
    path = Path(report_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"The evidence report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the evidence report at {path}.") from exc
    if not isinstance(report, dict):
        raise ValueError("The evidence report must contain a JSON object.")

    export_value = report.get("model_export")
    export_dir = Path(export_value).expanduser() if isinstance(export_value, str) else None
    config_files = ("config.json", "tokenizer_config.json")
    weights = (".safetensors", ".bin")
    export_ready = bool(
        export_dir is not None
        and export_dir.is_dir()
        and all((export_dir / name).is_file() for name in config_files)
        and any(item.is_file() and item.suffix in weights for item in export_dir.rglob("*"))
    )

    destination = (
        Path(bundle_path).expanduser()
        if bundle_path is not None
        else path.parent / "trainlm-scratch-artifact.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    resume_code = (report.get("resume") or {}).get("code")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname="report.json")
        if isinstance(resume_code, str) and resume_code:
            archive.writestr("resume.py", resume_code)
        if export_ready and export_dir is not None:
            root = export_dir.resolve()
            for item in sorted(export_dir.rglob("*")):
                if not item.is_file():
                    continue
                if item.is_symlink() or not item.resolve().is_relative_to(root):
                    raise ValueError("The model export contains a symlinked file.")
                archive.write(item, arcname=Path("model") / item.relative_to(export_dir))
    return destination
