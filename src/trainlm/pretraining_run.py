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
        super().__init__(
            f"Pretraining failed ({error_type}); inspect the run report at "
            f"{report_path}."
        )


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
    def metrics(self) -> dict[str, float]:
        training = self.training
        return {
            key: float(training[key])
            for key in ("train_loss", "eval_loss")
            if isinstance(training.get(key), (int, float))
        }

    def summary(self) -> str:
        training = self.training
        model = self.report.get("model", {})
        accelerator = self.report.get("accelerator", {})
        return (
            f"status={self.status} "
            f"model={model.get('model_class') or model.get('requested_model_type', 'unknown')} "
            f"steps={training.get('step', 'not_run')} "
            f"tokens={training.get('supervised_tokens', 'not_run')}/"
            f"{training.get('target_supervised_tokens', 'unknown')} "
            f"eval_loss={training.get('eval_loss', 'not_run')} "
            f"accelerator={accelerator.get('backend', accelerator.get('requested', 'unknown'))}"
        )


def run_pretraining(
    config: TrainLMPretrainingConfig,
    *,
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
    ``train``, ``evaluate``, and ``save_model`` methods.

    The report is written even when model preparation, training, evaluation,
    or export fails. In that case :class:`TrainLMPretrainingError` identifies
    the report path and preserves the original exception as its cause.
    Set ``show_progress=True`` to print sparse step, token, loss, and learning
    rate snapshots at the configured logging cadence.
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
            "extend_training_budget": extend_training_budget
        }
        if show_progress:
            trainer_options["callbacks"] = [_make_training_progress_callback(config)]
        trainer = TrainLMTrainer.from_scratch(config, **trainer_options)
        _capture_effective_configuration(report, trainer)
        _capture_runtime(report, trainer)

        state = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        checkpoint = _checkpoint_report(trainer)
        evaluation = trainer.evaluate()
        completed_export = Path(trainer.save_model(export_dir))
        _capture_export(report, completed_export)

        train_loss = float(state.loss) if state.loss is not None else None
        eval_loss = float(evaluation["eval_loss"])
        tokens_seen = int(state.tokens_seen)
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
        }
        report["checkpoint"] = checkpoint
        report["model_export"] = str(completed_export)
        report["status"] = (
            "passed"
            if tokens_seen >= config.max_tokens
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
    """Create a sparse console progress callback without extra dependencies."""
    from .training.callback import TrainerCallback

    tokens_per_step = (
        (config.sequence_length - 1)
        * config.per_device_train_batch_size
        * config.gradient_accumulation_steps
    )
    total_steps = math.ceil(config.max_tokens / tokens_per_step)

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
            learning_rate = metrics.get("learning_rate", state.learning_rate)
            loss_text = f"{loss:.4f}" if loss is not None else "unavailable"
            print(
                f"train step {step}/{total_steps} | "
                f"tokens {tokens_seen:,}/{config.max_tokens:,} | "
                f"loss {loss_text} | lr {learning_rate:.3g}"
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

    return {
        "schema_version": 6,
        "scope": "experimental_single_device_raw_text_pretraining",
        "status": "not_started",
        "source_revision": source_revision,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "device": config.accelerator,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "data": {
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "train_split": config.train_split,
            "validation_split": config.validation_split,
            "text_field": config.text_field,
            "max_train_examples": config.max_train_examples,
            "max_eval_examples": config.max_eval_examples,
        },
        "model": {
            "requested_model_type": config.model_type,
            "requested_config_overrides": dict(config.model_config_overrides),
        },
        "tokenizer": {
            "algorithm": "byte_level_bpe",
            "training_split": config.train_split,
            "requested_vocab_size": config.tokenizer_vocab_size,
        },
        "training_config": {
            "requested": {
                "target_supervised_tokens": config.max_tokens,
                "sequence_length": config.sequence_length,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "per_device_eval_batch_size": config.per_device_eval_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "bf16": config.bf16,
                "torch_compile": config.torch_compile,
                "logging_steps": config.logging_steps,
                "max_eval_batches": config.max_eval_batches,
                "seed": config.seed,
                "save_total_limit": config.save_total_limit,
            }
        },
        "accelerator": {"requested": config.accelerator},
        "checkpoint": {"latest_path": None},
        "limitations": ["does_not_certify_performance"],
    }


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
    engine_config = getattr(getattr(trainer, "engine", None), "config", None)
    trainer_config = getattr(engine_config, "trainer", None)
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
        "max_eval_batches": args.max_eval_batches,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "seed": args.seed,
        "report_to": args.report_to if isinstance(args.report_to, str) else list(args.report_to),
        "logging_verbosity": args.logging_verbosity,
        "loss_implementation": args.loss_implementation,
        "logits_chunk_size": args.logits_chunk_size,
        "max_grad_norm": getattr(trainer_config, "max_grad_norm", None),
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
