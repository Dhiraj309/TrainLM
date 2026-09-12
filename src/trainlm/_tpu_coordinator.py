"""Private coordinator for launching the existing single-VM TPU worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from trainlm.config import ModelSourceConfig


class TPUCoordinatorError(RuntimeError):
    """Raised when a TPU worker stage cannot produce a successful run."""


@dataclass(frozen=True, slots=True)
class _TPURunRequest:
    model: ModelSourceConfig
    manifest_dir: Path
    output_dir: Path
    max_steps: int
    gradient_accumulation_steps: int
    micro_batch_per_device: int
    sequence_length: int
    seed: int
    log_every_steps: int
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    scheduler: str
    warmup_steps: int
    precision: str
    save_every_steps: int | None = None
    resume_from_checkpoint: Path | None = None
    eval_manifest_dir: Path | None = None
    eval_every_steps: int | None = None

    def __post_init__(self) -> None:
        if self.save_every_steps is not None and (
            isinstance(self.save_every_steps, bool)
            or not isinstance(self.save_every_steps, int)
            or self.save_every_steps < 1
        ):
            raise ValueError("save_every_steps must be positive when configured.")
        if self.resume_from_checkpoint is not None:
            checkpoint = Path(self.resume_from_checkpoint)
            if not checkpoint.is_dir():
                raise ValueError(
                    f"TPU resume checkpoint directory does not exist: {checkpoint}"
                )
            object.__setattr__(self, "resume_from_checkpoint", checkpoint)
        if (self.eval_manifest_dir is None) != (self.eval_every_steps is None):
            raise ValueError(
                "eval_manifest_dir and eval_every_steps must be configured together."
            )
        if self.eval_every_steps is not None and (
            isinstance(self.eval_every_steps, bool)
            or not isinstance(self.eval_every_steps, int)
            or self.eval_every_steps < 1
        ):
            raise ValueError("eval_every_steps must be positive when configured.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["manifest_dir"] = str(self.manifest_dir)
        values["output_dir"] = str(self.output_dir)
        if self.resume_from_checkpoint is not None:
            values["resume_from_checkpoint"] = str(self.resume_from_checkpoint)
        if self.eval_manifest_dir is not None:
            values["eval_manifest_dir"] = str(self.eval_manifest_dir)
        return values


class _TPUCoordinator:
    """Own subprocess execution and stage-log handling for the public facade."""

    def __init__(self, worker_script: str | Path | None = None) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        self.worker_script = Path(
            worker_script
            or repository_root / "scripts" / "trainlm_tpu_worker.py"
        )

    def run(self, request: _TPURunRequest) -> dict[str, Any]:
        if not self.worker_script.is_file():
            raise TPUCoordinatorError(
                "The TrainLM TPU worker entry point is unavailable. Use an editable "
                "checkout or install a distribution that includes the worker scripts."
            )

        request.output_dir.mkdir(parents=True, exist_ok=True)
        (request.output_dir / "request.json").write_text(
            json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        completed_stages: list[str] = []
        try:
            self._run_stage("probe", request, "--probe-only")
            completed_stages.append("probe")
            self._run_stage("model_preflight", request, "--model-preflight")
            completed_stages.append("model_preflight")
            self._run_stage("train", request)
            completed_stages.append("train")
            worker_summary_path = request.output_dir / "summary.json"
            if not worker_summary_path.is_file():
                raise TPUCoordinatorError(
                    "TPU training exited successfully but did not write summary.json. "
                    f"Inspect {request.output_dir / 'train.log'}."
                )
            worker_summary = json.loads(
                worker_summary_path.read_text(encoding="utf-8")
            )
            metrics = self._read_metrics(request.output_dir / "metrics.jsonl")
        except Exception as exc:
            self._write_summary(
                request,
                status="failed",
                completed_stages=completed_stages,
                error=str(exc),
            )
            if isinstance(exc, TPUCoordinatorError):
                raise
            raise TPUCoordinatorError(f"TPU coordinator failed: {exc}") from exc

        return self._write_summary(
            request,
            status="completed",
            completed_stages=completed_stages,
            worker_summary=worker_summary,
            metrics=metrics,
        )

    @staticmethod
    def _read_metrics(path: Path) -> list[dict[str, float]]:
        if not path.is_file():
            return []
        snapshots = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TPUCoordinatorError(
                    f"Invalid TPU metrics artifact at {path}:{line_number}."
                ) from exc
            if not isinstance(value, dict) or any(
                not isinstance(key, str)
                or isinstance(item, bool)
                or not isinstance(item, (int, float))
                for key, item in value.items()
            ):
                raise TPUCoordinatorError(
                    f"Invalid TPU metric snapshot at {path}:{line_number}."
                )
            snapshots.append({key: float(item) for key, item in value.items()})
        return snapshots

    def _run_stage(
        self,
        stage: str,
        request: _TPURunRequest,
        mode: str | None = None,
    ) -> None:
        command = self._command(request)
        if mode is not None:
            command.append(mode)
        result = subprocess.run(
            command,
            cwd=self.worker_script.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path = request.output_dir / f"{stage}.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")
        if result.returncode:
            tail = "\n".join((result.stdout or "").splitlines()[-20:])
            raise TPUCoordinatorError(
                f"TPU {stage} stage failed with exit code {result.returncode}. "
                f"See {log_path}. Last output:\n{tail}"
            )

    def _command(self, request: _TPURunRequest) -> list[str]:
        model = request.model
        if model.initialization != "pretrained" or model.name_or_path is None:
            raise TPUCoordinatorError(
                "The current TPU coordinator requires a reconstructible pretrained "
                "Hugging Face model ID or path."
            )
        unsupported = {
            "cache_dir": model.cache_dir is not None,
            "config_overrides": bool(model.config_overrides),
            "dtype": model.dtype is not None,
            "local_files_only": model.local_files_only,
            "subfolder": model.subfolder is not None,
            "use_safetensors": model.use_safetensors is not None,
        }
        enabled = sorted(name for name, present in unsupported.items() if present)
        if enabled:
            raise TPUCoordinatorError(
                "The current TPU worker cannot preserve these model-source options: "
                + ", ".join(enabled)
                + "."
            )
        command = [
            sys.executable,
            str(self.worker_script),
            "--max-steps", str(request.max_steps),
            "--gradient-accumulation-steps", str(request.gradient_accumulation_steps),
            "--micro-batch-per-device", str(request.micro_batch_per_device),
            "--sequence-length", str(request.sequence_length),
            "--seed", str(request.seed),
            "--log-every-steps", str(request.log_every_steps),
            "--output-dir", str(request.output_dir.resolve()),
            "--cache-dir", str((request.output_dir / "xla_cache").resolve()),
            "--data-mode", "local",
            "--manifest-dir", str(request.manifest_dir.resolve()),
            "--model-id", model.name_or_path,
            "--learning-rate", str(request.learning_rate),
            "--beta1", str(request.betas[0]),
            "--beta2", str(request.betas[1]),
            "--eps", str(request.eps),
            "--weight-decay", str(request.weight_decay),
            "--z-loss", "0.0",
            "--scheduler", request.scheduler,
            "--warmup-steps", str(request.warmup_steps),
            "--precision", request.precision,
        ]
        if model.revision is not None:
            command.extend(("--model-revision", model.revision))
        if model.trust_remote_code:
            command.append("--trust-remote-code")
        if request.save_every_steps is not None:
            command.extend(("--save-every-steps", str(request.save_every_steps)))
        if request.resume_from_checkpoint is not None:
            command.extend(
                ("--resume-from-checkpoint", str(request.resume_from_checkpoint.resolve()))
            )
        if request.eval_manifest_dir is not None:
            command.extend(
                ("--eval-manifest-dir", str(request.eval_manifest_dir.resolve()))
            )
            command.extend(("--eval-every-steps", str(request.eval_every_steps)))
        return command

    @staticmethod
    def _write_summary(
        request: _TPURunRequest,
        *,
        status: str,
        completed_stages: list[str],
        worker_summary: dict[str, Any] | None = None,
        metrics: list[dict[str, float]] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        summary = {
            "status": status,
            "completed_stages": completed_stages,
            "request": request.to_dict(),
            "worker_summary": worker_summary,
            "metrics": metrics or [],
            "error": error,
        }
        path = request.output_dir / "coordinator_summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return summary


__all__ = ["TPUCoordinatorError"]
