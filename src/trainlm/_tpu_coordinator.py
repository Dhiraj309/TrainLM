"""Private coordinator for launching the existing single-VM TPU worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from trainlm.config import ModelSourceConfig


def _format_worker_detail(detail: str) -> str:
    """Render worker JSON events as compact, useful notebook progress."""

    try:
        value = json.loads(detail)
    except (TypeError, json.JSONDecodeError):
        return f"latest: {detail}"
    if not isinstance(value, dict):
        return f"latest: {detail}"
    stage = value.get("stage")
    if stage == "worker_entered":
        return f"workers online ({value.get('world_size', '?')} replicas)"
    if stage == "probe_passed":
        return f"collective probe passed ({value.get('world_size', '?')} replicas)"
    if stage == "data_preflight":
        return "validating packed data"
    if stage == "launch_dp8":
        return "launching TPU workers"
    if stage == "train_start" and isinstance(value.get("parallelism"), dict):
        topology = value["parallelism"]
        return (
            f"training started (DP{topology.get('data_parallel', '?')} / "
            f"MP{topology.get('model_parallel', '?')})"
        )
    if "step" in value:
        fields = [f"step {value['step']}"]
        if value.get("loss") is not None:
            fields.append(f"loss {float(value['loss']):.4f}")
        if value.get("learning_rate") is not None:
            fields.append(f"lr {float(value['learning_rate']):.3g}")
        tokens = value.get("global_tokens_seen", value.get("tokens_seen"))
        if tokens is not None:
            fields.append(f"tokens {int(tokens):,}")
        return " | ".join(fields)
    if stage:
        return str(stage)
    return f"latest: {detail}"


def _render_progress_document(path: Path) -> bool:
    """Update one IPython display for the live progress document."""

    if not path.is_file():
        return False
    try:
        from IPython.display import Markdown, display
    except ImportError:
        return False
    display(
        Markdown(path.read_text(encoding="utf-8")),
        display_id="trainlm-progress",
    )
    return True


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
    logging_verbosity: str = "normal"
    loss_implementation: str = "causal_lm"
    logits_chunk_size: int | None = None
    save_every_steps: int | None = None
    resume_from_checkpoint: Path | None = None
    eval_manifest_dir: Path | None = None
    eval_every_steps: int | None = None
    max_eval_batches: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logging_verbosity, str)
            or self.logging_verbosity not in {"quiet", "normal", "verbose"}
        ):
            raise ValueError(
                "logging_verbosity must be 'quiet', 'normal', or 'verbose'."
            )
        if self.loss_implementation not in {
            "auto", "causal_lm", "model", "chunked_linear"
        }:
            raise ValueError("Unsupported loss_implementation.")
        if self.logits_chunk_size is not None and (
            isinstance(self.logits_chunk_size, bool)
            or not isinstance(self.logits_chunk_size, int)
            or self.logits_chunk_size < 1
        ):
            raise ValueError("logits_chunk_size must be positive when configured.")
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
        if self.max_eval_batches is not None and (
            isinstance(self.max_eval_batches, bool)
            or not isinstance(self.max_eval_batches, int)
            or self.max_eval_batches < 1
        ):
            raise ValueError("max_eval_batches must be positive when configured.")
        if self.max_eval_batches is not None and self.eval_manifest_dir is None:
            raise ValueError("max_eval_batches requires an evaluation dataset.")

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
            # Launch PJRT only once for a training request.  The worker performs
            # the collective probe before constructing the model and training,
            # so separate probe/preflight launches only add two PJRT teardown
            # cycles.  On Kaggle those diagnostic-only teardowns can core dump
            # even after every rank reported success, poisoning the next launch.
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
        log_path = request.output_dir / f"{stage}.log"
        progress_path = request.output_dir / "progress.md"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if stage == "train":
            progress_path.unlink(missing_ok=True)
        started = time.monotonic()
        if request.logging_verbosity == "verbose":
            print(
                f"[TrainLM] {stage}: started; live progress: "
                f"{request.output_dir / 'progress.md'}",
                flush=True,
            )
        else:
            print(
                f"[TrainLM] {stage}: started (details: {log_path}; "
                f"progress: {request.output_dir / 'progress.md'})",
                flush=True,
            )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self.worker_script.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                # Diagnostic stages must not retain all ranks indefinitely if
                # PJRT or the compiler hangs. Training itself remains bounded
                # by the user's max_steps rather than these safety limits.
                timeout = {"probe": 900, "model_preflight": 1800}.get(stage)
                returncode = self._wait_with_heartbeat(
                    process,
                    stage=stage,
                    log_path=log_path,
                    progress_path=progress_path,
                    timeout=timeout,
                    verbosity=request.logging_verbosity,
                )
            except subprocess.TimeoutExpired as exc:
                self._terminate_process_group(process)
                raise TPUCoordinatorError(
                    f"TPU {stage} stage produced no progress for {timeout} "
                    "seconds. TrainLM terminated the worker process group; see "
                    f"{log_path}."
                ) from exc
            except BaseException:
                self._terminate_process_group(process)
                raise
        if returncode:
            # A failed launcher can leave spawned XLA ranks alive. Reclaim the
            # whole private process group before returning control to a notebook.
            self._terminate_process_group(process)
            tail = "\n".join(
                log_path.read_text(encoding="utf-8").splitlines()[-20:]
            )
            raise TPUCoordinatorError(
                f"TPU {stage} stage failed with exit code {returncode}. "
                f"See {log_path}. Last output:\n{tail}"
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_marker = next(
            (
                marker
                for marker in ("RAW: Dumping core", "exit() hanging: exiting process")
                if marker in log_text
            ),
            None,
        )
        if fatal_marker is not None:
            raise TPUCoordinatorError(
                f"TPU {stage} reported a fatal PJRT worker shutdown "
                f"({fatal_marker!r}) even though its launcher returned success. "
                "Restart the notebook session to reset the TPU runtime before "
                f"retrying; see {log_path}."
            )
        if request.logging_verbosity != "verbose":
            print(
                f"[TrainLM] {stage}: completed in {time.monotonic() - started:.1f}s",
                flush=True,
            )
        else:
            _render_progress_document(progress_path)

    @staticmethod
    def _wait_with_heartbeat(
        process: subprocess.Popen[Any],
        *,
        stage: str,
        log_path: Path,
        progress_path: Path | None = None,
        timeout: int | None,
        heartbeat_seconds: int = 10,
        verbosity: str = "normal",
    ) -> int:
        """Wait while reporting bounded, low-volume notebook progress."""

        if verbosity not in {"quiet", "normal", "verbose"}:
            raise ValueError("Unsupported coordinator logging verbosity.")
        if verbosity == "quiet":
            heartbeat_seconds = max(heartbeat_seconds, 60)
        elif verbosity == "verbose":
            heartbeat_seconds = min(heartbeat_seconds, 5)
        elapsed = 0
        inactive = 0
        previous_detail: str | None = None
        last_detail = "waiting for first worker event"
        while True:
            wait_seconds = heartbeat_seconds
            if timeout is not None:
                wait_seconds = min(wait_seconds, timeout - inactive)
                if wait_seconds <= 0:
                    raise subprocess.TimeoutExpired(str(log_path), timeout)
            try:
                return process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                elapsed += wait_seconds
                inactive += wait_seconds
                try:
                    lines = log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                except OSError:
                    lines = []
                if lines:
                    last_detail = lines[-1][-240:]
                detail_changed = last_detail != previous_detail
                if detail_changed:
                    inactive = 0
                    previous_detail = last_detail
                if verbosity != "quiet":
                    if verbosity == "verbose" and progress_path is not None:
                        if _render_progress_document(progress_path):
                            continue
                    rendered = (
                        _format_worker_detail(last_detail)
                        if verbosity == "verbose"
                        else f"latest: {last_detail}"
                    )
                    # Verbose mode keeps the notebook to one live artifact:
                    # progress.md. Emit only sparse liveness if the worker has
                    # not changed its event stream for a minute.
                    if verbosity != "verbose":
                        print(
                            f"[TrainLM] {stage}: still running ({elapsed}s); "
                            f"{rendered}; inactive={inactive}s",
                            flush=True,
                        )
                    elif inactive and inactive % 60 == 0:
                        print(
                            f"[TrainLM] {stage}: no new worker event for "
                            f"{inactive}s; last={rendered}",
                            flush=True,
                        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
        """Best-effort cleanup for a launcher and every spawned TPU rank."""

        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:  # pragma: no cover - TPU workers are POSIX
                process.terminate()
            else:  # pragma: no cover - TPU notebook workers are POSIX
                return
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - TPU notebook workers are POSIX
                    process.kill()
                process.wait()

    def _command(self, request: _TPURunRequest) -> list[str]:
        model = request.model
        if model.provider != "huggingface":
            raise TPUCoordinatorError(
                "The current TPU coordinator requires a reconstructible Hugging "
                "Face model source."
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
            "--model-source-json", json.dumps(asdict(model), sort_keys=True),
            "--learning-rate", str(request.learning_rate),
            "--beta1", str(request.betas[0]),
            "--beta2", str(request.betas[1]),
            "--eps", str(request.eps),
            "--weight-decay", str(request.weight_decay),
            "--z-loss", "0.0",
            "--scheduler", request.scheduler,
            "--warmup-steps", str(request.warmup_steps),
            "--precision", request.precision,
            "--logging-verbosity", request.logging_verbosity,
            "--loss-implementation", request.loss_implementation,
        ]
        if request.logits_chunk_size is not None:
            command.extend(("--logits-chunk-size", str(request.logits_chunk_size)))
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
        if request.max_eval_batches is not None:
            command.extend(("--max-eval-batches", str(request.max_eval_batches)))
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
