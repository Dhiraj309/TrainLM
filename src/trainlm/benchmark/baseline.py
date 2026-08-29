"""Locked 135M baseline workload and evidence evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .result import BenchmarkResult


@dataclass(frozen=True, slots=True)
class BaselineWorkload:
    """The geometry and acceptance gate for the generic 135M XLA run."""

    manifest_id: str
    manifest_version: int
    accelerator_type: str
    device_count: int
    sequence_length: int
    micro_batch_per_device: int
    gradient_accumulation_steps: int
    data_parallel_replicas: int
    expected_tokens_per_update: int
    minimum_global_tokens_per_second: int = 600_000

    def __post_init__(self) -> None:
        if not self.manifest_id.strip() or not self.accelerator_type.strip():
            raise ValueError("Baseline identity fields cannot be empty.")
        for name in (
            "manifest_version",
            "device_count",
            "sequence_length",
            "micro_batch_per_device",
            "gradient_accumulation_steps",
            "data_parallel_replicas",
            "expected_tokens_per_update",
            "minimum_global_tokens_per_second",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        calculated = (
            self.sequence_length
            * self.micro_batch_per_device
            * self.gradient_accumulation_steps
            * self.data_parallel_replicas
        )
        if calculated != self.expected_tokens_per_update:
            raise ValueError("Baseline token geometry does not match its fields.")


@dataclass(frozen=True, slots=True)
class BaselineEvaluation:
    """Evidence decision for the generic baseline go/no-go gate."""

    passed: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    global_tokens_per_second: float
    non_embedding_mfu: float
    unexpected_compile_count: int
    cpu_fallback_count: int


def load_baseline_workload(path: str | Path) -> BaselineWorkload:
    """Load and validate a locked baseline manifest without executing training."""

    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Baseline manifest root must be a mapping.")
    if data.get("status") != "locked_reference":
        raise ValueError("Baseline manifest must be a locked reference.")
    runtime = _mapping(data, "runtime")
    batch = _mapping(data, "batch")
    thresholds = _mapping(data, "acceptance_thresholds")
    hard_gate = _mapping(thresholds, "hard_90_percent")
    workload = BaselineWorkload(
        manifest_id=str(data["manifest_id"]),
        manifest_version=int(data["manifest_version"]),
        accelerator_type=str(runtime["accelerator_type"]),
        device_count=int(runtime["device_count"]),
        sequence_length=int(batch["sequence_length"]),
        micro_batch_per_device=int(batch["micro_batch_per_device"]),
        gradient_accumulation_steps=int(batch["gradient_accumulation_steps"]),
        data_parallel_replicas=int(batch["data_parallel_replicas"]),
        expected_tokens_per_update=int(
            batch["expected_tokens_per_optimizer_update"]
        ),
        minimum_global_tokens_per_second=int(
            hard_gate["minimum_global_tokens_per_second"]
        ),
    )
    if workload.accelerator_type != "v5e-8":
        raise ValueError("The generic M5 baseline requires accelerator v5e-8.")
    return workload


def evaluate_baseline(
    result: BenchmarkResult,
    workload: BaselineWorkload,
) -> BaselineEvaluation:
    """Evaluate one synchronized steady-state result against the M5 gate."""

    reasons: list[str] = []
    warnings: list[str] = []
    if result.measurement_kind != "steady_state":
        reasons.append("result must be a steady_state measurement")
    if result.workload_id != workload.manifest_id:
        reasons.append("workload ID does not match the locked manifest")
    if result.workload_version != workload.manifest_version:
        reasons.append("workload version does not match the locked manifest")
    if result.accelerator_type != workload.accelerator_type:
        reasons.append("accelerator type does not match the locked workload")
    if result.device_count != workload.device_count:
        reasons.append("device count does not match the locked workload")
    if result.data_parallel_replicas != workload.data_parallel_replicas:
        reasons.append("data-parallel replica count does not match")
    if result.scheduled_tokens_per_update != workload.expected_tokens_per_update:
        reasons.append("scheduled tokens per update do not match")
    if result.global_tokens_per_second < workload.minimum_global_tokens_per_second:
        reasons.append("global throughput is below the 600K go/no-go gate")
    if result.unexpected_compile_count:
        reasons.append("unexpected compilation occurred after warmup")
    if result.cpu_fallback_count:
        warnings.append("CPU fallback counters were observed")
    return BaselineEvaluation(
        passed=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        global_tokens_per_second=result.global_tokens_per_second,
        non_embedding_mfu=result.non_embedding_mfu,
        unexpected_compile_count=result.unexpected_compile_count,
        cpu_fallback_count=result.cpu_fallback_count,
    )


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    nested = value.get(name)
    if not isinstance(nested, Mapping):
        raise ValueError(f"Baseline manifest section '{name}' must be a mapping.")
    return nested
