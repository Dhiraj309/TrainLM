"""Evidence gate for the matched 1.3B SPMD FSDP benchmark."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Mapping


def _positive(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class FSDPScalingTarget:
    """Review-locked external target and matched workload identity."""

    workload_id: str
    workload_version: int
    parameter_count: int
    accelerator_type: str
    device_count: int
    data_replicas: int
    fsdp_shards: int
    global_tokens_per_second: float
    mfu: float
    source: str
    locked: bool

    def __post_init__(self) -> None:
        for name in ("workload_id", "accelerator_type", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty.")
        for name in (
            "workload_version",
            "parameter_count",
            "device_count",
            "data_replicas",
            "fsdp_shards",
        ):
            _positive_integer(name, getattr(self, name))
        _positive("global_tokens_per_second", self.global_tokens_per_second)
        if isinstance(self.mfu, bool) or not isinstance(self.mfu, (int, float)):
            raise TypeError("mfu must be numeric.")
        if not math.isfinite(self.mfu) or not 0 < self.mfu <= 1:
            raise ValueError("mfu must be in (0, 1].")
        if not isinstance(self.locked, bool):
            raise TypeError("locked must be boolean.")
        if self.data_replicas * self.fsdp_shards != self.device_count:
            raise ValueError("Target mesh dimensions must equal device_count.")


@dataclass(frozen=True, slots=True)
class FSDPScalingEvidence:
    """Measured TrainLM result and required graph/lifecycle artifacts."""

    workload_id: str
    workload_version: int
    parameter_count: int
    accelerator_type: str
    device_count: int
    data_replicas: int
    fsdp_shards: int
    global_tokens_per_second: float
    mfu: float
    peak_hbm_gib: float
    unexpected_compile_count: int
    cpu_fallback_count: int
    correctness_passed: bool
    resume_passed: bool
    export_passed: bool
    collective_artifact: str
    hbm_artifact: str

    def __post_init__(self) -> None:
        for name in (
            "workload_id",
            "accelerator_type",
            "collective_artifact",
            "hbm_artifact",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty.")
        for name in (
            "workload_version",
            "parameter_count",
            "device_count",
            "data_replicas",
            "fsdp_shards",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("global_tokens_per_second", "peak_hbm_gib"):
            _positive(name, getattr(self, name))
        if isinstance(self.mfu, bool) or not isinstance(self.mfu, (int, float)):
            raise TypeError("mfu must be numeric.")
        if not math.isfinite(self.mfu) or not 0 < self.mfu <= 1:
            raise ValueError("mfu must be in (0, 1].")
        for name in ("unexpected_compile_count", "cpu_fallback_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("correctness_passed", "resume_passed", "export_passed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if self.data_replicas * self.fsdp_shards != self.device_count:
            raise ValueError("Evidence mesh dimensions must equal device_count.")


@dataclass(frozen=True, slots=True)
class FSDPScalingEvaluation:
    passed: bool
    reasons: tuple[str, ...]
    throughput_ratio: float
    mfu_ratio: float


def evaluate_fsdp_scaling(
    evidence: FSDPScalingEvidence,
    target: FSDPScalingTarget,
) -> FSDPScalingEvaluation:
    """Compare a measured result with a review-locked matched target."""

    if not isinstance(evidence, FSDPScalingEvidence):
        raise TypeError("evidence must be FSDPScalingEvidence.")
    if not isinstance(target, FSDPScalingTarget):
        raise TypeError("target must be FSDPScalingTarget.")
    reasons: list[str] = []
    if not target.locked:
        reasons.append("comparison target is not review-locked")
    for field in (
        "workload_id",
        "workload_version",
        "parameter_count",
        "accelerator_type",
        "device_count",
        "data_replicas",
        "fsdp_shards",
    ):
        if getattr(evidence, field) != getattr(target, field):
            reasons.append(f"{field} does not match the target")
    if evidence.global_tokens_per_second < target.global_tokens_per_second:
        reasons.append("global throughput is below the locked target")
    if evidence.mfu < target.mfu:
        reasons.append("MFU is below the locked target")
    if evidence.unexpected_compile_count:
        reasons.append("unexpected compilation occurred after warmup")
    if evidence.cpu_fallback_count:
        reasons.append("CPU fallback counters were observed")
    for field in ("correctness_passed", "resume_passed", "export_passed"):
        if not getattr(evidence, field):
            reasons.append(f"{field.removesuffix('_passed')} evidence failed")
    return FSDPScalingEvaluation(
        passed=not reasons,
        reasons=tuple(reasons),
        throughput_ratio=(
            evidence.global_tokens_per_second / target.global_tokens_per_second
        ),
        mfu_ratio=evidence.mfu / target.mfu,
    )


def load_fsdp_scaling_evaluation(
    manifest_path: str | Path,
) -> FSDPScalingEvaluation:
    """Load a locked target and measured FSDP evidence from one bundle."""

    if not isinstance(manifest_path, (str, Path)):
        raise TypeError("manifest_path must be a path.")
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("manifest_path must reference an existing file.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid FSDP scaling manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("FSDP scaling manifest must contain a JSON object.")
    if set(manifest) != {"schema_version", "target", "evidence"}:
        raise ValueError("FSDP scaling manifest keys must match schema version 1.")
    version = manifest["schema_version"]
    if isinstance(version, bool) or version != 1:
        raise ValueError("FSDP scaling manifest supports schema_version=1 only.")

    target_payload = manifest["target"]
    evidence_payload = manifest["evidence"]
    if not isinstance(target_payload, Mapping) or set(target_payload) != {
        field.name for field in fields(FSDPScalingTarget)
    }:
        raise ValueError("FSDP scaling target keys must match the schema.")
    if not isinstance(evidence_payload, Mapping) or set(evidence_payload) != {
        field.name for field in fields(FSDPScalingEvidence)
    }:
        raise ValueError("FSDP scaling evidence keys must match the schema.")
    try:
        target = FSDPScalingTarget(**dict(target_payload))
        evidence = FSDPScalingEvidence(**dict(evidence_payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid FSDP scaling evidence: {exc}") from exc

    root = path.resolve().parent
    for reference in (evidence.collective_artifact, evidence.hbm_artifact):
        candidate = Path(reference)
        if candidate.is_absolute():
            raise ValueError("FSDP scaling artifact paths must be relative.")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(
                "FSDP scaling artifact path escapes the manifest or is missing."
            )
    return evaluate_fsdp_scaling(evidence, target)


__all__ = [
    "FSDPScalingEvaluation",
    "FSDPScalingEvidence",
    "FSDPScalingTarget",
    "evaluate_fsdp_scaling",
    "load_fsdp_scaling_evaluation",
]
