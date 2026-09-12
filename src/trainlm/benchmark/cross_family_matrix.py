"""Evidence records for matched 135M cross-family certification."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping


SupportLevel = Literal["certified", "experimental", "unsupported"]


@dataclass(frozen=True, slots=True)
class CrossFamilyRecord:
    """One measured support and performance record for a model family."""

    family_id: str
    support_level: SupportLevel
    workload_id: str
    accelerator_type: str
    device_count: int
    scheduled_tokens_per_update: int
    global_tokens_per_second: float
    model_flops_per_token: float
    peak_device_tflops: float
    full_attention: bool
    correctness_passed: bool
    graph_passed: bool
    export_passed: bool
    evidence_artifact: str

    def __post_init__(self) -> None:
        for name in (
            "family_id",
            "workload_id",
            "accelerator_type",
            "evidence_artifact",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty.")
        if self.support_level not in {"certified", "experimental", "unsupported"}:
            raise ValueError(f"Unsupported support level: {self.support_level!r}.")
        for name in ("device_count", "scheduled_tokens_per_update"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "global_tokens_per_second",
            "model_flops_per_token",
            "peak_device_tflops",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "full_attention",
            "correctness_passed",
            "graph_passed",
            "export_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")

    @property
    def architecture_adjusted_mfu(self) -> float:
        achieved_flops = self.global_tokens_per_second * self.model_flops_per_token
        peak_flops = self.device_count * self.peak_device_tflops * 1e12
        return achieved_flops / peak_flops


@dataclass(frozen=True, slots=True)
class CrossFamilyMatrixEvaluation:
    complete: bool
    certified: bool
    reasons: tuple[str, ...]
    records: tuple[CrossFamilyRecord, ...]


def evaluate_cross_family_matrix(
    records: tuple[CrossFamilyRecord, ...],
    *,
    advertised_families: tuple[str, ...],
    minimum_full_attention_mfu: float = 0.45,
) -> CrossFamilyMatrixEvaluation:
    """Validate matched records without executing or profiling model runs."""

    if not advertised_families or any(
        not isinstance(family, str) or not family.strip()
        for family in advertised_families
    ):
        raise ValueError("advertised_families must contain non-empty IDs.")
    if len(advertised_families) != len(set(advertised_families)):
        raise ValueError("advertised_families must be unique.")
    if (
        isinstance(minimum_full_attention_mfu, bool)
        or not isinstance(minimum_full_attention_mfu, (int, float))
        or not math.isfinite(minimum_full_attention_mfu)
        or not 0 < minimum_full_attention_mfu <= 1
    ):
        raise ValueError("minimum_full_attention_mfu must be in (0, 1].")
    if any(not isinstance(record, CrossFamilyRecord) for record in records):
        raise TypeError("records must contain CrossFamilyRecord values.")
    record_ids = [record.family_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Cross-family record IDs must be unique.")

    expected = set(advertised_families)
    observed = set(record_ids)
    reasons = [
        f"missing record for advertised family {family!r}"
        for family in sorted(expected - observed)
    ]
    reasons.extend(
        f"record for unadvertised family {family!r}"
        for family in sorted(observed - expected)
    )
    if records:
        reference = records[0]
        geometry = (
            "workload_id",
            "accelerator_type",
            "device_count",
            "scheduled_tokens_per_update",
        )
        for record in records[1:]:
            for field in geometry:
                if getattr(record, field) != getattr(reference, field):
                    reasons.append(
                        f"{record.family_id}: {field} does not match the matrix"
                    )
    for record in records:
        if record.support_level != "certified":
            reasons.append(
                f"{record.family_id}: support level is {record.support_level!r}"
            )
        for field in ("correctness_passed", "graph_passed", "export_passed"):
            if not getattr(record, field):
                evidence = field.removesuffix("_passed")
                reasons.append(f"{record.family_id}: {evidence} evidence failed")
        if (
            record.full_attention
            and record.architecture_adjusted_mfu < minimum_full_attention_mfu
        ):
            reasons.append(
                f"{record.family_id}: architecture-adjusted MFU is below "
                "the full-attention gate"
            )

    complete = expected == observed
    return CrossFamilyMatrixEvaluation(
        complete=complete,
        certified=complete and not reasons,
        reasons=tuple(reasons),
        records=records,
    )


def load_cross_family_matrix_evaluation(
    manifest_path: str | Path,
) -> CrossFamilyMatrixEvaluation:
    """Load a strict matrix manifest and verify each evidence artifact."""

    if not isinstance(manifest_path, (str, Path)):
        raise TypeError("manifest_path must be a path.")
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("manifest_path must reference an existing file.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cross-family matrix manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Cross-family matrix manifest must contain a JSON object.")
    if set(manifest) != {
        "schema_version",
        "advertised_families",
        "minimum_full_attention_mfu",
        "records",
    }:
        raise ValueError(
            "Cross-family matrix manifest keys must match schema version 1."
        )
    version = manifest["schema_version"]
    if isinstance(version, bool) or version != 1:
        raise ValueError("Cross-family matrix manifest supports schema_version=1 only.")
    advertised = manifest["advertised_families"]
    records_payload = manifest["records"]
    if isinstance(advertised, (str, bytes)) or not isinstance(advertised, list):
        raise ValueError("advertised_families must be a JSON array.")
    if isinstance(records_payload, (str, bytes)) or not isinstance(
        records_payload, list
    ):
        raise ValueError("records must be a JSON array.")

    expected_record = {field.name for field in fields(CrossFamilyRecord)}
    records = []
    root = path.resolve().parent
    for index, payload in enumerate(records_payload):
        if not isinstance(payload, Mapping) or set(payload) != expected_record:
            raise ValueError(f"record {index} keys must match the versioned schema.")
        try:
            record = CrossFamilyRecord(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cross-family record {index}: {exc}") from exc
        evidence_path = Path(record.evidence_artifact)
        if evidence_path.is_absolute():
            raise ValueError("Cross-family evidence paths must be relative.")
        resolved = (root / evidence_path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(
                "Cross-family evidence path escapes the manifest or is missing."
            )
        records.append(record)
    try:
        return evaluate_cross_family_matrix(
            tuple(records),
            advertised_families=tuple(advertised),
            minimum_full_attention_mfu=manifest["minimum_full_attention_mfu"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cross-family matrix gate: {exc}") from exc


__all__ = [
    "CrossFamilyMatrixEvaluation",
    "CrossFamilyRecord",
    "SupportLevel",
    "evaluate_cross_family_matrix",
    "load_cross_family_matrix_evaluation",
]
