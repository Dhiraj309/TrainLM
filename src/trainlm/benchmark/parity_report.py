"""Versioned reproducibility report for exact 135M parity certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ParityReport:
    schema_version: int
    report_id: str
    environment: tuple[tuple[str, str], ...]
    commands: tuple[str, ...]
    configuration_artifacts: tuple[str, ...]
    metric_artifacts: tuple[str, ...]
    profile_artifact: str
    hlo_artifact: str
    limitations: tuple[str, ...]
    numerical_alignment_passed: bool
    repeated_benchmark_passed: bool
    real_shard_stability_passed: bool
    plain_hf_export_passed: bool

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ParityReport supports schema_version=1 only.")
        if not isinstance(self.report_id, str) or not self.report_id:
            raise ValueError("report_id cannot be empty.")
        if not self.environment:
            raise ValueError("environment cannot be empty.")
        names = []
        for name, value in self.environment:
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not value
            ):
                raise ValueError(
                    "environment entries require non-empty names and values."
                )
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("environment names must be unique.")
        for name in ("commands", "configuration_artifacts", "metric_artifacts"):
            values = getattr(self, name)
            if not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings.")
        for name in ("profile_artifact", "hlo_artifact"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} cannot be empty.")
        if any(not isinstance(value, str) or not value for value in self.limitations):
            raise ValueError("limitations must contain non-empty strings.")
        for name in (
            "numerical_alignment_passed",
            "repeated_benchmark_passed",
            "real_shard_stability_passed",
            "plain_hf_export_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")

    @property
    def certified(self) -> bool:
        return all(
            (
                self.numerical_alignment_passed,
                self.repeated_benchmark_passed,
                self.real_shard_stability_passed,
                self.plain_hf_export_passed,
            )
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        status = "CERTIFIED" if self.certified else "NOT CERTIFIED"
        lines = [
            f"# TrainLM parity report: {self.report_id}",
            "",
            f"**Status:** {status}",
            "",
            "## Environment",
            *(f"- {name}: {value}" for name, value in self.environment),
            "",
            "## Reproduction commands",
            *(f"- {command}" for command in self.commands),
            "",
            "## Evidence",
            *(f"- Configuration: {path}" for path in self.configuration_artifacts),
            *(f"- Metrics: {path}" for path in self.metric_artifacts),
            f"- Profile: {self.profile_artifact}",
            f"- HLO: {self.hlo_artifact}",
            "",
            "## Limitations",
            *(f"- {item}" for item in self.limitations),
        ]
        if not self.limitations:
            lines.append("- None reported.")
        return "\n".join(lines) + "\n"


def load_parity_report(report_path: str | Path) -> ParityReport:
    """Load a strict report and verify every referenced artifact is confined."""

    if not isinstance(report_path, (str, Path)):
        raise TypeError("report_path must be a path.")
    path = Path(report_path)
    if not path.is_file():
        raise ValueError("report_path must reference an existing file.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid parity report: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Parity report must contain a JSON object.")
    expected = {field.name for field in fields(ParityReport)}
    if set(payload) != expected:
        raise ValueError("Parity report keys must match schema version 1.")

    values: dict[str, Any] = dict(payload)
    environment = values["environment"]
    if isinstance(environment, (str, bytes)) or not isinstance(environment, list):
        raise ValueError("environment must be a JSON array of name/value pairs.")
    if any(not isinstance(item, list) or len(item) != 2 for item in environment):
        raise ValueError("environment must contain two-item name/value arrays.")
    values["environment"] = tuple(tuple(item) for item in environment)
    for name in (
        "commands",
        "configuration_artifacts",
        "metric_artifacts",
        "limitations",
    ):
        items = values[name]
        if isinstance(items, (str, bytes)) or not isinstance(items, list):
            raise ValueError(f"{name} must be a JSON array.")
        values[name] = tuple(items)
    try:
        report = ParityReport(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parity report evidence: {exc}") from exc

    root = path.resolve().parent
    references = (
        *report.configuration_artifacts,
        *report.metric_artifacts,
        report.profile_artifact,
        report.hlo_artifact,
    )
    for reference in references:
        candidate = Path(reference)
        if candidate.is_absolute():
            raise ValueError("Parity report artifact paths must be relative.")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(
                "Parity report artifact path escapes the report or is missing."
            )
    return report


__all__ = ["ParityReport", "load_parity_report"]
