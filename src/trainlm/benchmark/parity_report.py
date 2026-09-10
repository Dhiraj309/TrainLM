"""Versioned reproducibility report for exact 135M parity certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


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
        if self.schema_version != 1:
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


__all__ = ["ParityReport"]
