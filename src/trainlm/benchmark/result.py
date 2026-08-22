"""Serializable benchmark result schema with derived throughput and MFU."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Sequence

from trainlm.benchmark.mfu import calculate_mfu


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")


def _non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Version 1 result for one synchronized steady-state measurement window.

    Token arrays contain the actual counts observed on every data-parallel
    replica during the complete measured window. Throughput therefore includes
    all replicas and excludes ignored labels rather than relying on nominal
    batch geometry.
    """

    schema_version: int
    run_id: str
    workload_id: str
    workload_version: int
    framework: str
    framework_revision: str
    backend: str
    accelerator_type: str
    measurement_kind: str
    cache_state: str
    device_synchronized: bool
    device_count: int
    host_count: int
    data_parallel_replicas: int
    warmup_steps: int
    measured_steps: int
    scheduled_tokens_per_update: int
    supervised_tokens_per_replica: tuple[int, ...]
    ignored_tokens_per_replica: tuple[int, ...]
    measurement_wall_seconds: float
    measurement_device_seconds: float
    total_step_seconds_median: float
    device_step_seconds_median: float
    compile_seconds: float
    peak_hbm_gib: float
    input_idle_fraction: float
    collective_seconds_median: float
    compile_count: int
    unexpected_compile_count: int
    cpu_fallback_count: int
    non_embedding_flops_per_token: float
    logits_inclusive_flops_per_token: float
    peak_flops_per_device: float
    global_tokens_per_second: float
    device_tokens_per_second: float
    non_embedding_mfu: float
    logits_inclusive_mfu: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("BenchmarkResult supports schema_version=1 only.")
        for name in (
            "run_id",
            "workload_id",
            "framework",
            "framework_revision",
            "backend",
            "accelerator_type",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty.")
        if self.measurement_kind not in {
            "compile",
            "steady_state",
            "stability",
        }:
            raise ValueError(
                "measurement_kind must be compile, steady_state, or stability."
            )
        if self.cache_state not in {"cold", "warm", "unknown"}:
            raise ValueError("cache_state must be cold, warm, or unknown.")
        if self.device_synchronized is not True:
            raise ValueError(
                "device_synchronized must be true for a valid timed result."
            )
        for name in (
            "workload_version",
            "device_count",
            "host_count",
            "data_parallel_replicas",
            "measured_steps",
            "scheduled_tokens_per_update",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "warmup_steps",
            "compile_count",
            "unexpected_compile_count",
            "cpu_fallback_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        if len(self.supervised_tokens_per_replica) != self.data_parallel_replicas:
            raise ValueError(
                "supervised_tokens_per_replica must contain one count per "
                "data-parallel replica."
            )
        if len(self.ignored_tokens_per_replica) != self.data_parallel_replicas:
            raise ValueError(
                "ignored_tokens_per_replica must contain one count per "
                "data-parallel replica."
            )
        for name, counts in (
            ("supervised", self.supervised_tokens_per_replica),
            ("ignored", self.ignored_tokens_per_replica),
        ):
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counts
            ):
                raise ValueError(
                    f"{name} token counts must be non-negative integers."
                )
        if self.total_supervised_tokens <= 0:
            raise ValueError("At least one supervised token is required.")
        if self.data_parallel_replicas > self.device_count:
            raise ValueError(
                "data_parallel_replicas cannot exceed device_count."
            )

        for name in (
            "measurement_wall_seconds",
            "measurement_device_seconds",
            "total_step_seconds_median",
            "device_step_seconds_median",
            "non_embedding_flops_per_token",
            "logits_inclusive_flops_per_token",
            "peak_flops_per_device",
            "global_tokens_per_second",
            "device_tokens_per_second",
        ):
            _positive(name, getattr(self, name))
        for name in (
            "compile_seconds",
            "peak_hbm_gib",
            "collective_seconds_median",
            "non_embedding_mfu",
            "logits_inclusive_mfu",
        ):
            _non_negative(name, getattr(self, name))
        if not 0 <= self.input_idle_fraction <= 1:
            raise ValueError("input_idle_fraction must be between zero and one.")
        if self.measurement_device_seconds > self.measurement_wall_seconds:
            raise ValueError(
                "measurement_device_seconds cannot exceed the synchronized "
                "wall-clock window."
            )
        if self.device_step_seconds_median > self.total_step_seconds_median:
            raise ValueError(
                "device_step_seconds_median cannot exceed "
                "total_step_seconds_median."
            )
        if self.unexpected_compile_count > self.compile_count:
            raise ValueError(
                "unexpected_compile_count cannot exceed compile_count."
            )
        if (
            self.logits_inclusive_flops_per_token
            < self.non_embedding_flops_per_token
        ):
            raise ValueError(
                "logits-inclusive FLOPs cannot be below non-embedding FLOPs."
            )
        if self.logits_inclusive_mfu < self.non_embedding_mfu:
            raise ValueError(
                "logits-inclusive MFU cannot be below non-embedding MFU."
            )
        expected_global_throughput = (
            self.total_supervised_tokens / self.measurement_wall_seconds
        )
        expected_device_throughput = (
            self.total_supervised_tokens / self.measurement_device_seconds
        )
        if not math.isclose(
            self.global_tokens_per_second,
            expected_global_throughput,
            rel_tol=1e-12,
        ):
            raise ValueError(
                "global_tokens_per_second must be derived from actual "
                "supervised tokens and the synchronized wall window."
            )
        if not math.isclose(
            self.device_tokens_per_second,
            expected_device_throughput,
            rel_tol=1e-12,
        ):
            raise ValueError(
                "device_tokens_per_second must be derived from actual "
                "supervised tokens and the device window."
            )
        expected_non_embedding_mfu = calculate_mfu(
            tokens_per_second=self.device_tokens_per_second,
            flops_per_token=self.non_embedding_flops_per_token,
            peak_flops_per_device=self.peak_flops_per_device,
            device_count=self.device_count,
        )
        expected_logits_inclusive_mfu = calculate_mfu(
            tokens_per_second=self.device_tokens_per_second,
            flops_per_token=self.logits_inclusive_flops_per_token,
            peak_flops_per_device=self.peak_flops_per_device,
            device_count=self.device_count,
        )
        if not math.isclose(
            self.non_embedding_mfu,
            expected_non_embedding_mfu,
            rel_tol=1e-12,
        ):
            raise ValueError("non_embedding_mfu does not match its inputs.")
        if not math.isclose(
            self.logits_inclusive_mfu,
            expected_logits_inclusive_mfu,
            rel_tol=1e-12,
        ):
            raise ValueError("logits_inclusive_mfu does not match its inputs.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")

    @property
    def total_supervised_tokens(self) -> int:
        return sum(self.supervised_tokens_per_replica)

    @property
    def total_ignored_tokens(self) -> int:
        return sum(self.ignored_tokens_per_replica)

    @classmethod
    def from_measurement(
        cls,
        *,
        supervised_tokens_per_replica: Sequence[int],
        ignored_tokens_per_replica: Sequence[int],
        measurement_wall_seconds: float,
        measurement_device_seconds: float,
        non_embedding_flops_per_token: float,
        logits_inclusive_flops_per_token: float,
        peak_flops_per_device: float,
        **values: Any,
    ) -> "BenchmarkResult":
        supervised = tuple(supervised_tokens_per_replica)
        ignored = tuple(ignored_tokens_per_replica)
        for name, counts in (("supervised", supervised), ("ignored", ignored)):
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counts
            ):
                raise ValueError(
                    f"{name} token counts must be non-negative integers."
                )
        total_supervised_tokens = sum(supervised)
        _positive("measurement_wall_seconds", measurement_wall_seconds)
        _positive("measurement_device_seconds", measurement_device_seconds)
        _positive(
            "non_embedding_flops_per_token",
            non_embedding_flops_per_token,
        )
        _positive(
            "logits_inclusive_flops_per_token",
            logits_inclusive_flops_per_token,
        )
        _positive("peak_flops_per_device", peak_flops_per_device)
        if total_supervised_tokens <= 0:
            raise ValueError("At least one supervised token is required.")
        device_count = values.get("device_count")
        if isinstance(device_count, bool) or not isinstance(device_count, int):
            raise ValueError("device_count must be a positive integer.")
        global_tokens_per_second = (
            total_supervised_tokens / measurement_wall_seconds
        )
        device_tokens_per_second = (
            total_supervised_tokens / measurement_device_seconds
        )
        non_embedding_mfu = calculate_mfu(
            tokens_per_second=device_tokens_per_second,
            flops_per_token=non_embedding_flops_per_token,
            peak_flops_per_device=peak_flops_per_device,
            device_count=device_count,
        )
        logits_inclusive_mfu = calculate_mfu(
            tokens_per_second=device_tokens_per_second,
            flops_per_token=logits_inclusive_flops_per_token,
            peak_flops_per_device=peak_flops_per_device,
            device_count=device_count,
        )
        return cls(
            supervised_tokens_per_replica=supervised,
            ignored_tokens_per_replica=ignored,
            measurement_wall_seconds=measurement_wall_seconds,
            measurement_device_seconds=measurement_device_seconds,
            non_embedding_flops_per_token=non_embedding_flops_per_token,
            logits_inclusive_flops_per_token=logits_inclusive_flops_per_token,
            peak_flops_per_device=peak_flops_per_device,
            global_tokens_per_second=global_tokens_per_second,
            device_tokens_per_second=device_tokens_per_second,
            non_embedding_mfu=non_embedding_mfu,
            logits_inclusive_mfu=logits_inclusive_mfu,
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["supervised_tokens_per_replica"] = list(
            self.supervised_tokens_per_replica
        )
        result["ignored_tokens_per_replica"] = list(
            self.ignored_tokens_per_replica
        )
        result["metadata"] = dict(self.metadata)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BenchmarkResult":
        data = dict(values)
        data["supervised_tokens_per_replica"] = tuple(
            data["supervised_tokens_per_replica"]
        )
        data["ignored_tokens_per_replica"] = tuple(
            data["ignored_tokens_per_replica"]
        )
        data["metadata"] = dict(data.get("metadata", {}))
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "BenchmarkResult":
        return cls.from_dict(json.loads(value))
