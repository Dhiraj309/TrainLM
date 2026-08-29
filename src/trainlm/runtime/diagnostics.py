"""Bounded, opt-in diagnostics for PyTorch/XLA execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _metric_samples(metrics: Any, name: str) -> int:
    metric_data = getattr(metrics, "metric_data", None)
    if not callable(metric_data):
        return 0
    try:
        data = metric_data(name)
        return max(0, int(data[0]))
    except (IndexError, TypeError, ValueError, KeyError):
        return 0


@dataclass(slots=True)
class XlaDiagnostics:
    """Collect compiler counters and bounded reports without hot-path logging."""

    metrics_module: Any
    profiler_module: Any | None = None
    max_report_chars: int = 16_384
    max_counter_entries: int = 128
    _hlo: str | None = None
    _profile_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_report_chars < 128:
            raise ValueError("max_report_chars must be at least 128.")
        if self.max_counter_entries < 1:
            raise ValueError("max_counter_entries must be positive.")

    def record_hlo(self, hlo: str | None) -> None:
        if hlo is not None and not isinstance(hlo, str):
            raise TypeError("hlo must be text or None.")
        self._hlo = self._truncate(hlo)

    def record_profile_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        if metadata is None:
            self._profile_metadata = {}
            return
        self._profile_metadata = {
            str(key): value
            for key, value in list(metadata.items())[: self.max_counter_entries]
        }

    def snapshot(self) -> dict[str, Any]:
        metrics = self.metrics_module
        counter_names = getattr(metrics, "counter_names", lambda: ())()
        counter_names = tuple(str(name) for name in counter_names)
        counters = {
            name: self._counter_value(metrics, name)
            for name in sorted(counter_names)
            if name.startswith("aten::")
        }
        counters = dict(list(counters.items())[: self.max_counter_entries])
        return {
            "compile_count": _metric_samples(metrics, "CompileTime"),
            "execute_count": _metric_samples(metrics, "ExecuteTime"),
            "aten_fallback_count": sum(counters.values()),
            "aten_fallbacks": counters,
            "short_metrics_report": self._report(metrics, "short_metrics_report"),
            "metrics_report": self._report(metrics, "metrics_report"),
            "hlo": self._hlo,
            "profile_metadata": dict(self._profile_metadata),
        }

    def clear(self) -> None:
        clear_all = getattr(self.metrics_module, "clear_all", None)
        if callable(clear_all):
            clear_all()

    def _counter_value(self, metrics: Any, name: str) -> int:
        counter_value = getattr(metrics, "counter_value", None)
        if not callable(counter_value):
            return 0
        try:
            return max(0, int(counter_value(name)))
        except (TypeError, ValueError, KeyError):
            return 0

    def _report(self, metrics: Any, method_name: str) -> str | None:
        report = getattr(metrics, method_name, None)
        if not callable(report):
            return None
        return self._truncate(str(report()))

    def _truncate(self, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) <= self.max_report_chars:
            return value
        return value[: self.max_report_chars] + "\n...[truncated]"
