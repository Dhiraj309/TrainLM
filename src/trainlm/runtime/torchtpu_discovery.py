"""Bounded native TPU discovery for the process that will own the device.

This module has no PyTorch import at module load time. It reports observed
facts; it does not select a production backend or certify hardware support.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _error(exc: Exception) -> str:
    # Exception text may include runtime configuration or secrets.
    return type(exc).__name__


def _property(props: Any, name: str) -> str | int | float | list[int] | None:
    try:
        value = getattr(props, name, None)
    except Exception:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, (tuple, list)) and len(value) <= 8:
        if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
            return list(value)
    return None


def discover_tpu(
    torch: Any, *, verify_allocation: bool = True, include_properties: bool = True
) -> dict[str, Any]:
    """Report TPU availability and hardware facts inside the owning process.

    A failed allocation is never translated into CPU success. Device count is
    process-visible and is not a chip or distributed-world count.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "device_type": None,
        "device_count_visible": None,
        "device_name": None,
        "topology": None,
        "hbm_bytes_per_device": None,
        "hbm_bytes_per_chip": None,
        "devices_per_chip": None,
        "allocation": "not_run",
        "error_type": None,
        "properties_error_type": None,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torch_tpu_version": _package_version("torch_tpu"),
        "libtpu_version": _package_version("libtpu"),
    }
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None or not callable(
        getattr(accelerator, "current_accelerator", None)
    ):
        report["status"] = "api_unavailable"
        return report
    try:
        current = accelerator.current_accelerator()
        report["device_type"] = getattr(current, "type", None)
        if report["device_type"] != "tpu":
            return report
        if not accelerator.is_available():
            return report
        report["device_count_visible"] = accelerator.device_count()
    except Exception as exc:
        report["status"] = "query_failed"
        report["error_type"] = _error(exc)
        return report

    if verify_allocation:
        try:
            torch.empty(1, device=torch.device("tpu")).cpu()
        except Exception as exc:
            report["status"] = "allocation_failed"
            report["allocation"] = "failed"
            report["error_type"] = _error(exc)
            return report
        report["allocation"] = "passed"

    if include_properties:
        tpu = getattr(torch, "tpu", None)
        getter = getattr(tpu, "get_device_properties", None)
        if callable(getter):
            try:
                props = getter()
                for field in (
                    "name",
                    "topology",
                    "hbm_bytes_per_device",
                    "hbm_bytes_per_chip",
                    "devices_per_chip",
                ):
                    report["device_name" if field == "name" else field] = _property(
                        props, field
                    )
            except Exception as exc:
                report["properties_error_type"] = _error(exc)
    report["status"] = "ready" if verify_allocation else "available_unverified"
    return report


def select_auto_backend(report: dict[str, Any], *, cuda_available: bool) -> str:
    """Preserve CPU/CUDA auto routing while native TPU training is unverified.

    A detected TPU must not silently become a CPU run. Explicit ``tpu`` still
    selects the historical XLA coordinator until a native route is validated.
    """
    if report.get("device_type") == "tpu":
        raise RuntimeError(
            "Native TorchTPU was detected, but TrainLM's native TPU training "
            "route is not integrated. Use accelerator='tpu' only for the "
            "existing PyTorch/XLA coordinator, or choose 'cpu'/'cuda' explicitly."
        )
    if (report.get("torch_tpu_version") is not None
            and report.get("status") in {"api_unavailable", "query_failed"}):
        raise RuntimeError(
            "TorchTPU is installed, but accelerator discovery failed. Check "
            "the torch/TorchTPU installation before using accelerator='auto'."
        )
    return "cuda" if cuda_available else "cpu"
