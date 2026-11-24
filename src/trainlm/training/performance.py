"""Measured training throughput and model FLOPs utilization reporting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainlm.benchmark.mfu import calculate_causal_lm_flops, calculate_mfu


_TPU_PEAK_FLOPS_PER_DEVICE = {
    # Google Cloud TPU v5e specifications, peak BF16 compute per chip.
    ("TPU v5e", "bf16"): 197_000_000_000_000.0,
}

_CUDA_PEAK_FLOPS_RULES: tuple[tuple[str, tuple[str, ...], float, str], ...] = (
    (
        "H100 PCIE",
        ("bf16", "fp16"),
        756_000_000_000_000.0,
        "https://resources.nvidia.com/en-us-tensor-core",
    ),
    (
        "H100",
        ("bf16", "fp16"),
        989_000_000_000_000.0,
        "https://resources.nvidia.com/en-us-tensor-core",
    ),
    (
        "A100",
        ("bf16", "fp16"),
        312_000_000_000_000.0,
        "https://www.nvidia.com/en-us/data-center/a100/",
    ),
    (
        "L40S",
        ("bf16", "fp16"),
        362_000_000_000_000.0,
        "https://www.nvidia.com/en-us/data-center/l40s/",
    ),
    (
        "L4",
        ("bf16", "fp16"),
        121_000_000_000_000.0,
        "https://www.nvidia.com/en-us/data-center/l4/",
    ),
    (
        "T4",
        ("fp16",),
        65_000_000_000_000.0,
        "https://www.nvidia.com/en-us/data-center/tesla-t4/",
    ),
    (
        "V100",
        ("fp16",),
        125_000_000_000_000.0,
        "https://www.nvidia.com/en-us/data-center/v100/",
    ),
)


def _resolve_hardware_peak(
    *,
    device_type: Any,
    device_name: Any,
    precision: Any,
) -> tuple[float | None, str | None]:
    tpu_peak = _TPU_PEAK_FLOPS_PER_DEVICE.get((device_name, precision))
    if tpu_peak is not None:
        return tpu_peak, "https://docs.cloud.google.com/tpu/docs/v5e"
    if device_type == "cuda" and isinstance(device_name, str):
        normalized = device_name.upper()
        for token, supported_precisions, peak, source in _CUDA_PEAK_FLOPS_RULES:
            if token in normalized and precision in supported_precisions:
                return peak, source
    return None, None


def build_performance_summary(
    *,
    measurements: Mapping[str, Any],
    model: Any,
    runtime: Any,
    sequence_length: int,
) -> dict[str, Any]:
    """Combine observed training timings with the standard dense-LM MFU formula.

    Throughput and data-loader wait come from synchronized wall-clock windows
    collected by the trainer. MFU follows the conventional model-FLOPs divided
    by hardware-peak definition; it is unavailable when model geometry or a
    documented hardware peak cannot be identified.
    """

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "measured" if measurements.get("wall_seconds", 0) > 0 else "not_measured",
        "measurement_scope": "synchronized_training_windows",
        "measured_optimizer_steps": int(measurements.get("measured_optimizer_steps", 0)),
        "supervised_tokens": int(measurements.get("supervised_tokens", 0)),
        "wall_seconds": float(measurements.get("wall_seconds", 0.0)),
        "tokens_per_second": float(measurements.get("tokens_per_second", 0.0)),
        "dataloader_wait_seconds": float(measurements.get("dataloader_wait_seconds", 0.0)),
        "dataloader_wait_fraction": float(measurements.get("dataloader_wait_fraction", 0.0)),
        "mfu_status": "unavailable",
        "mfu_unavailable_reason": None,
        "mfu_method": "standard_dense_causal_lm_model_flops_over_documented_peak",
        "mfu_definition": (
            "MFU uses measured supervised tokens/second and the standard dense-LM "
            "model FLOPs formula divided by the documented peak; it is not a "
            "hardware performance counter."
        ),
        "bottleneck_hint": _bottleneck_hint(
            float(measurements.get("dataloader_wait_fraction", 0.0)),
            measurements.get("wall_seconds", 0) > 0,
        ),
        "kernel_profile_available": False,
        "recent_windows": list(measurements.get("recent_windows", ())),
        "recent_window_limit": 100,
    }
    if result["status"] != "measured":
        result["mfu_unavailable_reason"] = "No synchronized training window was recorded."
        return result

    try:
        diagnostics = runtime.diagnostics()
        values = diagnostics.values
        backend = diagnostics.backend
        device_type = diagnostics.device_type
        precision = diagnostics.precision
        world_size = int(diagnostics.world_size)
        device_name = values.get("device_name")
    except Exception as exc:
        result["mfu_unavailable_reason"] = f"Runtime diagnostics unavailable ({type(exc).__name__})."
        return result

    hardware: dict[str, Any] = {
        "backend": backend,
        "device_type": device_type,
        "device_name": device_name,
        "precision": precision,
        "device_count": world_size,
        "peak_flops_per_device": None,
        "peak_source": None,
    }
    result["hardware"] = hardware
    global_tps = measurements.get("global_tokens_per_second")
    if world_size != 1 and global_tps is None:
        result["mfu_unavailable_reason"] = (
            "MFU currently requires a single-device measured window or explicit global_tokens_per_second."
        )
        return result
    peak, peak_source = _resolve_hardware_peak(
        device_type=device_type,
        device_name=device_name,
        precision=precision,
    )
    if peak is None:
        result["mfu_unavailable_reason"] = (
            "No documented peak is registered for this exact accelerator and precision."
        )
        return result

    model_config = getattr(model, "config", None)
    geometry = _model_geometry(model, model_config)
    if geometry is None:
        result["mfu_unavailable_reason"] = (
            "The model does not expose supported dense decoder-only geometry."
        )
        return result
    try:
        flops = calculate_causal_lm_flops(
            parameter_count=geometry["parameter_count"],
            vocabulary_size=geometry["vocabulary_size"],
            hidden_size=geometry["hidden_size"],
            num_hidden_layers=geometry["num_hidden_layers"],
            sequence_length=sequence_length,
            tie_word_embeddings=geometry["tie_word_embeddings"],
        )
    except (TypeError, ValueError):
        result["mfu_unavailable_reason"] = "Model geometry is not valid for the dense causal-LM FLOPs formula."
        return result

    tokens_per_second = (
        float(global_tps)
        if (world_size != 1 and global_tps is not None)
        else result["tokens_per_second"]
    )
    if tokens_per_second <= 0:
        result["mfu_unavailable_reason"] = "Measured token throughput is not positive."
        return result
    result.update({
        "mfu_status": "measured",
        "mfu_unavailable_reason": None,
        "model_geometry": geometry,
        "sequence_length": sequence_length,
        "non_embedding_flops_per_token": flops.non_embedding_flops_per_token,
        "logits_inclusive_flops_per_token": flops.logits_inclusive_flops_per_token,
        "non_embedding_mfu": calculate_mfu(
            tokens_per_second=tokens_per_second,
            flops_per_token=flops.non_embedding_flops_per_token,
            peak_flops_per_device=peak,
            device_count=world_size,
        ),
        "logits_inclusive_mfu": calculate_mfu(
            tokens_per_second=tokens_per_second,
            flops_per_token=flops.logits_inclusive_flops_per_token,
            peak_flops_per_device=peak,
            device_count=world_size,
        ),
    })
    windows = result["recent_windows"]
    if len(windows) >= 2:
        steady_windows = windows[1:]
        steady_wall = sum(
            float(w.get("train_wall_seconds", 0.0)) for w in steady_windows
        )
        steady_tokens = sum(
            float(w.get("supervised_tokens", 0.0)) for w in steady_windows
        )
        steady_wait = sum(
            float(w.get("dataloader_wait_seconds", 0.0)) for w in steady_windows
        )
        if steady_wall > 0 and steady_tokens > 0:
            steady_tps = steady_tokens / steady_wall
            steady_mfu_tps = (
                steady_tps * world_size if world_size != 1 else steady_tps
            )
            result["steady_state"] = {
                "measured_windows": len(steady_windows),
                "supervised_tokens": int(steady_tokens),
                "wall_seconds": steady_wall,
                "tokens_per_second": steady_mfu_tps,
                "dataloader_wait_fraction": steady_wait / steady_wall,
                "non_embedding_mfu": calculate_mfu(
                    tokens_per_second=steady_mfu_tps,
                    flops_per_token=flops.non_embedding_flops_per_token,
                    peak_flops_per_device=peak,
                    device_count=world_size,
                ),
                "logits_inclusive_mfu": calculate_mfu(
                    tokens_per_second=steady_mfu_tps,
                    flops_per_token=flops.logits_inclusive_flops_per_token,
                    peak_flops_per_device=peak,
                    device_count=world_size,
                ),
            }
    hardware["peak_flops_per_device"] = peak
    hardware["peak_source"] = peak_source
    return result


def _model_geometry(model: Any, config: Any) -> dict[str, Any] | None:
    if config is None or getattr(config, "is_encoder_decoder", False):
        return None
    if any(getattr(config, name, None) for name in ("num_experts", "n_experts", "num_local_experts")):
        return None
    vocabulary_size = _first_positive_int(config, "vocab_size")
    hidden_size = _first_positive_int(config, "hidden_size", "n_embd", "d_model")
    layers = _first_positive_int(config, "num_hidden_layers", "n_layer", "num_layers")
    try:
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    except (AttributeError, TypeError, ValueError):
        return None
    tied = getattr(config, "tie_word_embeddings", None)
    if (
        vocabulary_size is None
        or hidden_size is None
        or layers is None
        or isinstance(tied, bool) is False
        or parameter_count <= 0
    ):
        return None
    return {
        "parameter_count": parameter_count,
        "vocabulary_size": vocabulary_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": layers,
        "tie_word_embeddings": tied,
    }


def _first_positive_int(value: Any, *names: str) -> int | None:
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    return None


def _bottleneck_hint(wait_fraction: float, measured: bool) -> str:
    if not measured:
        return "unavailable"
    if wait_fraction >= 0.2:
        return "elevated_dataloader_wait"
    return "no_major_dataloader_wait_observed"


__all__ = ["build_performance_summary"]
