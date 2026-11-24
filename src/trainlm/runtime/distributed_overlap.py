"""Multi-device DDP overlap and FSDP2 (``fully_shard``) model wrapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class DistributedWrapResult:
    """Summary of multi-device DDP/FSDP2 model wrapping."""

    strategy: str
    wrapped: bool
    sharded_blocks: int = 0
    reason: str | None = None


def _is_distributed_ready() -> tuple[bool, int]:
    dist = getattr(torch, "distributed", None)
    if dist is None or not callable(getattr(dist, "is_available", None)) or not dist.is_available():
        return False, 1
    if not callable(getattr(dist, "is_initialized", None)) or not dist.is_initialized():
        return False, 1
    world_size = int(dist.get_world_size())
    return world_size > 1, world_size


def _find_decoder_blocks(model: nn.Module) -> nn.ModuleList | None:
    """Locate repeated transformer block containers structurally without family name guesses."""
    for attr in ("layers", "h", "blocks"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate
    for child in model.children():
        for attr in ("layers", "h", "blocks"):
            candidate = getattr(child, attr, None)
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                return candidate
    return None


def apply_distributed_data_parallel(
    model: nn.Module,
    *,
    device: torch.device | None = None,
    bucket_cap_mb: int = 50,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = True,
) -> tuple[nn.Module, DistributedWrapResult]:
    """Wrap ``model`` in ``DistributedDataParallel`` with comm-compute overlap settings."""
    ready, world_size = _is_distributed_ready()
    if not ready:
        return model, DistributedWrapResult(
            strategy="none",
            wrapped=False,
            reason=f"distributed process group not active (world_size={world_size})",
        )
    from torch.nn.parallel import DistributedDataParallel as DDP

    kwargs: dict[str, Any] = {
        "bucket_cap_mb": bucket_cap_mb,
        "gradient_as_bucket_view": gradient_as_bucket_view,
        "static_graph": static_graph,
    }
    if device is not None and device.type == "cuda" and device.index is not None:
        kwargs["device_ids"] = [device.index]
        kwargs["output_device"] = device.index
    try:
        wrapped = DDP(model, **kwargs)
    except TypeError:
        kwargs.pop("static_graph", None)
        kwargs.pop("gradient_as_bucket_view", None)
        wrapped = DDP(model, **kwargs)
    return wrapped, DistributedWrapResult(strategy="ddp", wrapped=True)


def apply_fsdp2_fully_shard(
    model: nn.Module,
    *,
    reshard_after_forward: bool = True,
    strict: bool = False,
) -> tuple[nn.Module, DistributedWrapResult]:
    """Apply PyTorch FSDP2 ``fully_shard`` to transformer blocks and root model."""
    ready, world_size = _is_distributed_ready()
    if not ready:
        msg = f"distributed process group not active (world_size={world_size})"
        if strict:
            raise RuntimeError(f"FSDP2 requires an active distributed group: {msg}")
        return model, DistributedWrapResult(strategy="none", wrapped=False, reason=msg)
    try:
        from torch.distributed.fsdp import fully_shard  # type: ignore[import-not-found]
    except ImportError as exc:
        if strict:
            raise RuntimeError("FSDP2 torch.distributed.fsdp.fully_shard is unavailable.") from exc
        return model, DistributedWrapResult(
            strategy="none",
            wrapped=False,
            reason="torch.distributed.fsdp.fully_shard unavailable",
        )

    blocks = _find_decoder_blocks(model)
    sharded_blocks = 0
    if blocks is not None:
        for block in blocks:
            fully_shard(block, reshard_after_forward=reshard_after_forward)
            sharded_blocks += 1
    fully_shard(model, reshard_after_forward=reshard_after_forward)
    return model, DistributedWrapResult(
        strategy="fsdp2",
        wrapped=True,
        sharded_blocks=sharded_blocks,
    )


def prepare_distributed_model(
    model: nn.Module,
    *,
    strategy: Literal["auto", "ddp", "fsdp2", "none"] = "auto",
    device: torch.device | None = None,
    bucket_cap_mb: int = 50,
    reshard_after_forward: bool = True,
) -> tuple[nn.Module, DistributedWrapResult]:
    """Prepare ``model`` for multi-device training according to ``strategy``."""
    if strategy == "none":
        return model, DistributedWrapResult(strategy="none", wrapped=False, reason="disabled")
    ready, world_size = _is_distributed_ready()
    if not ready:
        if strategy == "fsdp2":
            raise RuntimeError(
                f"distributed_strategy='fsdp2' requires world_size > 1 (got {world_size})."
            )
        if strategy == "ddp":
            raise RuntimeError(
                f"distributed_strategy='ddp' requires world_size > 1 (got {world_size})."
            )
        return model, DistributedWrapResult(
            strategy="none",
            wrapped=False,
            reason=f"single-device or uninitialized distributed group (world_size={world_size})",
        )
    if strategy == "fsdp2":
        return apply_fsdp2_fully_shard(
            model,
            reshard_after_forward=reshard_after_forward,
            strict=True,
        )
    return apply_distributed_data_parallel(
        model,
        device=device,
        bucket_cap_mb=bucket_cap_mb,
    )
