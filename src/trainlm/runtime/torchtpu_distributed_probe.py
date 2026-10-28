"""Source-level contracts for a single-host native TorchTPU collective probe.

Importing this module never imports torch or initializes a TPU. Launch setup
belongs in the parent process, before any TPU allocation or worker spawn.
"""

from __future__ import annotations

from typing import Any, Mapping


def _positive_int(value: str | None) -> int | None:
    if value is None or not value.isdecimal():
        return None
    result = int(value)
    return result if result > 0 else None


def inspect_launch_environment(environ: Mapping[str, str]) -> dict[str, Any]:
    """Check topology structure without exposing coordinator addresses."""
    world = _positive_int(environ.get("WORLD_SIZE"))
    raw_shape = environ.get("TORCH_TPU_TOPOLOGY", "")
    parts = raw_shape.split(",") if raw_shape else []
    shape = [_positive_int(part.strip()) for part in parts]
    topology = shape if shape and all(part is not None for part in shape) else None
    raw_addresses = environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
    addresses = raw_addresses.split(",") if raw_addresses else []
    address_count = len(addresses) if all(part.strip() for part in addresses) else 0
    node_rank_raw = environ.get("NODE_RANK")
    node_rank = int(node_rank_raw) if node_rank_raw and node_rank_raw.isdecimal() else None
    nnodes = _positive_int(environ.get("NNODES")) if "NNODES" in environ else None

    missing = [
        key for key in (
            "WORLD_SIZE", "TORCH_TPU_TOPOLOGY", "TORCH_TPU_SLICEBUILDER_ADDRESSES"
        ) if not environ.get(key)
    ]
    issues: list[str] = []
    if not missing:
        if world is None:
            issues.append("invalid_world_size")
        if topology is None:
            issues.append("invalid_topology")
        if not address_count:
            issues.append("invalid_address_list")
        if world is not None and topology is not None:
            product = 1
            for dimension in topology:
                product *= dimension
            if product != world:
                issues.append("topology_world_mismatch")
        if world is not None and address_count and address_count != world:
            issues.append("address_world_mismatch")
    if node_rank_raw is not None and node_rank is None:
        issues.append("invalid_node_rank")
    elif node_rank not in (None, 0):
        issues.append("multi_host_unsupported")
    if "NNODES" in environ and nnodes is None:
        issues.append("invalid_nnodes")
    elif nnodes not in (None, 1):
        issues.append("multi_host_unsupported")

    return {
        "schema_version": 1,
        "status": "incomplete" if missing else "invalid" if issues else "ready",
        "world_size": world,
        "topology_shape": topology,
        "coordinator_address_count": address_count,
        "node_rank": node_rank,
        "missing_keys": missing,
        "issues": sorted(set(issues)),
    }


def prepare_launch_environment(torch: Any, environ: Mapping[str, str]) -> dict[str, Any]:
    """Invoke TorchTPU's parent-process helper, then inspect its result.

    The upstream helper may populate process environment variables. Call this
    only in an opt-in launcher before any TPU runtime initialization.
    """
    helper = getattr(getattr(getattr(torch, "tpu", None), "distributed", None),
                     "set_tpu_launch_env", None)
    if not callable(helper):
        return {"status": "api_unavailable", "error_type": None}
    try:
        helper()
    except Exception as exc:
        return {"status": "preparation_failed", "error_type": type(exc).__name__}
    return inspect_launch_environment(environ)


def summarize_workers(records: list[dict[str, Any]], expected_world: int) -> dict[str, Any]:
    """Reconcile independent worker observations after torchrun exits."""
    if expected_world < 2:
        raise ValueError("collective probe requires at least two workers")
    ranks = [item.get("rank") for item in records]
    local_ranks = [item.get("local_rank") for item in records]
    valid = (
        len(records) == expected_world
        and all(type(rank) is int for rank in ranks)
        and sorted(ranks) == list(range(expected_world))
        and all(type(rank) is int for rank in local_ranks)
        and sorted(local_ranks) == list(range(expected_world))
        and all(type(item.get("world_size")) is int
                and item["world_size"] == expected_world for item in records)
        and all(item.get("collective") == "passed" for item in records)
    )
    return {
        "status": "passed" if valid else "failed",
        "expected_world_size": expected_world,
        "reports_received": len(records),
        "ranks": sorted(rank for rank in ranks if type(rank) is int),
        "local_ranks": sorted(rank for rank in local_ranks if type(rank) is int),
    }
