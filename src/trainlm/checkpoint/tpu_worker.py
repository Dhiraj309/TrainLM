"""Rank-local checkpoint persistence used by private TPU workers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from trainlm.training import TrainerPhase


SCHEMA_VERSION = 1
_REQUIRED_KEYS = {
    "schema_version", "world_size", "rank", "model", "optimizer",
    "scheduler", "runtime", "trainer", "cpu_rng_state",
}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return type(value)((key, _cpu_tree(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def save_tpu_worker_checkpoint(engine, destination: str | Path) -> Path:
    """Atomically save one replicated-DP worker's exact training state."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rank = engine.runtime.rank
    world_size = engine.runtime.world_size
    payload = _cpu_tree({
        "schema_version": SCHEMA_VERSION,
        "world_size": world_size,
        "rank": rank,
        "model": engine.model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "runtime": dict(engine.runtime.state_dict()),
        "trainer": {
            key: value for key, value in asdict(engine.state).items()
            if key not in {"phase", "is_training", "should_stop", "failure"}
        },
        "cpu_rng_state": torch.get_rng_state(),
    })
    shard = root / f"rank-{rank:05d}.pt"
    temporary = shard.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(shard)
    engine.runtime.barrier(f"checkpoint-shards:{engine.state.step}")
    if engine.runtime.is_primary_process:
        missing = [
            index for index in range(world_size)
            if not (root / f"rank-{index:05d}.pt").is_file()
        ]
        if missing:
            raise RuntimeError(f"Checkpoint is missing rank shards: {missing}")
        manifest = root / "manifest.json"
        manifest_tmp = manifest.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "status": "committed",
            "world_size": world_size,
            "step": engine.state.step,
            "micro_step": engine.state.micro_step,
            "shards": [f"rank-{index:05d}.pt" for index in range(world_size)],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_tmp.replace(manifest)
    engine.runtime.barrier(f"checkpoint-published:{engine.state.step}")
    return root


def load_tpu_worker_checkpoint(engine, source: str | Path) -> Path:
    """Restore the current rank after validating a committed topology."""

    root = Path(source)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Committed TPU checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "committed":
        raise ValueError("TPU checkpoint is not a committed schema-version-1 checkpoint.")
    if manifest.get("world_size") != engine.runtime.world_size:
        raise ValueError("TPU checkpoint world size does not match the active topology.")
    shard = root / f"rank-{engine.runtime.rank:05d}.pt"
    if not shard.is_file():
        raise FileNotFoundError(f"TPU checkpoint rank shard is missing: {shard}")
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_KEYS:
        raise ValueError("TPU checkpoint rank shard has an invalid state layout.")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported TPU checkpoint rank-shard schema.")
    if payload["world_size"] != engine.runtime.world_size or payload["rank"] != engine.runtime.rank:
        raise ValueError("TPU checkpoint rank shard does not match the active topology.")
    if "device_rng_state" not in payload["runtime"]:
        raise ValueError("TPU checkpoint rank shard is missing device RNG state.")
    engine.model.load_state_dict(payload["model"], strict=True)
    engine.optimizer.load_state_dict(payload["optimizer"])
    engine.scheduler.load_state_dict(payload["scheduler"])
    engine.runtime.load_state_dict(payload["runtime"])
    for key, value in payload["trainer"].items():
        if not hasattr(engine.state, key):
            raise ValueError(f"Unknown trainer-state field in TPU checkpoint: {key}")
        setattr(engine.state, key, value)
    engine.state.phase = TrainerPhase.RESUMING
    torch.set_rng_state(payload["cpu_rng_state"])
    return root


__all__ = ["load_tpu_worker_checkpoint", "save_tpu_worker_checkpoint"]
