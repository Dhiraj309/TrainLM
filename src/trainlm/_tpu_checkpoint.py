"""Rank-local checkpoint persistence used by private TPU workers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from trainlm.training import TrainerPhase


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, SCHEMA_VERSION})
_REQUIRED_KEYS_V1 = {
    "schema_version", "world_size", "rank", "model", "optimizer",
    "scheduler", "runtime", "trainer", "cpu_rng_state",
}
_REQUIRED_KEYS_V2 = {
    "schema_version", "world_size", "mesh_axes", "rank", "model", "optimizer",
    "scheduler", "runtime", "trainer", "cpu_rng_state",
}


def _validated_mesh_axes(value: Any, *, world_size: int) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError("TPU checkpoint mesh_axes must be a non-empty object.")
    axes: dict[str, int] = {}
    size = 1
    for name, axis_size in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "TPU checkpoint mesh axis names must be non-empty strings."
            )
        if (
            isinstance(axis_size, bool)
            or not isinstance(axis_size, int)
            or axis_size < 1
        ):
            raise ValueError(
                "TPU checkpoint mesh axis sizes must be positive integers."
            )
        axes[name] = axis_size
        size *= axis_size
    if size != world_size:
        raise ValueError(
            f"TPU checkpoint mesh size {size} does not match world size {world_size}."
        )
    return axes


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


def save_tpu_worker_checkpoint(
    engine,
    destination: str | Path,
    *,
    _stage_hook: Callable[[str, Path], None] | None = None,
) -> Path:
    """Atomically save one replicated-DP worker's exact training state."""

    def stage(name: str, path: Path) -> None:
        if _stage_hook is not None:
            _stage_hook(name, path)

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite committed checkpoint: {manifest}"
        )
    rank = engine.runtime.rank
    world_size = engine.runtime.world_size
    generation = f"step-{engine.state.step:012d}-micro-{engine.state.micro_step:012d}"
    runtime_state = dict(engine.runtime.state_dict())
    mesh_axes = _validated_mesh_axes(
        runtime_state.get("mesh_axes"), world_size=world_size
    )
    payload = _cpu_tree({
        "schema_version": SCHEMA_VERSION,
        "world_size": world_size,
        "mesh_axes": mesh_axes,
        "rank": rank,
        "model": engine.model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "runtime": runtime_state,
        "trainer": {
            key: value for key, value in asdict(engine.state).items()
            if key not in {"phase", "is_training", "should_stop", "failure"}
        },
        "cpu_rng_state": torch.get_rng_state(),
    })
    shard = root / f"{generation}-rank-{rank:05d}.pt"
    temporary = shard.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    stage("shard-staged", temporary)
    temporary.replace(shard)
    stage("shard-published", shard)
    engine.runtime.barrier(f"checkpoint-shards:{engine.state.step}")
    if engine.runtime.is_primary_process:
        missing = [
            index for index in range(world_size)
            if not (root / f"{generation}-rank-{index:05d}.pt").is_file()
        ]
        if missing:
            raise RuntimeError(f"Checkpoint is missing rank shards: {missing}")
        manifest_tmp = manifest.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "status": "committed",
            "world_size": world_size,
            "mesh_axes": mesh_axes,
            "step": engine.state.step,
            "micro_step": engine.state.micro_step,
            "generation": generation,
            "shards": [
                f"{generation}-rank-{index:05d}.pt"
                for index in range(world_size)
            ],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage("manifest-staged", manifest_tmp)
        manifest_tmp.replace(manifest)
        stage("manifest-published", manifest)
    engine.runtime.barrier(f"checkpoint-published:{engine.state.step}")
    return root


def load_tpu_worker_checkpoint(engine, source: str | Path) -> Path:
    """Restore the current rank after validating a committed topology."""

    root = Path(source)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Committed TPU checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_version = manifest.get("schema_version")
    if (
        isinstance(manifest_version, bool)
        or manifest_version not in SUPPORTED_SCHEMA_VERSIONS
        or manifest.get("status") != "committed"
    ):
        raise ValueError(
            "TPU checkpoint is not a committed supported-schema checkpoint."
        )
    if manifest.get("world_size") != engine.runtime.world_size:
        raise ValueError("TPU checkpoint world size does not match the active topology.")
    checkpoint_mesh = (
        _validated_mesh_axes(
            manifest.get("mesh_axes"), world_size=engine.runtime.world_size
        )
        if manifest_version >= 2
        else None
    )
    active_runtime_state = dict(engine.runtime.state_dict())
    active_mesh = _validated_mesh_axes(
        active_runtime_state.get("mesh_axes"), world_size=engine.runtime.world_size
    )
    if manifest_version >= 2 and checkpoint_mesh != active_mesh:
        raise ValueError("TPU checkpoint mesh axes do not match the active topology.")
    shards = manifest.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != engine.runtime.world_size
        or len(shards) != len(set(shards))
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            for name in shards
        )
    ):
        raise ValueError("TPU checkpoint manifest has an invalid shard list.")
    shard = root / shards[engine.runtime.rank]
    if not shard.is_file():
        raise FileNotFoundError(f"TPU checkpoint rank shard is missing: {shard}")
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    required_keys = (
        _REQUIRED_KEYS_V2 if manifest_version >= 2 else _REQUIRED_KEYS_V1
    )
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError("TPU checkpoint rank shard has an invalid state layout.")
    if payload["schema_version"] != manifest_version:
        raise ValueError("Unsupported TPU checkpoint rank-shard schema.")
    if payload["world_size"] != engine.runtime.world_size or payload["rank"] != engine.runtime.rank:
        raise ValueError("TPU checkpoint rank shard does not match the active topology.")
    runtime_payload = payload["runtime"]
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("TPU checkpoint runtime state must be a mapping.")
    payload_mesh = _validated_mesh_axes(
        (
            payload["mesh_axes"]
            if manifest_version >= 2
            else runtime_payload.get("mesh_axes")
        ),
        world_size=engine.runtime.world_size,
    )
    if manifest_version >= 2 and payload_mesh != checkpoint_mesh:
        raise ValueError("TPU checkpoint rank shard mesh does not match its manifest.")
    if payload_mesh != active_mesh:
        raise ValueError("TPU checkpoint mesh axes do not match the active topology.")
    if (
        payload["trainer"].get("step") != manifest.get("step")
        or payload["trainer"].get("micro_step") != manifest.get("micro_step")
    ):
        raise ValueError("TPU checkpoint shard progress does not match its manifest.")
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


def find_latest_committed_tpu_checkpoint(directory: str | Path) -> Path | None:
    """Return the newest durable checkpoint, ignoring all partial attempts."""

    root = Path(directory)
    if not root.is_dir():
        return None
    committed: list[tuple[int, int, str, Path]] = []
    for candidate in root.iterdir():
        manifest_path = candidate / "manifest.json"
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or not manifest_path.is_file()
        ):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        step = manifest.get("step")
        micro_step = manifest.get("micro_step")
        world_size = manifest.get("world_size")
        mesh_axes = manifest.get("mesh_axes")
        shards = manifest.get("shards")
        if (
            isinstance(manifest.get("schema_version"), bool)
            or manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS
            or manifest.get("status") != "committed"
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or isinstance(micro_step, bool)
            or not isinstance(micro_step, int)
            or micro_step < 0
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size < 1
            or (
                manifest.get("schema_version") >= 2
                and not _is_valid_mesh_axes(mesh_axes, world_size=world_size)
            )
            or not isinstance(shards, list)
            or len(shards) != world_size
            or len(shards) != len(set(shards))
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or not (candidate / name).is_file()
                for name in shards
            )
        ):
            continue
        committed.append((step, micro_step, candidate.name, candidate))
    if not committed:
        return None
    return max(committed)[-1]


def _is_valid_mesh_axes(value: Any, *, world_size: int) -> bool:
    try:
        _validated_mesh_axes(value, world_size=world_size)
    except ValueError:
        return False
    return True


__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "find_latest_committed_tpu_checkpoint",
    "load_tpu_worker_checkpoint",
    "save_tpu_worker_checkpoint",
]
