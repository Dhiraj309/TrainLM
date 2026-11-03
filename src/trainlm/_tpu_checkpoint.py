"""Rank-local checkpoint persistence used by private TPU workers."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import torch

from trainlm.training import TrainerPhase


SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, SCHEMA_VERSION})
_REQUIRED_KEYS_V1 = {
    "schema_version", "world_size", "rank", "model", "optimizer",
    "scheduler", "runtime", "trainer", "cpu_rng_state",
}
_REQUIRED_KEYS_V2 = {
    "schema_version", "world_size", "mesh_axes", "rank", "model", "optimizer",
    "scheduler", "runtime", "trainer", "cpu_rng_state",
}
_REQUIRED_KEYS_V3 = _REQUIRED_KEYS_V2


def _checkpoint_file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size_bytes += len(chunk)
            digest.update(chunk)
    return {
        "name": path.name,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _validated_shard_integrity(
    manifest: Mapping[str, Any], shards: list[str], *, world_size: int
) -> list[dict[str, Any]]:
    integrity = manifest.get("shard_integrity")
    if not isinstance(integrity, list) or len(integrity) != world_size:
        raise ValueError(
            "TPU checkpoint manifest has invalid shard integrity data."
        )
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(integrity):
        if not isinstance(item, Mapping) or set(item) != {
            "name", "size_bytes", "sha256"
        }:
            raise ValueError(
                "TPU checkpoint manifest has invalid shard integrity data."
            )
        name = item["name"]
        size_bytes = item["size_bytes"]
        sha256 = item["sha256"]
        if (
            name != shards[index]
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ValueError("TPU checkpoint manifest has invalid shard integrity data.")
        validated.append(dict(item))
    return validated


def _verify_shard_integrity(path: Path, metadata: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"TPU checkpoint rank shard is missing: {path}")
    if path.stat().st_size != metadata["size_bytes"]:
        raise ValueError(f"TPU checkpoint rank shard size mismatch: {path.name}")
    actual = _checkpoint_file_metadata(path)["sha256"]
    if actual != metadata["sha256"]:
        raise ValueError(f"TPU checkpoint rank shard checksum mismatch: {path.name}")


def _all_ranks_succeeded(engine, local_success: bool, *, phase: str) -> bool:
    """Synchronize checkpoint phase success so one rank cannot strand peers."""
    world_size = engine.runtime.world_size
    if world_size == 1:
        return local_success
    if not getattr(engine.runtime, "is_distributed", False):
        raise RuntimeError(
            "A multi-rank checkpoint requires a distributed runtime collective."
        )
    reducer = getattr(engine.runtime, "reduce_sum", None)
    if not callable(reducer):
        raise RuntimeError(
            "A multi-rank checkpoint runtime must implement reduce_sum."
        )
    marker = torch.tensor(
        int(local_success), dtype=torch.int32, device=engine.runtime.device
    )
    succeeded = reducer(marker)
    if not isinstance(succeeded, torch.Tensor) or succeeded.ndim != 0:
        raise RuntimeError(
            f"The runtime returned an invalid checkpoint {phase} status reduction."
        )
    return int(succeeded.item()) == world_size


def _raise_checkpoint_phase_error(
    engine, error: Exception | None, *, phase: str
) -> None:
    if error is not None and engine.runtime.world_size == 1:
        raise error
    if error is not None:
        raise RuntimeError(
            f"Checkpoint {phase} failed on rank {engine.runtime.rank}: {error}."
        ) from error
    raise RuntimeError(
        f"Checkpoint {phase} failed on one or more ranks."
    )


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
    rank = engine.runtime.rank
    world_size = engine.runtime.world_size
    generation = f"step-{engine.state.step:012d}-micro-{engine.state.micro_step:012d}"
    shard = root / f"{generation}-rank-{rank:05d}.pt"
    local_error: Exception | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.json"
        if manifest.exists():
            raise FileExistsError(
                f"Refusing to overwrite committed checkpoint: {manifest}"
            )
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
        temporary = shard.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        stage("shard-staged", temporary)
        temporary.replace(shard)
        stage("shard-published", shard)
    except Exception as exc:
        local_error = exc
    if not _all_ranks_succeeded(engine, local_error is None, phase="shard write"):
        _raise_checkpoint_phase_error(engine, local_error, phase="shard write")

    engine.runtime.barrier(f"checkpoint-shards:{engine.state.step}")
    manifest_error: Exception | None = None
    if engine.runtime.is_primary_process:
        try:
            manifest = root / "manifest.json"
            shard_names = [
                f"{generation}-rank-{index:05d}.pt"
                for index in range(world_size)
            ]
            missing = [
                index for index, name in enumerate(shard_names)
                if (root / name).is_symlink() or not (root / name).is_file()
            ]
            if missing:
                raise RuntimeError(f"Checkpoint is missing rank shards: {missing}")
            shard_integrity = [
                _checkpoint_file_metadata(root / name) for name in shard_names
            ]
            manifest_tmp = manifest.with_suffix(".json.tmp")
            manifest_tmp.write_text(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "status": "committed",
                "world_size": world_size,
                "mesh_axes": mesh_axes,
                "step": engine.state.step,
                "micro_step": engine.state.micro_step,
                "generation": generation,
                "shards": shard_names,
                "shard_integrity": shard_integrity,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stage("manifest-staged", manifest_tmp)
            manifest_tmp.replace(manifest)
            stage("manifest-published", manifest)
        except Exception as exc:
            manifest_error = exc
    if not _all_ranks_succeeded(
        engine, manifest_error is None, phase="manifest publication"
    ):
        _raise_checkpoint_phase_error(
            engine, manifest_error, phase="manifest publication"
        )
    engine.runtime.barrier(f"checkpoint-published:{engine.state.step}")
    return root


def load_tpu_worker_checkpoint(engine, source: str | Path) -> Path:
    """Restore the current rank after validating a committed topology."""

    root = Path(source)
    payload: dict[str, Any] | None = None
    local_error: Exception | None = None
    try:
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Committed TPU checkpoint manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("TPU checkpoint manifest must be a JSON object.")
        manifest_version = manifest.get("schema_version")
        if (
            isinstance(manifest_version, bool)
            or not isinstance(manifest_version, int)
            or manifest_version not in SUPPORTED_SCHEMA_VERSIONS
            or manifest.get("status") != "committed"
        ):
            raise ValueError(
                "TPU checkpoint is not a committed supported-schema checkpoint."
            )
        world_size = engine.runtime.world_size
        if manifest.get("world_size") != world_size:
            raise ValueError(
                "TPU checkpoint world size does not match the active topology."
            )
        checkpoint_mesh = (
            _validated_mesh_axes(
                manifest.get("mesh_axes"), world_size=world_size
            )
            if manifest_version >= 2
            else None
        )
        active_runtime_state = dict(engine.runtime.state_dict())
        active_mesh = _validated_mesh_axes(
            active_runtime_state.get("mesh_axes"), world_size=world_size
        )
        if manifest_version >= 2 and checkpoint_mesh != active_mesh:
            raise ValueError(
                "TPU checkpoint mesh axes do not match the active topology."
            )
        shards = manifest.get("shards")
        if (
            not isinstance(shards, list)
            or len(shards) != world_size
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                for name in shards
            )
            or len(shards) != len(set(shards))
        ):
            raise ValueError("TPU checkpoint manifest has an invalid shard list.")
        shard = root / shards[engine.runtime.rank]
        if manifest_version >= 3:
            integrity = _validated_shard_integrity(
                manifest, shards, world_size=world_size
            )
            _verify_shard_integrity(shard, integrity[engine.runtime.rank])
        elif shard.is_symlink() or not shard.is_file():
            raise FileNotFoundError(f"TPU checkpoint rank shard is missing: {shard}")
        candidate = torch.load(shard, map_location="cpu", weights_only=False)
        required_keys = (
            _REQUIRED_KEYS_V3
            if manifest_version >= 3
            else _REQUIRED_KEYS_V2
            if manifest_version >= 2
            else _REQUIRED_KEYS_V1
        )
        if not isinstance(candidate, dict) or set(candidate) != required_keys:
            raise ValueError("TPU checkpoint rank shard has an invalid state layout.")
        if candidate["schema_version"] != manifest_version:
            raise ValueError("Unsupported TPU checkpoint rank-shard schema.")
        if (
            candidate["world_size"] != world_size
            or candidate["rank"] != engine.runtime.rank
        ):
            raise ValueError(
                "TPU checkpoint rank shard does not match the active topology."
            )
        runtime_payload = candidate["runtime"]
        trainer_payload = candidate["trainer"]
        model_payload = candidate["model"]
        if not isinstance(runtime_payload, Mapping):
            raise ValueError("TPU checkpoint runtime state must be a mapping.")
        if not isinstance(trainer_payload, Mapping):
            raise ValueError("TPU checkpoint trainer state must be a mapping.")
        if not isinstance(model_payload, Mapping):
            raise ValueError("TPU checkpoint model state must be a mapping.")
        payload_mesh = _validated_mesh_axes(
            (
                candidate["mesh_axes"]
                if manifest_version >= 2
                else runtime_payload.get("mesh_axes")
            ),
            world_size=world_size,
        )
        if manifest_version >= 2 and payload_mesh != checkpoint_mesh:
            raise ValueError(
                "TPU checkpoint rank shard mesh does not match its manifest."
            )
        if payload_mesh != active_mesh:
            raise ValueError(
                "TPU checkpoint mesh axes do not match the active topology."
            )
        if (
            trainer_payload.get("step") != manifest.get("step")
            or trainer_payload.get("micro_step") != manifest.get("micro_step")
        ):
            raise ValueError(
                "TPU checkpoint shard progress does not match its manifest."
            )
        if "device_rng_state" not in runtime_payload:
            raise ValueError("TPU checkpoint rank shard is missing device RNG state.")
        if set(model_payload) != set(engine.model.state_dict()):
            raise ValueError(
                "TPU checkpoint model keys do not match the active model."
            )
        for key, current_value in engine.model.state_dict().items():
            saved_value = model_payload[key]
            current_shape = getattr(current_value, "shape", None)
            saved_shape = getattr(saved_value, "shape", None)
            if current_shape is not None and saved_shape != current_shape:
                raise ValueError(
                    f"TPU checkpoint model tensor shape mismatch for {key!r}."
                )
        for key in trainer_payload:
            if not hasattr(engine.state, key):
                raise ValueError(
                    f"Unknown trainer-state field in TPU checkpoint: {key}"
                )
        payload = candidate
    except Exception as exc:
        local_error = exc

    if not _all_ranks_succeeded(
        engine, local_error is None, phase="resume preflight"
    ):
        _raise_checkpoint_phase_error(engine, local_error, phase="resume preflight")
    assert payload is not None
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
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        schema_version = manifest.get("schema_version")
        step = manifest.get("step")
        micro_step = manifest.get("micro_step")
        world_size = manifest.get("world_size")
        mesh_axes = manifest.get("mesh_axes")
        shards = manifest.get("shards")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in SUPPORTED_SCHEMA_VERSIONS
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
                schema_version >= 2
                and not _is_valid_mesh_axes(mesh_axes, world_size=world_size)
            )
            or not isinstance(shards, list)
            or len(shards) != world_size
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                for name in shards
            )
            or len(shards) != len(set(shards))
            or any(
                (candidate / name).is_symlink()
                or not (candidate / name).is_file()
                for name in shards
            )
        ):
            continue
        if schema_version >= 3:
            try:
                _validated_shard_integrity(manifest, shards, world_size=world_size)
            except ValueError:
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
