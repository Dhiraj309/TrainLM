from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import socket

import pytest
import torch
import torch.distributed as dist

from trainlm._tpu_checkpoint import (
    find_latest_committed_tpu_checkpoint,
    load_tpu_worker_checkpoint,
    save_tpu_worker_checkpoint,
)
from trainlm.training import TrainerPhase, TrainerState


class Runtime:
    rank = 0
    world_size = 1
    is_primary_process = True

    def __init__(self):
        self.loaded = None

    def barrier(self, name):
        del name

    def state_dict(self):
        return {
            "backend": "xla", "rank": 0, "world_size": 1,
            "mesh_axes": {"data": 1},
            "device_rng_state": 17,
        }

    def load_state_dict(self, state):
        self.loaded = state


class GlooRuntime(Runtime):
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.is_primary_process = rank == 0
        self.is_distributed = True
        self.device = torch.device("cpu")

    def barrier(self, name):
        del name
        dist.barrier()

    def reduce_sum(self, value):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def state_dict(self):
        return {
            "backend": "xla",
            "rank": self.rank,
            "world_size": self.world_size,
            "mesh_axes": {"data": self.world_size},
            "device_rng_state": 17 + self.rank,
        }


@dataclass
class Engine:
    model: object
    optimizer: object
    scheduler: object
    runtime: Runtime
    state: TrainerState


def engine():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return Engine(
        model,
        optimizer,
        scheduler,
        Runtime(),
        TrainerState(step=3, micro_step=6),
    )


def _save_until_process_kill(destination, stage):
    def kill_at(current_stage, _path):
        if current_stage == stage:
            os._exit(91)

    save_tpu_worker_checkpoint(engine(), destination, _stage_hook=kill_at)


def _kill_before_checkpoint(destination):
    destination.mkdir(parents=True)
    os._exit(92)


def _gloo_checkpoint_worker(
    rank,
    world_size,
    port,
    destination,
    fail_rank,
    corrupt_rank,
    fail_manifest,
    results,
):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(101)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        current = Engine(
            model,
            optimizer,
            scheduler,
            GlooRuntime(rank, world_size),
            TrainerState(step=3, micro_step=6),
        )
        expected = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

        def fail_one_rank(stage, _path):
            if rank == fail_rank and stage == "shard-published":
                raise OSError("injected rank-local write failure")
            if rank == 0 and fail_manifest and stage == "manifest-staged":
                raise OSError("injected manifest publication failure")

        try:
            save_tpu_worker_checkpoint(
                current,
                destination,
                _stage_hook=(
                    fail_one_rank
                    if fail_rank is not None or fail_manifest
                    else None
                ),
            )
        except Exception as exc:
            results.put(("failed-save", rank, type(exc).__name__, str(exc)))
            return
        if fail_rank is not None:
            results.put(("unexpected-save-success", rank))
            return
        if corrupt_rank is not None:
            if rank == 0:
                manifest = json.loads(
                    (Path(destination) / "manifest.json").read_text()
                )
                shard = Path(destination) / manifest["shards"][corrupt_rank]
                damaged = bytearray(shard.read_bytes())
                damaged[-1] ^= 1
                shard.write_bytes(damaged)
            dist.barrier()
        with torch.no_grad():
            for parameter in current.model.parameters():
                parameter.zero_()
        current.state.step = 0
        current.state.micro_step = 0
        before = {
            key: value.detach().clone()
            for key, value in current.model.state_dict().items()
        }
        try:
            load_tpu_worker_checkpoint(current, destination)
        except Exception as exc:
            unchanged = all(
                torch.equal(before[key], value)
                for key, value in current.model.state_dict().items()
            )
            results.put(
                ("failed-load", rank, str(exc), unchanged, current.state.step)
            )
        else:
            restored = all(
                torch.equal(expected[key], value)
                for key, value in current.model.state_dict().items()
            )
            results.put(("restored", rank, restored, current.state.step))
    finally:
        dist.destroy_process_group()


def _run_gloo_checkpoint_workers(
    tmp_path, *, fail_rank=None, corrupt_rank=None, fail_manifest=False
):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    destination = str(tmp_path / "checkpoint-gloo")
    processes = [
        context.Process(
            target=_gloo_checkpoint_worker,
            args=(
                rank,
                2,
                port,
                destination,
                fail_rank,
                corrupt_rank,
                fail_manifest,
                results,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)
    assert not alive, "distributed checkpoint workers did not finish together"
    assert [process.exitcode for process in processes] == [0, 0]
    messages = [results.get(timeout=5) for _ in processes]
    return Path(destination), messages


def test_rank_checkpoint_round_trip(tmp_path):
    original = engine()
    expected = {key: value.detach().clone() for key, value in original.model.state_dict().items()}
    destination = save_tpu_worker_checkpoint(original, tmp_path / "checkpoint-3")
    with torch.no_grad():
        for parameter in original.model.parameters():
            parameter.zero_()
    original.state.step = 0
    original.state.micro_step = 0

    load_tpu_worker_checkpoint(original, destination)

    assert original.state.step == 3
    assert original.state.micro_step == 6
    assert original.state.phase is TrainerPhase.RESUMING
    assert all(torch.equal(expected[key], value) for key, value in original.model.state_dict().items())
    assert json.loads((destination / "manifest.json").read_text())["status"] == "committed"


def test_v3_checkpoint_records_and_checks_rank_shard_integrity(tmp_path):
    current = engine()
    destination = save_tpu_worker_checkpoint(current, tmp_path / "checkpoint-v3")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert len(manifest["shard_integrity"]) == 1
    assert manifest["shard_integrity"][0]["name"] == manifest["shards"][0]

    shard = destination / manifest["shards"][0]
    damaged = bytearray(shard.read_bytes())
    damaged[-1] ^= 1
    shard.write_bytes(damaged)
    with torch.no_grad():
        for parameter in current.model.parameters():
            parameter.zero_()
    before = {key: value.detach().clone() for key, value in current.model.state_dict().items()}

    with pytest.raises(ValueError, match="checksum"):
        load_tpu_worker_checkpoint(current, destination)

    assert all(
        torch.equal(before[key], value)
        for key, value in current.model.state_dict().items()
    )


def test_distributed_checkpoint_round_trip_synchronizes_all_ranks(tmp_path):
    destination, messages = _run_gloo_checkpoint_workers(tmp_path)

    assert sorted(message[:3] for message in messages) == [
        ("restored", 0, True),
        ("restored", 1, True),
    ]
    assert all(message[3] == 3 for message in messages)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["world_size"] == 2
    assert len(manifest["shard_integrity"]) == 2


def test_distributed_shard_failure_aborts_without_hanging_peers(tmp_path):
    destination, messages = _run_gloo_checkpoint_workers(tmp_path, fail_rank=1)

    assert all(message[0] == "failed-save" for message in messages)
    assert any("injected rank-local write failure" in message[3] for message in messages)
    assert not (destination / "manifest.json").exists()


def test_distributed_manifest_publication_failure_reaches_all_ranks(tmp_path):
    destination, messages = _run_gloo_checkpoint_workers(
        tmp_path, fail_manifest=True
    )

    assert all(message[0] == "failed-save" for message in messages)
    assert any("injected manifest publication failure" in message[3] for message in messages)
    assert not (destination / "manifest.json").exists()


def test_distributed_resume_preflights_all_shards_before_restoring(tmp_path):
    _destination, messages = _run_gloo_checkpoint_workers(tmp_path, corrupt_rank=1)

    assert all(message[0] == "failed-load" for message in messages)
    assert all(message[3] is True for message in messages)
    assert all(message[4] == 0 for message in messages)
    assert any("checksum mismatch" in message[2] for message in messages)


def test_resume_rejects_incomplete_or_wrong_topology(tmp_path):
    current = engine()
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_tpu_worker_checkpoint(current, tmp_path / "missing")

    destination = save_tpu_worker_checkpoint(current, tmp_path / "checkpoint")
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest["world_size"] = 8
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="world size"):
        load_tpu_worker_checkpoint(current, destination)


def test_resume_rejects_same_world_size_with_different_mesh(tmp_path):
    current = engine()
    destination = save_tpu_worker_checkpoint(current, tmp_path / "checkpoint")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mesh_axes"] = {"data": 1, "fsdp": 1}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="mesh axes"):
        load_tpu_worker_checkpoint(current, destination)


def test_resume_migrates_legacy_checkpoint_using_rank_runtime_mesh(tmp_path):
    current = engine()
    destination = save_tpu_worker_checkpoint(current, tmp_path / "checkpoint")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest.pop("mesh_axes")
    manifest_path.write_text(json.dumps(manifest))
    shard_path = destination / manifest["shards"][0]
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 1
    payload.pop("mesh_axes")
    torch.save(payload, shard_path)

    load_tpu_worker_checkpoint(current, destination)

    assert current.state.step == 3
    assert current.runtime.loaded["mesh_axes"] == {"data": 1}


def test_legacy_resume_rejects_rank_runtime_mesh_mismatch(tmp_path):
    current = engine()
    destination = save_tpu_worker_checkpoint(current, tmp_path / "checkpoint")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest.pop("mesh_axes")
    manifest_path.write_text(json.dumps(manifest))
    shard_path = destination / manifest["shards"][0]
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 1
    payload.pop("mesh_axes")
    payload["runtime"]["mesh_axes"] = {"fsdp": 1}
    torch.save(payload, shard_path)

    with pytest.raises(ValueError, match="mesh axes"):
        load_tpu_worker_checkpoint(current, destination)


def test_recovery_rejects_mesh_geometry_that_disagrees_with_world_size(tmp_path):
    root = tmp_path / "checkpoints"
    destination = save_tpu_worker_checkpoint(engine(), root / "checkpoint-3")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mesh_axes"] = {"data": 2}
    manifest_path.write_text(json.dumps(manifest))

    assert find_latest_committed_tpu_checkpoint(root) is None


def test_recovery_ignores_non_object_and_invalid_schema_manifests(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    non_object = root / "checkpoint-non-object"
    non_object.mkdir()
    (non_object / "manifest.json").write_text("[]")
    invalid_schema = root / "checkpoint-invalid-schema"
    invalid_schema.mkdir()
    (invalid_schema / "manifest.json").write_text(
        json.dumps({"schema_version": [3], "status": "committed"})
    )
    invalid_shards = root / "checkpoint-invalid-shards"
    invalid_shards.mkdir()
    (invalid_shards / "manifest.json").write_text(
        json.dumps({
            "schema_version": 3,
            "status": "committed",
            "step": 1,
            "micro_step": 1,
            "world_size": 1,
            "mesh_axes": {"data": 1},
            "shards": [["nested-name"]],
        })
    )

    assert find_latest_committed_tpu_checkpoint(root) is None


def test_recovery_ignores_compute_staging_and_incomplete_persistence(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "compute-interrupted").mkdir()
    staging = root / "checkpoint-2"
    staging.mkdir()
    (staging / "step-000000000002-micro-000000000004-rank-00000.pt.tmp").write_bytes(
        b"partial"
    )

    durable = save_tpu_worker_checkpoint(engine(), root / "checkpoint-3")

    assert find_latest_committed_tpu_checkpoint(root) == durable


@pytest.mark.parametrize(
    "stage",
    ["shard-staged", "shard-published", "manifest-staged"],
)
def test_recovery_survives_real_process_kill_during_persistence(tmp_path, stage):
    root = tmp_path / "checkpoints"
    durable = save_tpu_worker_checkpoint(engine(), root / "checkpoint-3")
    interrupted = root / f"checkpoint-interrupted-{stage}"
    process = multiprocessing.get_context("spawn").Process(
        target=_save_until_process_kill,
        args=(interrupted, stage),
    )

    process.start()
    process.join(timeout=30)

    assert process.exitcode == 91
    assert find_latest_committed_tpu_checkpoint(root) == durable
    assert not (interrupted / "manifest.json").exists()


def test_recovery_survives_real_process_kill_during_compute(tmp_path):
    root = tmp_path / "checkpoints"
    durable = save_tpu_worker_checkpoint(engine(), root / "checkpoint-3")
    interrupted = root / "checkpoint-compute-interrupted"
    process = multiprocessing.get_context("spawn").Process(
        target=_kill_before_checkpoint,
        args=(interrupted,),
    )

    process.start()
    process.join(timeout=30)

    assert process.exitcode == 92
    assert find_latest_committed_tpu_checkpoint(root) == durable


def test_manifest_publish_is_the_durable_commit_boundary(tmp_path):
    root = tmp_path / "checkpoints"
    save_tpu_worker_checkpoint(engine(), root / "checkpoint-2")
    published = root / "checkpoint-3-published"
    process = multiprocessing.get_context("spawn").Process(
        target=_save_until_process_kill,
        args=(published, "manifest-published"),
    )

    process.start()
    process.join(timeout=30)

    assert process.exitcode == 91
    assert find_latest_committed_tpu_checkpoint(root) == published


def test_committed_checkpoint_is_immutable_and_progress_is_verified(tmp_path):
    current = engine()
    destination = save_tpu_worker_checkpoint(current, tmp_path / "checkpoint-3")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_tpu_worker_checkpoint(current, destination)

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["micro_step"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="progress"):
        load_tpu_worker_checkpoint(current, destination)


def test_recovery_rejects_manifest_with_missing_or_unsafe_shards(tmp_path):
    root = tmp_path / "checkpoints"
    destination = save_tpu_worker_checkpoint(engine(), root / "checkpoint-3")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"] = ["../rank-00000.pt"]
    manifest_path.write_text(json.dumps(manifest))

    assert find_latest_committed_tpu_checkpoint(root) is None
