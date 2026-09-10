from dataclasses import dataclass
import json
import multiprocessing
import os

import pytest
import torch

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
            "device_rng_state": 17,
        }

    def load_state_dict(self, state):
        self.loaded = state


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
