from dataclasses import dataclass
import json

import pytest
import torch

from trainlm._tpu_checkpoint import (
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
    return Engine(model, optimizer, scheduler, Runtime(), TrainerState(step=3, micro_step=6))


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
