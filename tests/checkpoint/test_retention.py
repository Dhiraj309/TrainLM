"""Retention bounds disk usage without removing unrelated or incomplete saves."""

from pathlib import Path
import shutil

import pytest

from trainlm import TrainLMPretrainingConfig, TrainLMTrainingArguments
from trainlm.checkpoint.retention import maintain_local_checkpoints


def _publish(root: Path, step: int, limit: int | None = None) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.pt").write_bytes(b"published checkpoint state")
    maintain_local_checkpoints(
        output_dir=root, just_saved=path, step=step, limit=limit
    )
    return path


@pytest.mark.parametrize("factory", [TrainLMTrainingArguments, TrainLMPretrainingConfig])
@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "3"])
def test_retention_rejects_invalid_limits(factory, limit):
    with pytest.raises(ValueError, match="save_total_limit"):
        factory(save_total_limit=limit)


def test_retention_rejects_unsupported_legacy_coordinator():
    with pytest.raises(ValueError, match="legacy XLA coordinator"):
        TrainLMTrainingArguments(accelerator="tpu", save_total_limit=2)


def test_retention_orders_steps_numerically_and_protects_the_new_save(tmp_path):
    for step in (1, 2, 9, 10):
        _publish(tmp_path, step, limit=2)
    assert {path.name for path in tmp_path.iterdir()} == {
        "checkpoint-9", "checkpoint-10"
    }

    # Restarting from an earlier checkpoint must not delete the new save itself.
    _publish(tmp_path, 3, limit=2)
    assert {path.name for path in tmp_path.iterdir()} == {
        "checkpoint-3", "checkpoint-10"
    }


def test_retention_preserves_unmarked_partial_custom_and_linked_directories(tmp_path):
    root = tmp_path / "checkpoints"
    old = _publish(root, 0)
    unmarked = _publish(root, 1)
    (unmarked / ".trainlm-checkpoint.json").unlink()
    partial = _publish(root, 2)
    (partial / "trainer_state.pt.tmp").write_bytes(b"interrupted write")
    export = _publish(root, 3)
    (export / "model.safetensors").write_bytes(b"user export")
    malformed = _publish(root, 4)
    (malformed / ".trainlm-checkpoint.json").write_text("invalid JSON")
    manual = root / "manual-backup"
    shutil.copytree(old, manual)
    outside = tmp_path / "outside"
    shutil.copytree(old, outside)
    (root / "checkpoint-5").symlink_to(outside, target_is_directory=True)
    linked_state = _publish(root, 6)
    (linked_state / "trainer_state.pt").unlink()
    (linked_state / "trainer_state.pt").symlink_to(outside / "trainer_state.pt")
    linked_marker = _publish(root, 7)
    (linked_marker / ".trainlm-checkpoint.json").unlink()
    (linked_marker / ".trainlm-checkpoint.json").symlink_to(
        outside / ".trainlm-checkpoint.json"
    )

    _publish(root, 10, limit=1)

    assert not old.exists()
    assert {path.name for path in root.iterdir()} == {
        *(f"checkpoint-{step}" for step in range(1, 8)),
        "manual-backup", "checkpoint-10",
    }
    assert (outside / "trainer_state.pt").read_bytes() == b"published checkpoint state"
    assert (partial / "trainer_state.pt.tmp").read_bytes() == b"interrupted write"
    assert (export / "model.safetensors").read_bytes() == b"user export"


def test_retention_does_not_run_for_an_external_or_unpublished_save(tmp_path):
    root = tmp_path / "checkpoints"
    _publish(root, 1)
    _publish(root, 2)
    outside = _publish(tmp_path / "elsewhere", 3)
    maintain_local_checkpoints(
        output_dir=root, just_saved=outside, step=3, limit=1
    )
    incomplete = root / "checkpoint-3"
    incomplete.mkdir()
    (incomplete / "trainer_state.pt.tmp").write_bytes(b"pending")
    maintain_local_checkpoints(
        output_dir=root, just_saved=incomplete, step=3, limit=1
    )
    assert (root / "checkpoint-1" / "trainer_state.pt").is_file()
    assert (root / "checkpoint-2" / "trainer_state.pt").is_file()


def test_retention_reports_cleanup_failure_and_keeps_new_save(tmp_path, monkeypatch):
    _publish(tmp_path, 1)

    def cannot_delete(path):
        raise PermissionError("read-only directory")

    monkeypatch.setattr("trainlm.checkpoint.retention.shutil.rmtree", cannot_delete)
    with pytest.warns(RuntimeWarning, match="new checkpoint is saved"):
        current = _publish(tmp_path, 2, limit=1)
    assert (current / "trainer_state.pt").is_file()
    assert (tmp_path / "checkpoint-1" / "trainer_state.pt").is_file()
