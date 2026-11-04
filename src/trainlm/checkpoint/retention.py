"""Conservative retention for the public trainer's local checkpoint files."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import warnings


_CHECKPOINT_NAME = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
_MARKER = ".trainlm-checkpoint.json"
_STATE = "trainer_state.pt"


def _marker(step: int) -> dict[str, object]:
    return {"schema_version": 1, "kind": "trainlm_local_checkpoint", "step": step}


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None or path.is_symlink() or not path.is_dir():
        return None
    return int(match[1])


def _has_only_checkpoint_files(path: Path) -> bool:
    """Keep incomplete saves, exports, user files, and symlinks out of cleanup."""
    entries = tuple(path.iterdir())
    return (
        {entry.name for entry in entries} <= {_STATE, _MARKER}
        and all(entry.is_file() and not entry.is_symlink() for entry in entries)
        and (path / _STATE).is_file()
        and (path / _STATE).stat().st_size > 0
    )


def _owned_checkpoint_step(path: Path) -> int | None:
    try:
        step = _checkpoint_step(path)
        if step is None or not _has_only_checkpoint_files(path):
            return None
        marker = path / _MARKER
        if not marker.is_file() or marker.stat().st_size > 1024:
            return None
        if json.loads(marker.read_text(encoding="utf-8")) != _marker(step):
            return None
        return step
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def maintain_local_checkpoints(
    *, output_dir: Path, just_saved: Path, step: int, limit: int | None
) -> None:
    """Mark a published save and prune older owned directories within its root.

    The caller must finish publishing checkpoint state and run metadata first.
    Only single-writer, numbered, local training saves participate. Separately
    named manual saves and unmarked legacy directories are left intact. The new
    save is protected even when a caller resumes from an older step.
    """
    try:
        if (
            just_saved.parent.resolve() != output_dir.resolve()
            or _checkpoint_step(just_saved) != step
            or not _has_only_checkpoint_files(just_saved)
        ):
            return
        marker = just_saved / _MARKER
        temporary = marker.with_name(marker.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(_marker(step), sort_keys=True) + "\n")
        temporary.replace(marker)
        if limit is None:
            return

        candidates = []
        for path in output_dir.iterdir():
            checkpoint_step = _owned_checkpoint_step(path)
            if checkpoint_step is not None and path.name != just_saved.name:
                candidates.append((checkpoint_step, path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        # The current save takes one slot; preserve the newest remaining saves.
        for _, path in candidates[limit - 1:]:
            if _owned_checkpoint_step(path) is not None:
                shutil.rmtree(path)
    except OSError as exc:
        warnings.warn(
            f"Checkpoint retention could not finish ({type(exc).__name__}); "
            "the new checkpoint is saved. Check directory permissions or "
            "remove older checkpoints manually.",
            RuntimeWarning,
            stacklevel=2,
        )
