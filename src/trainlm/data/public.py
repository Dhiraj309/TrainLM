"""Public validated packed-binary dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from torch.utils.data import IterableDataset, get_worker_info

from .huggingface import HuggingFacePackedShardSource, HuggingFaceShardSourceConfig
from .manifest import (
    PackedBinaryShardManifest,
    ValidatedPackedBinaryShard,
    validate_packed_binary_shard,
)
from .partition import PartitionedPackedBatchReader, plan_packed_batch_partition
from .reader import ContiguousPackedBatchReader


@dataclass(frozen=True, slots=True)
class _ResolvedPackedShard:
    shard_id: str
    manifest_file: Path
    data_file: Path
    document_index_file: Path | None
    manifest: PackedBinaryShardManifest
    validation: ValidatedPackedBinaryShard


class PackedBinDataset(IterableDataset):
    """Validated fixed-length causal-LM examples from packed ``.bin`` shards.

    Use :meth:`from_directory` for local manifests or :meth:`from_hub` for a
    revision-pinned Hugging Face dataset source. Partition parameters describe
    host/rank ownership and are deterministic for a given seed and epoch.
    """

    def __init__(
        self,
        shards: Iterable[Any],
        *,
        sequence_length: int,
        split: str = "train",
        seed: int = 42,
        epoch: int = 0,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__()
        self.shards = tuple(shards)
        self.sequence_length = sequence_length
        self._reader = ContiguousPackedBatchReader(
            self.shards, batch_size=1, sequence_length=sequence_length
        )
        plan = plan_packed_batch_partition(
            self._reader,
            split=split,
            seed=seed if split == "train" else 0,
            epoch=epoch if split == "train" else 0,
            world_size=world_size,
            rank=rank,
        )
        self._partition = PartitionedPackedBatchReader(self._reader, plan)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        sequence_length: int,
        manifest_pattern: str = "*.manifest.json",
        **partition: Any,
    ) -> "PackedBinDataset":
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Packed dataset directory does not exist: {root}")
        manifest_files = tuple(sorted(root.glob(manifest_pattern)))
        if not manifest_files:
            raise FileNotFoundError(
                f"No packed shard manifests matching {manifest_pattern!r} in {root}"
            )
        shards = tuple(cls._resolve_local_manifest(root, path) for path in manifest_files)
        cls._require_unique_shards(shards)
        dataset = cls(shards, sequence_length=sequence_length, **partition)
        dataset._manifest_root = root
        return dataset

    @classmethod
    def from_hub(
        cls,
        source: HuggingFaceShardSourceConfig,
        *,
        sequence_length: int,
        **partition: Any,
    ) -> "PackedBinDataset":
        shards = HuggingFacePackedShardSource(source).resolve()
        cls._require_unique_shards(shards)
        dataset = cls(shards, sequence_length=sequence_length, **partition)
        dataset._manifest_root = None
        return dataset

    def __len__(self) -> int:
        return len(self._partition)

    def __iter__(self):
        worker = get_worker_info()
        start = 0 if worker is None else worker.id
        stride = 1 if worker is None else worker.num_workers
        for index in range(start, len(self._partition), stride):
            batch = self._partition.read_batch(index)
            yield {
                "input_ids": batch["input_ids"].squeeze(0),
                "labels": batch["labels"].squeeze(0),
                "attention_mask": batch["attention_mask"].squeeze(0),
                "loss_mask": batch["loss_mask"].squeeze(0),
            }

    def close(self) -> None:
        self._reader.close()

    def coordinator_manifest_dir(self, output_dir: str | Path) -> Path:
        """Return a validated local manifest tree for the private TPU worker."""

        if getattr(self, "_manifest_root", None) is not None:
            return self._manifest_root
        root = Path(output_dir) / ".trainlm_data"
        root.mkdir(parents=True, exist_ok=True)
        for index, shard in enumerate(self.shards):
            manifest_name = f"shard_{index:05d}.manifest.json"
            manifest_path = root / manifest_name
            data_path = root / shard.manifest.data_path
            self._link_or_copy(Path(shard.data_file), data_path)
            document_path = None
            if shard.manifest.documents.path is not None:
                document_path = root / shard.manifest.documents.path
                self._link_or_copy(Path(shard.document_index_file), document_path)
            manifest_path.write_text(shard.manifest.to_json() + "\n", encoding="utf-8")
        return root

    @staticmethod
    def _resolve_local_manifest(root: Path, path: Path) -> _ResolvedPackedShard:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            manifest = PackedBinaryShardManifest.from_dict(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid packed shard manifest: {path}") from exc
        data_file = root / manifest.data_path
        document_file = (
            root / manifest.documents.path
            if manifest.documents.path is not None
            else None
        )
        validation = validate_packed_binary_shard(
            manifest, data_file, document_index_file=document_file
        )
        return _ResolvedPackedShard(
            shard_id=manifest.shard_id,
            manifest_file=path,
            data_file=data_file,
            document_index_file=document_file,
            manifest=manifest,
            validation=validation,
        )

    @staticmethod
    def _require_unique_shards(shards: tuple[Any, ...]) -> None:
        ids = tuple(shard.shard_id for shard in shards)
        if len(ids) != len(set(ids)):
            raise ValueError("Packed dataset shard IDs must be unique.")

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.resolve() == source.resolve():
                return
            destination.unlink()
        try:
            os.symlink(source.resolve(), destination)
        except OSError:
            shutil.copy2(source, destination)


__all__ = ["PackedBinDataset"]
