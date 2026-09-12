"""Public validated packed-binary dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
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
        source: HuggingFaceShardSourceConfig | str,
        *,
        sequence_length: int,
        revision: str | None = None,
        shard_range: tuple[int, int] | range | None = None,
        filename_template: str = "laughlm-v1/laughlm-v1_shard_{index:05d}.bin",
        header_bytes: int = 1024,
        vocab_size: int = 32011,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        **partition: Any,
    ) -> "PackedBinDataset":
        """Load manifest-backed shards or a numbered legacy ``.bin`` range.

        Passing a repo ID selects the LaughLM-compatible raw layout by default.
        ``shard_range`` is end-exclusive. A mutable Hub selector is resolved to
        one commit before files are downloaded, scanned, and staged for workers.
        """
        if isinstance(source, HuggingFaceShardSourceConfig):
            if revision is not None or shard_range is not None:
                raise ValueError(
                    "revision and shard_range are only used with a repo ID."
                )
            shards = HuggingFacePackedShardSource(source).resolve()
            resolved_revision = source.revision
        elif isinstance(source, str):
            shards, resolved_revision = cls._resolve_hub_bins(
                source,
                revision=revision,
                shard_range=shard_range,
                filename_template=filename_template,
                header_bytes=header_bytes,
                vocab_size=vocab_size,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            raise TypeError("source must be a Hugging Face dataset repo ID or config.")
        cls._require_unique_shards(shards)
        dataset = cls(shards, sequence_length=sequence_length, **partition)
        dataset._manifest_root = None
        dataset.hub_revision = resolved_revision
        return dataset

    @classmethod
    def _resolve_hub_bins(
        cls,
        repo_id: str,
        *,
        revision: str | None,
        shard_range: tuple[int, int] | range | None,
        filename_template: str,
        header_bytes: int,
        vocab_size: int,
        cache_dir: str | Path | None,
        local_files_only: bool,
    ) -> tuple[tuple[_ResolvedPackedShard, ...], str]:
        if shard_range is None:
            raise ValueError("shard_range=(start, stop) is required for raw Hub bins.")
        indices = cls._shard_indices(shard_range)
        resolved_revision = _resolve_hub_revision(
            repo_id, revision or "main", local_files_only=local_files_only
        )
        shards = []
        for index in indices:
            try:
                filename = filename_template.format(index=index)
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(
                    "filename_template must contain a valid {index} field."
                ) from exc
            data_file = _download_hub_file(
                repo_id=repo_id,
                filename=filename,
                revision=resolved_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            shard_id = Path(filename).stem
            manifest, validation = _inspect_legacy_uint16_bin(
                data_file,
                shard_id=shard_id,
                data_path=filename,
                header_bytes=header_bytes,
                vocab_size=vocab_size,
            )
            shards.append(_ResolvedPackedShard(
                shard_id=shard_id,
                manifest_file=data_file.with_suffix(".manifest.json"),
                data_file=data_file,
                document_index_file=None,
                manifest=manifest,
                validation=validation,
            ))
        return tuple(shards), resolved_revision

    @staticmethod
    def _shard_indices(value: tuple[int, int] | range) -> tuple[int, ...]:
        if isinstance(value, range):
            if value.step != 1:
                raise ValueError("shard_range must have a step of 1.")
            start, stop = value.start, value.stop
        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in value
            )
        ):
            start, stop = value
        else:
            raise TypeError("shard_range must be range(start, stop) or (start, stop).")
        if start < 0 or stop <= start:
            raise ValueError("shard_range must satisfy 0 <= start < stop.")
        return tuple(range(start, stop))

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


def _resolve_hub_revision(
    repo_id: str, revision: str, *, local_files_only: bool
) -> str:
    from huggingface_hub import HfApi

    if local_files_only and len(revision) != 40:
        raise ValueError("Offline Hub loading requires a 40-character commit SHA.")
    if len(revision) == 40 and all(
        character in "0123456789abcdef" for character in revision
    ):
        return revision
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    resolved = info.sha
    if (
        not isinstance(resolved, str)
        or len(resolved) != 40
        or any(character not in "0123456789abcdef" for character in resolved)
    ):
        raise ValueError("Hugging Face did not resolve the dataset to a commit SHA.")
    return resolved


def _download_hub_file(**kwargs: Any) -> Path:
    from huggingface_hub import hf_hub_download

    values = {
        "repo_type": "dataset",
        "library_name": "trainlm",
        **kwargs,
    }
    if values.get("cache_dir") is None:
        values.pop("cache_dir")
    path = Path(hf_hub_download(**values))
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face download did not resolve: {path}")
    return path


def _inspect_legacy_uint16_bin(
    path: Path,
    *,
    shard_id: str,
    data_path: str,
    header_bytes: int,
    vocab_size: int,
) -> tuple[PackedBinaryShardManifest, ValidatedPackedBinaryShard]:
    if (
        isinstance(header_bytes, bool)
        or not isinstance(header_bytes, int)
        or header_bytes < 0
    ):
        raise ValueError("header_bytes must be a non-negative integer.")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 1
    ):
        raise ValueError("vocab_size must be a positive integer.")
    file_size = path.stat().st_size
    payload_bytes = file_size - header_bytes
    if payload_bytes <= 0 or payload_bytes % 2:
        raise ValueError("Packed uint16 shard has invalid header/payload geometry.")
    digest = hashlib.sha256()
    minimum: int | None = None
    maximum: int | None = None
    with path.open("rb") as handle:
        remaining_header = header_bytes
        while remaining_header:
            chunk = handle.read(min(8 * 1024 * 1024, remaining_header))
            if not chunk:
                raise ValueError("Packed token shard ends inside its header.")
            digest.update(chunk)
            remaining_header -= len(chunk)
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            if len(chunk) % 2:
                raise ValueError("Packed uint16 shard ends with an incomplete token.")
            for (token,) in struct.iter_unpack("<H", chunk):
                minimum = token if minimum is None else min(minimum, token)
                maximum = token if maximum is None else max(maximum, token)
    if minimum is None or maximum is None:
        raise ValueError("Packed token shard contains no tokens.")
    if maximum >= vocab_size:
        raise ValueError(
            f"Packed token {maximum} is outside configured vocab_size={vocab_size}."
        )
    token_count = payload_bytes // 2
    checksum = digest.hexdigest()
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id=shard_id,
        data_path=data_path,
        compatibility_profile=(
            "legacy_1024_uint16" if header_bytes == 1024 else "explicit_v1"
        ),
        header_bytes=header_bytes,
        token_dtype="uint16",
        byte_order="little",
        token_count=token_count,
        token_id_min=minimum,
        token_id_max=maximum,
        vocab_size=vocab_size,
        file_size_bytes=file_size,
        sha256=checksum,
    )
    return manifest, ValidatedPackedBinaryShard(
        token_count=token_count,
        token_id_min=minimum,
        token_id_max=maximum,
        sha256=checksum,
        document_count=None,
    )
