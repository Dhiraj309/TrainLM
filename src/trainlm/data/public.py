"""Public validated packed-binary dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .cursor import PackedDataCursor, PackedDataCursorState
from .gcs import is_gcs_uri, stage_gcs_prefix
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
    target_ids_file: Path | None
    loss_mask_file: Path | None
    manifest: PackedBinaryShardManifest
    validation: ValidatedPackedBinaryShard


class PackedBinDataset(IterableDataset):
    """Validated fixed-length causal-LM examples from packed ``.bin`` shards.

    Use :meth:`from_directory` for local manifests, :meth:`from_gcs` for
    same-region ``gs://`` buckets, or :meth:`from_hub` for a revision-pinned
    Hugging Face dataset source. Partition parameters describe host/rank
    ownership and are deterministic for a given seed and epoch.
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
        self._cursor: PackedDataCursor | None = None
        self._source_revision: str | None = None
        self._supervised_tokens_per_epoch_cache: int | None = None

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        sequence_length: int,
        manifest_pattern: str = "*.manifest.json",
        **partition: Any,
    ) -> "PackedBinDataset":
        if is_gcs_uri(directory):
            return cls.from_gcs(
                str(directory),
                sequence_length=sequence_length,
                manifest_pattern=manifest_pattern,
                **partition,
            )
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
    def from_gcs(
        cls,
        gcs_uri: str,
        *,
        sequence_length: int,
        manifest_pattern: str = "*.manifest.json",
        cache_dir: str | Path | None = None,
        max_shards: int | None = None,
        header_bytes: int = 1024,
        vocab_size: int = 32011,
        filesystem: Any = None,
        runner: Any = None,
        **partition: Any,
    ) -> "PackedBinDataset":
        """Stage and mmap packed shards from a same-region ``gs://bucket/prefix`` URI."""
        local_root = stage_gcs_prefix(
            gcs_uri,
            cache_dir=cache_dir,
            max_shards=max_shards,
            filesystem=filesystem,
            runner=runner,
        )
        manifest_files = tuple(sorted(local_root.rglob(manifest_pattern)))
        if manifest_files:
            if max_shards is not None:
                manifest_files = manifest_files[:max_shards]
            shards = tuple(
                cls._resolve_local_manifest(path.parent, path) for path in manifest_files
            )
        else:
            bin_files = tuple(sorted(local_root.rglob("*.bin")))
            if max_shards is not None:
                bin_files = bin_files[:max_shards]
            if not bin_files:
                raise FileNotFoundError(
                    f"No manifests or *.bin shards staged from {gcs_uri!r} in {local_root}"
                )
            resolved = []
            for data_file in bin_files:
                rel_path = data_file.relative_to(local_root).as_posix()
                shard_id = data_file.stem
                manifest, validation = _inspect_legacy_uint16_bin(
                    data_file,
                    shard_id=shard_id,
                    data_path=rel_path,
                    header_bytes=header_bytes,
                    vocab_size=vocab_size,
                )
                resolved.append(
                    _ResolvedPackedShard(
                        shard_id=shard_id,
                        manifest_file=data_file.with_suffix(".manifest.json"),
                        data_file=data_file,
                        document_index_file=None,
                        target_ids_file=None,
                        loss_mask_file=None,
                        manifest=manifest,
                        validation=validation,
                    )
                )
            shards = tuple(resolved)
        cls._require_unique_shards(shards)
        dataset = cls(shards, sequence_length=sequence_length, **partition)
        dataset._manifest_root = local_root if manifest_files else None
        return dataset

    @classmethod
    def from_hub(
        cls,
        source: HuggingFaceShardSourceConfig | str,
        *,
        sequence_length: int,
        revision: str | None = None,
        subfolder: str | None = None,
        max_shards: int | None = None,
        shard_range: tuple[int, int] | range | None = None,
        filename_template: str = "laughlm-v1/laughlm-v1_shard_{index:05d}.bin",
        header_bytes: int = 1024,
        vocab_size: int = 32011,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        **partition: Any,
    ) -> "PackedBinDataset":
        """Load manifest-backed shards, a subfolder of ``.bin`` shards, or a numbered range.

        Passing ``subfolder`` auto-discovers sorted ``*.bin`` files under that
        directory (for example ``stage1`` or ``stage2``) and derives each shard's
        ``token_count`` from ``file_size // 2``. ``max_shards`` limits downloads
        for single-shard ablations.
        """
        if isinstance(source, HuggingFaceShardSourceConfig):
            if revision is not None or shard_range is not None or subfolder is not None or max_shards is not None:
                raise ValueError(
                    "revision, subfolder, max_shards, and shard_range are only used with a repo ID."
                )
            shards = HuggingFacePackedShardSource(source).resolve()
            resolved_revision = source.revision
        elif isinstance(source, str):
            shards, resolved_revision = cls._resolve_hub_bins(
                source,
                revision=revision,
                subfolder=subfolder,
                max_shards=max_shards,
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
        dataset._source_revision = resolved_revision
        return dataset

    @classmethod
    def _resolve_hub_bins(
        cls,
        repo_id: str,
        *,
        revision: str | None,
        subfolder: str | None,
        max_shards: int | None,
        shard_range: tuple[int, int] | range | None,
        filename_template: str,
        header_bytes: int,
        vocab_size: int,
        cache_dir: str | Path | None,
        local_files_only: bool,
    ) -> tuple[tuple[_ResolvedPackedShard, ...], str]:
        if max_shards is not None and (
            isinstance(max_shards, bool) or not isinstance(max_shards, int) or max_shards < 1
        ):
            raise ValueError("max_shards must be a positive integer.")
        resolved_revision = _resolve_hub_revision(
            repo_id, revision or "main", local_files_only=local_files_only
        )
        if subfolder is not None:
            if shard_range is not None:
                raise ValueError("Specify either subfolder or shard_range, not both.")
            filenames = discover_hub_subfolder_bins(
                repo_id,
                subfolder=subfolder,
                revision=resolved_revision,
                max_shards=max_shards,
            )
        else:
            if shard_range is None:
                raise ValueError(
                    "Either subfolder or shard_range=(start, stop) is required for raw Hub bins."
                )
            indices = cls._shard_indices(shard_range)
            if max_shards is not None:
                indices = indices[:max_shards]
            resolved_names: list[str] = []
            for index in indices:
                try:
                    resolved_names.append(filename_template.format(index=index))
                except (IndexError, KeyError, ValueError) as exc:
                    raise ValueError(
                        "filename_template must contain a valid {index} field."
                    ) from exc
            filenames = tuple(resolved_names)
        shards = []
        for filename in filenames:
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
                target_ids_file=None,
                loss_mask_file=None,
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

    def _count_supervised_tokens_per_epoch(self) -> int:
        """Count valid next-token targets without changing the training cursor."""
        if self._supervised_tokens_per_epoch_cache is not None:
            return self._supervised_tokens_per_epoch_cache

        has_supervision_sidecars = any(
            shard.manifest.target_ids is not None
            or shard.manifest.loss_mask is not None
            for shard in self.shards
        )
        if not has_supervision_sidecars:
            count = len(self) * (self.sequence_length - 1)
            self._supervised_tokens_per_epoch_cache = count
            return count

        count = 0
        for batch_index in range(len(self._partition)):
            batch = self._partition.read_batch(batch_index)
            labels = batch["labels"][..., 1:]
            target_mask = (
                batch["attention_mask"][..., :-1]
                & batch["attention_mask"][..., 1:]
                & batch["loss_mask"][..., 1:]
                & labels.ne(-100)
            )
            count += int(target_mask.sum().item())

        self._supervised_tokens_per_epoch_cache = count
        return count

    def __iter__(self):
        worker = get_worker_info()
        if worker is not None:
            for index in range(worker.id, len(self._partition), worker.num_workers):
                yield self._public_example(self._partition.read_batch(index))
            return

        if self._cursor is None or self._cursor.position >= len(self._partition):
            self._cursor = PackedDataCursor(
                self._partition,
                source_revision=self._source_revision,
            )
        for batch in self._cursor:
            yield self._public_example(batch)

    @staticmethod
    def _public_example(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": batch["input_ids"].squeeze(0),
            "labels": batch["labels"].squeeze(0),
            "attention_mask": batch["attention_mask"].squeeze(0),
            "loss_mask": batch["loss_mask"].squeeze(0),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the exact next-batch cursor for zero-worker iteration."""
        cursor = self._cursor or PackedDataCursor(
            self._partition,
            source_revision=self._source_revision,
        )
        return {"schema_version": 1, "cursor": cursor.state.to_dict()}

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate a saved cursor without changing the active iterator."""
        self._validated_cursor(state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor after verifying its data and partition identity."""
        self._cursor = self._validated_cursor(state)

    def _validated_cursor(self, state: Mapping[str, Any]) -> PackedDataCursor:
        if (
            not isinstance(state, Mapping)
            or isinstance(state.get("schema_version"), bool)
            or not isinstance(state.get("schema_version"), int)
            or state["schema_version"] != 1
        ):
            raise ValueError("Unsupported packed dataset state schema.")
        cursor_state = PackedDataCursorState.from_dict(state.get("cursor", {}))
        return PackedDataCursor.from_state(
            self._partition,
            cursor_state,
            source_revision=self._source_revision,
        )

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
            for sidecar, source in (
                (shard.manifest.target_ids, shard.target_ids_file),
                (shard.manifest.loss_mask, shard.loss_mask_file),
            ):
                if sidecar is not None:
                    if source is None:
                        raise RuntimeError(
                            f"Validated shard {shard.shard_id} lost a sidecar path."
                        )
                    self._link_or_copy(source, root / sidecar.path)
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
        target_ids_file = (
            root / manifest.target_ids.path
            if manifest.target_ids is not None
            else None
        )
        loss_mask_file = (
            root / manifest.loss_mask.path
            if manifest.loss_mask is not None
            else None
        )
        validation = validate_packed_binary_shard(
            manifest,
            data_file,
            document_index_file=document_file,
            target_ids_file=target_ids_file,
            loss_mask_file=loss_mask_file,
        )
        return _ResolvedPackedShard(
            shard_id=manifest.shard_id,
            manifest_file=path,
            data_file=data_file,
            document_index_file=document_file,
            target_ids_file=target_ids_file,
            loss_mask_file=loss_mask_file,
            manifest=manifest,
            validation=validation,
        )

    @staticmethod
    def _require_unique_shards(shards: tuple[Any, ...]) -> None:
        ids = tuple(shard.shard_id for shard in shards)
        if len(ids) != len(set(ids)):
            raise ValueError("Packed dataset shard IDs must be unique.")
        paths = []
        for shard in shards:
            paths.append(shard.manifest.data_path)
            if shard.manifest.documents.path is not None:
                paths.append(shard.manifest.documents.path)
            if shard.manifest.target_ids is not None:
                paths.append(shard.manifest.target_ids.path)
            if shard.manifest.loss_mask is not None:
                paths.append(shard.manifest.loss_mask.path)
        if len(paths) != len(set(paths)):
            raise ValueError("Packed dataset shards must use unique payload paths.")

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


def discover_hub_subfolder_bins(
    repo_id: str,
    *,
    subfolder: str,
    revision: str = "main",
    max_shards: int | None = None,
    list_files_fn: Any = None,
) -> tuple[str, ...]:
    """Discover sorted ``*.bin`` shard paths under a Hub dataset subfolder."""
    clean_sub = subfolder.strip().strip("/")
    if not clean_sub:
        raise ValueError("subfolder cannot be empty.")
    if max_shards is not None and (
        isinstance(max_shards, bool) or not isinstance(max_shards, int) or max_shards < 1
    ):
        raise ValueError("max_shards must be a positive integer.")
    if list_files_fn is None:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    else:
        files = list_files_fn(repo_id=repo_id, revision=revision)
    prefix = f"{clean_sub}/"
    matched = sorted(
        path
        for path in files
        if isinstance(path, str)
        and path.startswith(prefix)
        and path.endswith(".bin")
    )
    if not matched:
        raise FileNotFoundError(
            f"No *.bin shards found under '{clean_sub}/' in dataset '{repo_id}@{revision}'."
        )
    if max_shards is not None:
        matched = matched[:max_shards]
    return tuple(matched)


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
            tokens = torch.frombuffer(bytearray(chunk), dtype=torch.uint16).to(
                dtype=torch.int32
            )
            chunk_minimum = int(tokens.min().item())
            chunk_maximum = int(tokens.max().item())
            minimum = (
                chunk_minimum if minimum is None else min(minimum, chunk_minimum)
            )
            maximum = (
                chunk_maximum if maximum is None else max(maximum, chunk_maximum)
            )
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
