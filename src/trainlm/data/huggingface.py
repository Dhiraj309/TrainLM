"""Revision-pinned Hugging Face source for packed binary shards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .manifest import (
    PackedBinaryShardManifest,
    ValidatedPackedBinaryShard,
    validate_packed_binary_shard,
)

DownloadFile = Callable[..., str]

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty.")
    return value


def _require_relative_path(name: str, value: Any) -> str:
    _require_text(name, value)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or value in {".", ".."}
        or ".." in path.parts
    ):
        raise ValueError(f"{name} must be a safe relative POSIX path.")
    return value


@dataclass(frozen=True, slots=True)
class HuggingFaceShardSpec:
    """Expected shard identity and its manifest path in a dataset repo."""

    shard_id: str
    manifest_path: str

    def __post_init__(self) -> None:
        _require_text("shard_id", self.shard_id)
        _require_relative_path("manifest_path", self.manifest_path)


@dataclass(frozen=True, slots=True)
class HuggingFaceShardSourceConfig:
    """Immutable Hub location for an ordered set of packed shards."""

    repo_id: str
    revision: str
    shards: tuple[HuggingFaceShardSpec, ...]
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    repo_type: str = "dataset"

    def __post_init__(self) -> None:
        _require_text("repo_id", self.repo_id)
        if self.repo_id.count("/") != 1 or any(
            not part for part in self.repo_id.split("/")
        ):
            raise ValueError("repo_id must have the form 'owner/name'.")
        if not isinstance(self.revision, str) or _COMMIT_SHA.fullmatch(
            self.revision
        ) is None:
            raise ValueError("revision must be a lowercase 40-character commit SHA.")
        if self.repo_type != "dataset":
            raise ValueError("Packed shard sources require repo_type='dataset'.")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be boolean.")
        if isinstance(self.shards, (str, bytes)) or not isinstance(
            self.shards, (list, tuple)
        ):
            raise ValueError("shards must be a list or tuple.")
        object.__setattr__(self, "shards", tuple(self.shards))
        if not self.shards:
            raise ValueError("At least one packed shard must be requested.")
        if any(not isinstance(item, HuggingFaceShardSpec) for item in self.shards):
            raise ValueError("shards must contain HuggingFaceShardSpec values.")
        shard_ids = [item.shard_id for item in self.shards]
        manifest_paths = [item.manifest_path for item in self.shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("Hugging Face shard IDs must be unique.")
        if len(manifest_paths) != len(set(manifest_paths)):
            raise ValueError("Hugging Face manifest paths must be unique.")
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise ValueError("cache_dir must be a string or Path.")
            if not str(self.cache_dir).strip():
                raise ValueError("cache_dir cannot be empty.")


@dataclass(frozen=True, slots=True)
class ResolvedHuggingFaceShard:
    """A locally cached, fully validated shard with immutable provenance."""

    shard_id: str
    repo_id: str
    revision: str
    manifest_path: str
    data_path: str
    manifest_file: Path
    data_file: Path
    document_index_file: Path | None
    manifest: PackedBinaryShardManifest
    validation: ValidatedPackedBinaryShard


def _default_download_file(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


class HuggingFacePackedShardSource:
    """Resolve and validate manifests and payloads from one pinned snapshot."""

    def __init__(
        self,
        config: HuggingFaceShardSourceConfig,
        *,
        download_file: DownloadFile | None = None,
    ) -> None:
        if not isinstance(config, HuggingFaceShardSourceConfig):
            raise TypeError("config must be a HuggingFaceShardSourceConfig.")
        self.config = config
        self._download_file = download_file or _default_download_file

    def resolve(self) -> tuple[ResolvedHuggingFaceShard, ...]:
        resolved = []
        data_paths: set[str] = set()
        for spec in self.config.shards:
            manifest_file = self._download(spec.manifest_path)
            manifest = self._read_manifest(manifest_file)
            if manifest.shard_id != spec.shard_id:
                raise ValueError(
                    f"Manifest shard_id '{manifest.shard_id}' does not match "
                    f"requested shard_id '{spec.shard_id}'."
                )
            if manifest.data_path in data_paths:
                raise ValueError(
                    f"Multiple manifests resolve data path '{manifest.data_path}'."
                )
            data_paths.add(manifest.data_path)

            data_file = self._download(manifest.data_path)
            document_file = None
            if manifest.documents.storage != "unavailable":
                document_path = manifest.documents.path
                if document_path is None:
                    raise ValueError("Document-index path is missing from manifest.")
                document_file = self._download(document_path)
            validation = validate_packed_binary_shard(
                manifest,
                data_file,
                document_index_file=document_file,
            )
            resolved.append(
                ResolvedHuggingFaceShard(
                    shard_id=spec.shard_id,
                    repo_id=self.config.repo_id,
                    revision=self.config.revision,
                    manifest_path=spec.manifest_path,
                    data_path=manifest.data_path,
                    manifest_file=manifest_file,
                    data_file=data_file,
                    document_index_file=document_file,
                    manifest=manifest,
                    validation=validation,
                )
            )
        return tuple(resolved)

    def _download(self, filename: str) -> Path:
        kwargs: dict[str, Any] = {
            "repo_id": self.config.repo_id,
            "filename": filename,
            "repo_type": self.config.repo_type,
            "revision": self.config.revision,
            "local_files_only": self.config.local_files_only,
            "library_name": "trainlm",
        }
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = self.config.cache_dir
        # Deliberately omit ``token``. huggingface_hub then uses its standard
        # saved credential or HF_TOKEN without exposing secrets to TrainLM.
        path = Path(self._download_file(**kwargs))
        if not path.is_file():
            raise FileNotFoundError(
                f"Hugging Face download did not resolve a file: {filename}"
            )
        return path

    @staticmethod
    def _read_manifest(path: Path) -> PackedBinaryShardManifest:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Shard manifest is not UTF-8 JSON: {path}") from exc
        try:
            return PackedBinaryShardManifest.from_json(text)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid packed shard manifest: {path}") from exc
