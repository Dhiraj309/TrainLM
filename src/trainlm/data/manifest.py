"""Versioned manifests and eager validation for packed token shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Literal, Mapping

TokenDType = Literal["uint16", "uint32", "int32", "int64"]
ByteOrder = Literal["little", "big"]
CompatibilityProfile = Literal["explicit_v1", "legacy_1024_uint16"]
DocumentIndexStorage = Literal["unavailable", "uint64_offsets"]

_DTYPE_FORMATS: dict[str, tuple[str, int]] = {
    "uint16": ("H", 2),
    "uint32": ("I", 4),
    "int32": ("i", 4),
    "int64": ("q", 8),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 hexadecimal.")
    return value


def _require_relative_path(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty.")
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
class DocumentIndex:
    """Optional document boundaries stored as token offsets."""

    storage: DocumentIndexStorage = "unavailable"
    document_count: int | None = None
    path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.storage not in {"unavailable", "uint64_offsets"}:
            raise ValueError(f"Unsupported document-index storage: {self.storage}")
        values = (self.document_count, self.path, self.sha256, self.size_bytes)
        if self.storage == "unavailable":
            if any(value is not None for value in values):
                raise ValueError(
                    "Unavailable document metadata cannot declare index fields."
                )
            return

        _require_integer("document_count", self.document_count, minimum=1)
        _require_relative_path("document index path", self.path)
        _require_sha256("document index sha256", self.sha256)
        _require_integer("document index size_bytes", self.size_bytes)
        expected_size = (self.document_count + 1) * 8
        if self.size_bytes != expected_size:
            raise ValueError(
                "Document index size must contain document_count + 1 uint64 offsets."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DocumentIndex":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PackedBinaryShardManifest:
    """Complete interpretation and integrity contract for one token shard."""

    schema_version: int
    shard_id: str
    data_path: str
    compatibility_profile: CompatibilityProfile
    header_bytes: int
    token_dtype: TokenDType
    byte_order: ByteOrder
    token_count: int
    token_id_min: int
    token_id_max: int
    vocab_size: int
    file_size_bytes: int
    sha256: str
    documents: DocumentIndex = field(default_factory=DocumentIndex)
    manifest_type: Literal["packed_binary_shard"] = "packed_binary_shard"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("PackedBinaryShardManifest supports schema_version=1.")
        if self.manifest_type != "packed_binary_shard":
            raise ValueError("Unsupported packed-binary manifest type.")
        if not isinstance(self.shard_id, str) or not self.shard_id.strip():
            raise ValueError("shard_id cannot be empty.")
        _require_relative_path("data_path", self.data_path)
        if self.compatibility_profile not in {
            "explicit_v1",
            "legacy_1024_uint16",
        }:
            raise ValueError(
                f"Unsupported compatibility profile: {self.compatibility_profile}"
            )
        _require_integer("header_bytes", self.header_bytes)
        if self.token_dtype not in _DTYPE_FORMATS:
            raise ValueError(f"Unsupported token dtype: {self.token_dtype}")
        if self.byte_order not in {"little", "big"}:
            raise ValueError(f"Unsupported byte order: {self.byte_order}")
        _require_integer("token_count", self.token_count, minimum=1)
        _require_integer("token_id_min", self.token_id_min)
        _require_integer("token_id_max", self.token_id_max)
        _require_integer("vocab_size", self.vocab_size, minimum=1)
        if self.token_id_min > self.token_id_max:
            raise ValueError("token_id_min cannot exceed token_id_max.")
        if self.token_id_max >= self.vocab_size:
            raise ValueError("Token bounds must be within vocab_size.")
        _require_integer("file_size_bytes", self.file_size_bytes, minimum=1)
        _require_sha256("sha256", self.sha256)
        if not isinstance(self.documents, DocumentIndex):
            raise ValueError("documents must be a DocumentIndex.")

        item_size = _DTYPE_FORMATS[self.token_dtype][1]
        expected_size = self.header_bytes + self.token_count * item_size
        if self.file_size_bytes != expected_size:
            raise ValueError(
                "file_size_bytes does not match header, dtype, and token_count."
            )
        if self.compatibility_profile == "legacy_1024_uint16" and (
            self.header_bytes != 1024
            or self.token_dtype != "uint16"
            or self.byte_order != "little"
        ):
            raise ValueError(
                "legacy_1024_uint16 requires a 1024-byte header and "
                "little-endian uint16 tokens."
            )

    @property
    def item_size(self) -> int:
        return _DTYPE_FORMATS[self.token_dtype][1]

    @property
    def struct_prefix(self) -> str:
        return "<" if self.byte_order == "little" else ">"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PackedBinaryShardManifest":
        values = dict(data)
        values["documents"] = DocumentIndex.from_dict(
            values.get("documents", {})
        )
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> "PackedBinaryShardManifest":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Packed-binary manifest JSON root must be an object.")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class ValidatedPackedBinaryShard:
    """Observed values after full integrity validation."""

    token_count: int
    token_id_min: int
    token_id_max: int
    sha256: str
    document_count: int | None


def _file_sha256(path: Path, *, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_and_hash_tokens(
    manifest: PackedBinaryShardManifest,
    path: Path,
    *,
    chunk_bytes: int,
    chunk_tokens: int,
) -> tuple[int, int | None, int | None, str, int | None]:
    format_code, item_size = _DTYPE_FORMATS[manifest.token_dtype]
    unpack = struct.Struct(f"{manifest.struct_prefix}{format_code}")
    digest = hashlib.sha256()
    observed_min: int | None = None
    observed_max: int | None = None
    observed_count = 0
    invalid_token: int | None = None
    with path.open("rb") as handle:
        remaining_header = manifest.header_bytes
        while remaining_header:
            chunk = handle.read(min(chunk_bytes, remaining_header))
            if not chunk:
                raise ValueError("Packed token shard ends inside its header.")
            digest.update(chunk)
            remaining_header -= len(chunk)
        while chunk := handle.read(chunk_tokens * item_size):
            digest.update(chunk)
            if len(chunk) % item_size:
                raise ValueError("Token payload ends with an incomplete value.")
            for (token,) in struct.iter_unpack(unpack.format, chunk):
                if invalid_token is None and (
                    token < 0 or token >= manifest.vocab_size
                ):
                    invalid_token = token
                observed_min = token if observed_min is None else min(
                    observed_min, token
                )
                observed_max = token if observed_max is None else max(
                    observed_max, token
                )
                observed_count += 1
    return (
        observed_count,
        observed_min,
        observed_max,
        digest.hexdigest(),
        invalid_token,
    )


def _validate_document_index(
    manifest: PackedBinaryShardManifest,
    path: Path | None,
    *,
    chunk_bytes: int,
) -> int | None:
    documents = manifest.documents
    if documents.storage == "unavailable":
        if path is not None:
            raise ValueError("Manifest declares no document index.")
        return None
    if path is None:
        raise ValueError("Document index file is required by the manifest.")
    if not path.is_file():
        raise FileNotFoundError(f"Document index does not exist: {path}")
    if path.stat().st_size != documents.size_bytes:
        raise ValueError("Document index size does not match the manifest.")
    if _file_sha256(path, chunk_bytes=chunk_bytes) != documents.sha256:
        raise ValueError("Document index checksum does not match the manifest.")

    unpack = struct.Struct(f"{manifest.struct_prefix}Q")
    read_size = max(1, chunk_bytes // unpack.size) * unpack.size
    first_offset: int | None = None
    previous_offset: int | None = None
    offset_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(read_size):
            if len(chunk) % unpack.size:
                raise ValueError("Document index ends with an incomplete offset.")
            for (offset,) in struct.iter_unpack(unpack.format, chunk):
                if first_offset is None:
                    first_offset = offset
                if previous_offset is not None and previous_offset >= offset:
                    raise ValueError("Document offsets must be strictly increasing.")
                previous_offset = offset
                offset_count += 1
    if offset_count != documents.document_count + 1:
        raise ValueError("Document offset count does not match the manifest.")
    if first_offset != 0 or previous_offset != manifest.token_count:
        raise ValueError("Document offsets must span the complete token payload.")
    return documents.document_count


def validate_packed_binary_shard(
    manifest: PackedBinaryShardManifest,
    data_file: str | Path,
    *,
    document_index_file: str | Path | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
    chunk_tokens: int = 1_048_576,
) -> ValidatedPackedBinaryShard:
    """Fully validate a shard before a reader or memmap is constructed."""

    if not isinstance(manifest, PackedBinaryShardManifest):
        raise TypeError("manifest must be a PackedBinaryShardManifest.")
    _require_integer("chunk_bytes", chunk_bytes, minimum=1)
    _require_integer("chunk_tokens", chunk_tokens, minimum=1)
    path = Path(data_file)
    if not path.is_file():
        raise FileNotFoundError(f"Packed token shard does not exist: {path}")
    if path.stat().st_size != manifest.file_size_bytes:
        raise ValueError("Packed token shard size does not match the manifest.")
    (
        token_count,
        token_min,
        token_max,
        observed_sha256,
        invalid_token,
    ) = _scan_and_hash_tokens(
        manifest,
        path,
        chunk_bytes=chunk_bytes,
        chunk_tokens=chunk_tokens,
    )
    if observed_sha256 != manifest.sha256:
        raise ValueError("Packed token shard checksum does not match the manifest.")
    if invalid_token is not None:
        raise ValueError(
            f"Token id {invalid_token} is outside [0, {manifest.vocab_size})."
        )
    if token_count != manifest.token_count:
        raise ValueError("Observed token count does not match the manifest.")
    if token_min != manifest.token_id_min or token_max != manifest.token_id_max:
        raise ValueError("Observed token bounds do not match the manifest.")
    document_count = _validate_document_index(
        manifest,
        None if document_index_file is None else Path(document_index_file),
        chunk_bytes=chunk_bytes,
    )
    return ValidatedPackedBinaryShard(
        token_count=token_count,
        token_id_min=token_min,
        token_id_max=token_max,
        sha256=observed_sha256,
        document_count=document_count,
    )
