"""Streaming writer for explicit, validated uint32 packed-token shards."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

from .manifest import (
    DocumentIndex,
    PackedBinaryShardManifest,
)

_SHARD_ID = re.compile(r"[A-Za-z0-9._-]{1,96}\Z")
_UINT32_LIMIT = 1 << 32


def _write_little_endian(
    stream,
    values: list[int],
    *,
    typecode: str,
    digest: Any,
) -> None:
    if not values:
        return
    block = array(typecode, values)
    if sys.byteorder != "little":
        block.byteswap()
    payload = block.tobytes()
    stream.write(payload)
    digest.update(payload)


def write_packed_binary_shard(
    token_documents: Iterable[Iterable[int]],
    output_dir: str | Path,
    *,
    shard_id: str,
    vocab_size: int,
    flush_tokens: int = 65_536,
) -> Path:
    """Write documents as one little-endian uint32 shard plus a manifest.

    The source can be a generator, so token IDs are validated and written in
    bounded chunks. An explicit uint64 document-offset file preserves document
    boundaries. The manifest is installed last, after both payload files.

    Returns the path to the created manifest.
    """
    if not isinstance(shard_id, str) or not _SHARD_ID.fullmatch(shard_id):
        raise ValueError("shard_id must contain only letters, digits, '.', '_' or '-'.")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or not 1 <= vocab_size <= _UINT32_LIMIT
    ):
        raise ValueError("vocab_size must be an integer in [1, 2**32].")
    if (
        isinstance(flush_tokens, bool)
        or not isinstance(flush_tokens, int)
        or flush_tokens < 1
    ):
        raise ValueError("flush_tokens must be a positive integer.")
    if array("I").itemsize != 4 or array("Q").itemsize != 8:
        raise RuntimeError("This platform does not provide 32-bit/64-bit array types.")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{shard_id}.bin"
    offsets_path = root / f"{shard_id}.documents.u64"
    manifest_path = root / f"{shard_id}.manifest.json"
    temporary_paths: list[Path] = []

    def temporary(suffix: str) -> tuple[int, Path]:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{shard_id}.", suffix=suffix, dir=root
        )
        path = Path(name)
        temporary_paths.append(path)
        return descriptor, path

    try:
        data_fd, temporary_data = temporary(".bin.tmp")
        offsets_fd, temporary_offsets = temporary(".u64.tmp")
        token_digest = hashlib.sha256()
        offsets_digest = hashlib.sha256()
        token_buffer: list[int] = []
        offset_buffer = [0]
        token_count = 0
        document_count = 0
        token_min: int | None = None
        token_max: int | None = None

        with os.fdopen(data_fd, "wb") as data_stream, os.fdopen(
            offsets_fd, "wb"
        ) as offsets_stream:
            for document in token_documents:
                if isinstance(document, (str, bytes, bytearray)):
                    raise TypeError("Each document must be an iterable of token IDs.")
                document_tokens = 0
                for token in document:
                    if isinstance(token, bool) or not isinstance(token, int):
                        raise TypeError("Token IDs must be integers, not booleans.")
                    if not 0 <= token < vocab_size:
                        raise ValueError(
                            f"Token id {token} is outside [0, {vocab_size})."
                        )
                    token_buffer.append(token)
                    token_count += 1
                    document_tokens += 1
                    token_min = token if token_min is None else min(token_min, token)
                    token_max = token if token_max is None else max(token_max, token)
                    if len(token_buffer) >= flush_tokens:
                        _write_little_endian(
                            data_stream, token_buffer, typecode="I",
                            digest=token_digest,
                        )
                        token_buffer.clear()
                if document_tokens:
                    document_count += 1
                    offset_buffer.append(token_count)
                    if len(offset_buffer) >= flush_tokens:
                        _write_little_endian(
                            offsets_stream, offset_buffer, typecode="Q",
                            digest=offsets_digest,
                        )
                        offset_buffer.clear()

            _write_little_endian(
                data_stream, token_buffer, typecode="I", digest=token_digest
            )
            _write_little_endian(
                offsets_stream, offset_buffer, typecode="Q",
                digest=offsets_digest,
            )
            data_stream.flush()
            offsets_stream.flush()

        if (
            token_count == 0
            or document_count == 0
            or token_min is None
            or token_max is None
        ):
            raise ValueError("At least one non-empty token document is required.")

        manifest = PackedBinaryShardManifest(
            schema_version=1,
            shard_id=shard_id,
            data_path=data_path.name,
            compatibility_profile="explicit_v1",
            header_bytes=0,
            token_dtype="uint32",
            byte_order="little",
            token_count=token_count,
            token_id_min=token_min,
            token_id_max=token_max,
            vocab_size=vocab_size,
            file_size_bytes=token_count * 4,
            sha256=token_digest.hexdigest(),
            documents=DocumentIndex(
                storage="uint64_offsets",
                document_count=document_count,
                path=offsets_path.name,
                sha256=offsets_digest.hexdigest(),
                size_bytes=(document_count + 1) * 8,
            ),
        )
        manifest_fd, temporary_manifest = temporary(".manifest.tmp")
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as stream:
            stream.write(manifest.to_json() + "\n")

        os.replace(temporary_data, data_path)
        temporary_paths.remove(temporary_data)
        os.replace(temporary_offsets, offsets_path)
        temporary_paths.remove(temporary_offsets)
        os.replace(temporary_manifest, manifest_path)
        temporary_paths.remove(temporary_manifest)
        return manifest_path
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


__all__ = ["write_packed_binary_shard"]
