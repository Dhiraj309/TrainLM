"""Host-only shard preflight; never imports or initializes torch_xla."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from trainlm.data import (
    HuggingFacePackedShardSource, HuggingFaceShardSourceConfig,
    HuggingFaceShardSpec, PackedBinaryShardManifest, validate_packed_binary_shard,
)

def _local_shards(directory: Path):
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No shard manifests found in {directory}")
    shards = []
    for manifest_path in paths:
        manifest = PackedBinaryShardManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        data_path = manifest_path.parent / Path(manifest.data_path)
        document_path = None
        if manifest.documents.path is not None:
            document_path = manifest_path.parent / Path(manifest.documents.path)
        validation = validate_packed_binary_shard(
            manifest, data_path, document_index_file=document_path
        )
        shards.append(
            SimpleNamespace(
                shard_id=manifest.shard_id,
                data_file=data_path,
                manifest=manifest,
                validation=validation,
            )
        )
    return shards


def _raw_shards(args: argparse.Namespace):
    """Validate explicit raw-token layout once, without publishing sidecars."""
    import numpy as np
    from trainlm.data import ValidatedPackedBinaryShard

    dtype = np.dtype(("<" if args.byte_order == "little" else ">") + {
        "uint16": "u2", "uint32": "u4", "int32": "i4", "int64": "i8",
    }[args.token_dtype])
    paths = [Path(p).resolve(strict=True) for p in args.bin_path]
    if len(set(paths)) != len(paths) or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Raw shard paths and shard names must be unique.")
    shards = []
    for path in paths:
        before = path.stat()
        payload_bytes = before.st_size - args.header_bytes
        if payload_bytes <= 0 or payload_bytes % dtype.itemsize:
            raise ValueError(f"Invalid header/payload geometry: {path}")
        count = payload_bytes // dtype.itemsize
        if args.expected_token_count is not None and count != args.expected_token_count:
            raise ValueError(f"Token count differs from supplied metadata: {path}")
        digest = hashlib.sha256()
        minimum, maximum = None, None
        with path.open("rb") as handle:
            remaining_header = args.header_bytes
            while remaining_header:
                chunk = handle.read(min(8 * 1024 * 1024, remaining_header))
                if not chunk:
                    raise ValueError(f"Truncated header: {path}")
                digest.update(chunk)
                remaining_header -= len(chunk)
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                values = np.frombuffer(chunk, dtype=dtype)
                lo, hi = int(values.min()), int(values.max())
                if lo < 0 or hi >= args.token_vocab_size:
                    raise ValueError(f"Token IDs [{lo}, {hi}] exceed vocabulary bounds in {path}")
                minimum = lo if minimum is None else min(minimum, lo)
                maximum = hi if maximum is None else max(maximum, hi)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Shard changed while being validated: {path}")
        manifest = PackedBinaryShardManifest(
            schema_version=1, shard_id=path.stem, data_path=path.name,
            compatibility_profile="explicit_v1", header_bytes=args.header_bytes,
            token_dtype=args.token_dtype, byte_order=args.byte_order,
            token_count=count, token_id_min=minimum, token_id_max=maximum,
            vocab_size=args.token_vocab_size, file_size_bytes=before.st_size,
            sha256=digest.hexdigest(),
        )
        validation = ValidatedPackedBinaryShard(
            token_count=count, token_id_min=minimum, token_id_max=maximum,
            sha256=digest.hexdigest(), document_count=None,
        )
        shards.append(SimpleNamespace(
            shard_id=manifest.shard_id, data_file=path, manifest=manifest, validation=validation,
        ))
        print({"stage": "raw_shard_validated", "shard": path.name,
               "tokens": count, "header_bytes": args.header_bytes}, flush=True)
    return shards


def _shards(args: argparse.Namespace):
    if args.data_mode == "raw":
        return _raw_shards(args)
    if args.data_mode == "local":
        return _local_shards(Path(args.manifest_dir))
    if len(args.dataset_revision) != 40:
        raise ValueError("--dataset-revision must be a 40-character commit SHA")
    source = HuggingFacePackedShardSource(
        HuggingFaceShardSourceConfig(
            repo_id=args.dataset_repo,
            revision=args.dataset_revision,
            cache_dir=args.hf_cache_dir,
            shards=tuple(
                HuggingFaceShardSpec(
                    shard_id=f"laughlm-v1_shard_{index:05d}",
                    manifest_path=args.manifest_template.format(
                        root=args.dataset_root, index=index
                    ),
                )
                for index in range(
                    args.shard_start, args.shard_start + args.shard_count
                )
            ),
        )
    )
    return list(source.resolve())

