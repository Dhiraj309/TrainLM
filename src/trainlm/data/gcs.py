"""Same-region ``gs://`` GCS shard discovery and local SSD staging."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable


def is_gcs_uri(path: str | Path) -> bool:
    """Return True when ``path`` is a Google Cloud Storage ``gs://`` URI."""
    return isinstance(path, str) and path.startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/prefix`` into ``(bucket, prefix)``."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}.")
    rest = uri[len("gs://") :].strip("/")
    if not rest:
        raise ValueError(f"GCS URI must include a bucket name: {uri!r}.")
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"GCS URI bucket cannot be empty: {uri!r}.")
    return bucket, prefix


def _default_cache_dir(bucket: str, prefix: str, cache_dir: str | Path | None) -> Path:
    base = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "trainlm" / "gcs"
    )
    digest = hashlib.sha1(f"{bucket}/{prefix}".encode("utf-8")).hexdigest()[:12]
    return (base / f"{bucket}_{digest}").resolve()


def _resolve_gcs_filesystem(filesystem: Any = None) -> Any | None:
    if filesystem is not None:
        return filesystem
    try:
        import fsspec  # type: ignore[import-not-found]

        return fsspec.filesystem("gcs")
    except Exception:
        return None


def stage_gcs_prefix(
    gcs_uri: str,
    *,
    cache_dir: str | Path | None = None,
    max_shards: int | None = None,
    filesystem: Any = None,
    runner: Callable[..., Any] | None = None,
) -> Path:
    """Mirror packed manifests and ``.bin``/sidecar shards from ``gs://`` to local disk.

    Prefers ``fsspec``/``gcsfs`` when available (enabling selective ``max_shards``
    staging); otherwise falls back to ``gcloud storage cp -r`` or ``gsutil -m cp -r``.
    Already-staged files are reused on subsequent calls.
    """
    if max_shards is not None and (
        isinstance(max_shards, bool) or not isinstance(max_shards, int) or max_shards < 1
    ):
        raise ValueError("max_shards must be a positive integer.")

    bucket, prefix = parse_gcs_uri(gcs_uri)
    local_root = _default_cache_dir(bucket, prefix, cache_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    remote_prefix = f"{bucket}/{prefix}".rstrip("/")

    fs = _resolve_gcs_filesystem(filesystem)
    if fs is not None:
        entries = sorted(str(p) for p in fs.find(remote_prefix))
        manifests = [p for p in entries if p.endswith(".manifest.json")]
        bins = [p for p in entries if p.endswith(".bin")]
        if manifests:
            selected_manifests = manifests[:max_shards] if max_shards is not None else manifests
            stems = {
                Path(m).name[: -len(".manifest.json")] for m in selected_manifests
            }
            to_fetch = list(selected_manifests)
            for p in entries:
                if p in to_fetch:
                    continue
                if any(Path(p).name.startswith(stem) for stem in stems):
                    to_fetch.append(p)
        elif bins:
            to_fetch = bins[:max_shards] if max_shards is not None else bins
        else:
            raise FileNotFoundError(
                f"No packed manifests or *.bin shards found under {gcs_uri!r}."
            )

        for remote_path in to_fetch:
            rel = remote_path[len(remote_prefix) :].lstrip("/")
            if not rel:
                rel = Path(remote_path).name
            dest = local_root / rel
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_dest = dest.with_name(f".{dest.name}.part")
            fs.get(remote_path, str(tmp_dest))
            tmp_dest.replace(dest)
        return local_root

    exec_fn = runner or subprocess.run
    gcloud_bin = shutil.which("gcloud")
    gsutil_bin = shutil.which("gsutil")
    normalized_uri = f"gs://{remote_prefix}/*"
    if gcloud_bin is not None:
        exec_fn([gcloud_bin, "storage", "cp", "-r", normalized_uri, str(local_root)], check=True)
    elif gsutil_bin is not None:
        exec_fn([gsutil_bin, "-m", "cp", "-r", normalized_uri, str(local_root)], check=True)
    else:
        raise RuntimeError(
            "Staging gs:// shards requires either 'fsspec'+'gcsfs' or 'gcloud'/'gsutil' CLI."
        )
    return local_root


__all__ = [
    "is_gcs_uri",
    "parse_gcs_uri",
    "stage_gcs_prefix",
]
