"""Same-region ``gs://`` GCS shard discovery, checksum verification, and race-free local SSD staging."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

logger = logging.getLogger(__name__)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_optional_sha256_sidecars(local_root: Path) -> int:
    """Verify any ``*.sha256`` sidecars present alongside staged ``.bin`` files."""
    verified = 0
    for sidecar in sorted(local_root.rglob("*.sha256")):
        raw = sidecar.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        expected = raw.split()[0].lower()
        name = sidecar.name[: -len(".sha256")]
        candidate = sidecar.with_name(name)
        if not candidate.is_file() and not name.endswith(".bin"):
            candidate = sidecar.with_name(f"{name}.bin")
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Checksum sidecar {sidecar.name} has no matching payload file in {sidecar.parent}."
            )
        actual = _sha256_file(candidate)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for staged GCS file {candidate.name}: "
                f"expected {expected}, got {actual}."
            )
        verified += 1
    return verified


def _select_ordered_entries(
    entries: list[str], *, gcs_uri: str, max_shards: int | None
) -> list[str]:
    manifests = sorted(p for p in entries if p.endswith(".manifest.json"))
    bins = sorted(p for p in entries if p.endswith(".bin"))
    sha_files = {p for p in entries if p.endswith(".sha256")}
    if manifests:
        selected = manifests[:max_shards] if max_shards is not None else manifests
        stems = {Path(m).name[: -len(".manifest.json")] for m in selected}
        chosen = list(selected)
        for p in sorted(entries):
            if p not in chosen and any(Path(p).name.startswith(stem) for stem in stems):
                chosen.append(p)
        return chosen
    if bins:
        selected_bins = bins[:max_shards] if max_shards is not None else bins
        chosen = list(selected_bins)
        for b in selected_bins:
            for cand in (f"{b}.sha256", str(Path(b).with_suffix(".sha256"))):
                if cand in sha_files and cand not in chosen:
                    chosen.append(cand)
        return chosen
    raise FileNotFoundError(
        f"No packed manifests or *.bin shards found under {gcs_uri!r}."
    )


def _emit_same_region_notice(*, bucket: str, gcs_uri: str, local_root: Path) -> None:
    msg = (
        f"[TrainLM GCS] Staging {gcs_uri} -> {local_root}. "
        f"Verify '{bucket}' is a Single-Region bucket in the SAME region as your "
        f"TPU slice to guarantee $0.00 network egress."
    )
    logger.info(msg)
    print(msg, flush=True)


def _distributed_staging_context() -> tuple[bool, bool, Any | None]:
    """Return ``(is_distributed, is_local_leader, torch_distributed_module)``."""
    try:
        import torch  # noqa: PLC0415

        dist_mod = getattr(torch, "distributed", None)
        if (
            dist_mod is not None
            and callable(getattr(dist_mod, "is_available", None))
            and dist_mod.is_available()
            and callable(getattr(dist_mod, "is_initialized", None))
            and dist_mod.is_initialized()
            and int(dist_mod.get_world_size()) > 1
        ):
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None and local_rank_env.strip():
                is_leader = int(local_rank_env) == 0
            else:
                is_leader = int(dist_mod.get_rank()) == 0
            return True, is_leader, dist_mod
    except Exception:
        pass
    return False, True, None


def _write_staged_marker(local_root: Path, *, scope_tag: str) -> Path:
    marker = local_root / ".staged.ok"
    tmp = local_root / f".staged.ok.tmp.{os.getpid()}"
    tmp.write_text(f"{scope_tag}\n", encoding="utf-8")
    tmp.replace(marker)
    return marker


def _stage_gcs_files_locally(
    *,
    gcs_uri: str,
    remote_prefix: str,
    local_root: Path,
    max_shards: int | None,
    filesystem: Any,
    runner: Callable[..., Any] | None,
    which: Callable[[str], str | None] | None,
) -> None:
    fs = _resolve_gcs_filesystem(filesystem)
    if fs is not None:
        entries = sorted(str(p) for p in fs.find(remote_prefix))
        to_fetch = _select_ordered_entries(entries, gcs_uri=gcs_uri, max_shards=max_shards)
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
        verify_optional_sha256_sidecars(local_root)
        return

    which_fn = which or shutil.which
    exec_fn = runner or subprocess.run
    gcloud_bin = which_fn("gcloud")
    gsutil_bin = which_fn("gsutil")
    normalized_uri = f"gs://{remote_prefix}/*"
    if gcloud_bin is not None:
        if max_shards is not None:
            ls_res = exec_fn(
                [gcloud_bin, "storage", "ls", f"gs://{remote_prefix}/**"],
                check=False,
                capture_output=True,
                text=True,
            )
            listed = [
                line.strip()
                for line in (getattr(ls_res, "stdout", "") or "").splitlines()
                if line.strip().startswith("gs://") and not line.strip().endswith("/")
            ]
            if listed:
                to_fetch = _select_ordered_entries(
                    listed, gcs_uri=gcs_uri, max_shards=max_shards
                )
                for uri in to_fetch:
                    rel = uri[len(f"gs://{remote_prefix}") :].lstrip("/") or Path(uri).name
                    dest = local_root / rel
                    if dest.exists() and dest.stat().st_size > 0:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    exec_fn([gcloud_bin, "storage", "cp", uri, str(dest)], check=True)
                verify_optional_sha256_sidecars(local_root)
                return
        exec_fn([gcloud_bin, "storage", "cp", "-r", normalized_uri, str(local_root)], check=True)
    elif gsutil_bin is not None:
        exec_fn([gsutil_bin, "-m", "cp", "-r", normalized_uri, str(local_root)], check=True)
    else:
        raise RuntimeError(
            "Staging gs:// shards requires either 'fsspec'+'gcsfs' or 'gcloud'/'gsutil' CLI."
        )
    verify_optional_sha256_sidecars(local_root)


def stage_gcs_prefix(
    gcs_uri: str,
    *,
    cache_dir: str | Path | None = None,
    max_shards: int | None = None,
    filesystem: Any = None,
    runner: Callable[..., Any] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> Path:
    """Mirror packed manifests and ``.bin``/sidecar shards from ``gs://`` to local disk.

    In distributed runs (``torch.distributed.is_initialized()``), only the node-local
    leader (``local_rank == 0`` or ``rank == 0``) downloads/verifies files and writes
    an atomic ``.staged.ok`` marker before synchronizing via ``barrier()``.
    """
    if max_shards is not None and (
        isinstance(max_shards, bool) or not isinstance(max_shards, int) or max_shards < 1
    ):
        raise ValueError("max_shards must be a positive integer.")

    bucket, prefix = parse_gcs_uri(gcs_uri)
    local_root = _default_cache_dir(bucket, prefix, cache_dir)
    remote_prefix = f"{bucket}/{prefix}".rstrip("/")
    is_distributed, is_leader, dist_mod = _distributed_staging_context()
    scope_tag = f"max_shards={max_shards or 'all'}"

    if is_leader:
        local_root.mkdir(parents=True, exist_ok=True)
        _emit_same_region_notice(bucket=bucket, gcs_uri=gcs_uri, local_root=local_root)
        _stage_gcs_files_locally(
            gcs_uri=gcs_uri,
            remote_prefix=remote_prefix,
            local_root=local_root,
            max_shards=max_shards,
            filesystem=filesystem,
            runner=runner,
            which=which,
        )
        _write_staged_marker(local_root, scope_tag=scope_tag)

    if is_distributed and dist_mod is not None:
        dist_mod.barrier()
        marker = local_root / ".staged.ok"
        if not marker.is_file():
            raise RuntimeError(
                f"GCS staging barrier completed without expected marker: {marker}"
            )
    return local_root


__all__ = [
    "is_gcs_uri",
    "parse_gcs_uri",
    "stage_gcs_prefix",
    "verify_optional_sha256_sidecars",
]
