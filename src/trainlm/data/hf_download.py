from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union, Literal

from huggingface_hub import hf_hub_download, snapshot_download


RepoType = Literal["dataset", "model", "space"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _normalize_files(
    data_files: Optional[Union[str, Sequence[str]]]
) -> Optional[List[str]]:
    """
    Normalize a file input into a list of strings.
    """
    if data_files is None:
        return None

    if isinstance(data_files, str):
        return [data_files]

    return list(data_files)


def _is_pattern(file_name: str) -> bool:
    """
    Detect glob-like patterns.
    """
    return any(ch in file_name for ch in ["*", "?", "["])


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def download_hf_files(
    repo_id: str,
    repo_type: RepoType = "dataset",
    data_files: Optional[Union[str, Sequence[str]]] = None,
    revision: str = "main",
    cache_dir: Optional[Union[str, Path]] = None,
    local_dir: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
) -> List[str]:
    """
    Download files from the Hugging Face Hub.

    Parameters
    ----------
    repo_id : str
        Hugging Face repo name, e.g. "org/dataset-name"

    repo_type : {"dataset", "model", "space"}
        Type of repository.

    data_files : str | Sequence[str] | None
        Exact file paths or glob patterns inside the repo.
        If None, downloads the full snapshot.

    revision : str
        Branch / tag / commit to download from.

    cache_dir : str | Path | None
        Cache location.

    local_dir : str | Path | None
        Optional directory to materialize files.

    token : str | None
        HF auth token for private repos.

    Returns
    -------
    List[str]
        Local paths to downloaded files.
    """
    files = _normalize_files(data_files)

    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    local_dir_str = str(local_dir) if local_dir is not None else None

    # --------------------------------------------------------
    # Case 1: full snapshot
    # --------------------------------------------------------
    if files is None:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir_str,
            local_dir=local_dir_str,
            token=token,
        )
        return [str(Path(snapshot_path))]

    # --------------------------------------------------------
    # Case 2: patterns -> snapshot_download with allow_patterns
    # --------------------------------------------------------
    if any(_is_pattern(f) for f in files):
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir_str,
            local_dir=local_dir_str,
            allow_patterns=files,
            token=token,
        )
        return [str(Path(snapshot_path))]

    # --------------------------------------------------------
    # Case 3: exact file names -> hf_hub_download per file
    # --------------------------------------------------------
    downloaded: List[str] = []

    for file_name in files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_name,
            revision=revision,
            cache_dir=cache_dir_str,
            local_dir=local_dir_str,
            token=token,
        )
        downloaded.append(str(Path(local_path)))

    return downloaded
