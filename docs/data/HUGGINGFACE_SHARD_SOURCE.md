# Revision-pinned Hugging Face shard source

`HuggingFacePackedShardSource` resolves packed token shards from a Hugging Face
dataset repository without making the reader aware of remote storage. It
downloads each declared manifest and its referenced files into the standard Hub
cache, validates them, and returns immutable local paths to M3-F3.

## Deterministic source identity

Every source requires:

- a dataset repository ID in `owner/name` form;
- an exact lowercase 40-character commit SHA;
- an ordered tuple of unique shard IDs and manifest paths; and
- an optional Hub cache directory and cache-only policy.

Branches such as `main` and mutable tags are rejected. Using one immutable
revision for every manifest, payload, and document index prevents a repository
update from mixing files from different dataset versions during one resolve.
The declared tuple order is preserved; the source never relies on Hub listing
order or filename globbing.

```python
from trainlm.data import (
    HuggingFacePackedShardSource,
    HuggingFaceShardSourceConfig,
    HuggingFaceShardSpec,
)

source = HuggingFacePackedShardSource(HuggingFaceShardSourceConfig(
    repo_id="LaughTaleAI/LaughLM-Tokenized-Fine",
    revision="0123456789abcdef0123456789abcdef01234567",
    cache_dir="/tmp/laughlm_hf_cache",
    shards=tuple(
        HuggingFaceShardSpec(
            shard_id=f"fineweb-edu_shard_{index:05d}",
            manifest_path=(
                f"fineweb-edu/fineweb-edu_shard_{index:05d}.manifest.json"
            ),
        )
        for index in range(28)
    ),
))

train_shards = source.resolve()
```

Each manifest's `shard_id` must match the requested ID. Duplicate manifest or
payload paths are rejected. Resolution returns only after size, checksum, token
range, and optional document-index validation succeeds.

## Authentication and secrets

TrainLM deliberately has no token configuration field. It omits the `token`
argument when calling `hf_hub_download`, allowing `huggingface_hub` to use its
standard saved credential or the `HF_TOKEN` environment variable. Tokens are
therefore absent from TrainLM configs, logs, reports, and checkpoints.

## Offline and cache reuse

Set `local_files_only=True` to prohibit downloads and resolve only files already
present in the selected Hub cache. The standard `HF_HUB_OFFLINE=1` environment
variable is also honored by `huggingface_hub`. A cache miss fails instead of
silently changing revisions or locations.

TrainLM uses `hf_hub_download` without `local_dir`, preserving the Hub's
content-addressed snapshot cache and deduplication behavior. `cache_dir` maps
directly to the official Hub API; when omitted, normal `HF_HOME` and
`HF_HUB_CACHE` resolution applies.

Official references:

- [Hugging Face file download API](https://huggingface.co/docs/huggingface_hub/main/package_reference/file_download)
- [Hugging Face environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

This story performs deterministic acquisition and integrity validation only.
It does not create memmaps, batches, workers, partitions, or prefetch queues.
