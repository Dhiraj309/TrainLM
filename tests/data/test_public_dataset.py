"""Public packed-binary dataset construction and partitioning."""

from __future__ import annotations

import hashlib
import struct

import pytest
import torch

from trainlm import PackedBinDataset
from trainlm.data import HuggingFaceShardSourceConfig, HuggingFaceShardSpec
from trainlm.data.huggingface import HuggingFacePackedShardSource

from .test_contiguous_reader import _local_shard


def _write_manifest(root, shard):
    data_path = root / shard.manifest.data_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shard.data_file.replace(data_path)
    path = root / f"{shard.shard_id}.manifest.json"
    path.write_text(shard.manifest.to_json() + "\n", encoding="utf-8")


def test_directory_dataset_validates_and_yields_fixed_examples(tmp_path):
    shard = _local_shard(tmp_path, shard_id="train", tokens=tuple(range(16)))
    _write_manifest(tmp_path, shard)

    dataset = PackedBinDataset.from_directory(tmp_path, sequence_length=4)

    examples = list(dataset)
    assert len(dataset) == 4
    assert len(examples) == 4
    assert torch.equal(examples[0]["input_ids"], torch.tensor([0, 1, 2, 3]))
    assert examples[0]["labels"].shape == (4,)
    assert examples[0]["attention_mask"].all()
    dataset.close()


def test_directory_dataset_partitions_without_overlap(tmp_path):
    shard = _local_shard(tmp_path, shard_id="train", tokens=tuple(range(16)))
    _write_manifest(tmp_path, shard)
    ranks = [
        PackedBinDataset.from_directory(
            tmp_path,
            sequence_length=4,
            seed=7,
            world_size=2,
            rank=rank,
        )
        for rank in range(2)
    ]

    owned = [{tuple(item["input_ids"].tolist()) for item in dataset} for dataset in ranks]
    assert owned[0].isdisjoint(owned[1])
    assert owned[0] | owned[1] == {
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    }
    for dataset in ranks:
        dataset.close()


def test_hub_dataset_uses_revision_pinned_validated_source(tmp_path, monkeypatch):
    shard = _local_shard(tmp_path, shard_id="train", tokens=tuple(range(8)))
    source = HuggingFaceShardSourceConfig(
        repo_id="org/data",
        revision="a" * 40,
        shards=(HuggingFaceShardSpec("train", "train.manifest.json"),),
    )
    monkeypatch.setattr(HuggingFacePackedShardSource, "resolve", lambda self: (shard,))

    dataset = PackedBinDataset.from_hub(source, sequence_length=4)

    assert len(list(dataset)) == 2
    staged = dataset.coordinator_manifest_dir(tmp_path / "run")
    assert tuple(staged.glob("*.manifest.json"))
    assert (staged / shard.manifest.data_path).is_file()
    dataset.close()


def test_hub_bin_range_downloads_and_validates_end_to_end(tmp_path, monkeypatch):
    revision = "b" * 40
    files = {}
    for index, tokens in enumerate(((1, 2, 3, 4), (5, 6, 7, 8))):
        path = tmp_path / f"laughlm-v1_shard_{index:05d}.bin"
        path.write_bytes(bytes(1024) + struct.pack("<4H", *tokens))
        files[f"laughlm-v1/laughlm-v1_shard_{index:05d}.bin"] = path
    downloads = []

    monkeypatch.setattr(
        "trainlm.data.public._resolve_hub_revision",
        lambda repo_id, requested, local_files_only: revision,
    )

    def download(**kwargs):
        downloads.append(kwargs)
        return files[kwargs["filename"]]

    monkeypatch.setattr("trainlm.data.public._download_hub_file", download)

    dataset = PackedBinDataset.from_hub(
        "LaughTaleAI/LaughLM-Tokenized-Fine",
        revision="main",
        shard_range=(0, 2),
        sequence_length=4,
        vocab_size=32,
    )

    assert dataset.hub_revision == revision
    assert [call["filename"] for call in downloads] == [
        "laughlm-v1/laughlm-v1_shard_00000.bin",
        "laughlm-v1/laughlm-v1_shard_00001.bin",
    ]
    assert [tuple(example["input_ids"].tolist()) for example in dataset] == [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
    ]
    staged = dataset.coordinator_manifest_dir(tmp_path / "run")
    manifests = sorted(staged.glob("*.manifest.json"))
    assert len(manifests) == 2
    assert hashlib.sha256(files[downloads[0]["filename"]].read_bytes()).hexdigest() in (
        manifests[0].read_text(encoding="utf-8")
    )
    dataset.close()


@pytest.mark.parametrize("shard_range", [(2, 2), (-1, 2), range(0, 3, 2)])
def test_hub_bin_range_rejects_invalid_ranges(shard_range):
    with pytest.raises((TypeError, ValueError), match="shard_range"):
        PackedBinDataset._shard_indices(shard_range)


def test_directory_dataset_rejects_missing_or_invalid_manifests(tmp_path):
    with pytest.raises(FileNotFoundError, match="No packed shard manifests"):
        PackedBinDataset.from_directory(tmp_path, sequence_length=4)
    (tmp_path / "bad.manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid packed shard manifest"):
        PackedBinDataset.from_directory(tmp_path, sequence_length=4)
