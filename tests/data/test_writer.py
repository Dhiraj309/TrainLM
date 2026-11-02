import json

import pytest
import torch

from trainlm import PackedBinDataset
from trainlm.data.manifest import (
    PackedBinaryShardManifest,
    validate_packed_binary_shard,
)
from trainlm.data.writer import write_packed_binary_shard


def test_writer_creates_valid_uint32_shard_and_document_index(tmp_path):
    manifest_path = write_packed_binary_shard(
        iter(([1, 2], [3, 4], [5, 6])),
        tmp_path,
        shard_id="train-part-00000",
        vocab_size=8,
        flush_tokens=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = tmp_path / manifest["data_path"]
    offsets_path = tmp_path / manifest["documents"]["path"]

    assert manifest["token_dtype"] == "uint32"
    assert manifest["byte_order"] == "little"
    assert manifest["token_count"] == 6
    assert manifest["documents"]["document_count"] == 3
    assert validate_packed_binary_shard(
        PackedBinaryShardManifest.from_dict(manifest),
        data_path,
        document_index_file=offsets_path,
    ).document_count == 3

    dataset = PackedBinDataset.from_directory(
        tmp_path, sequence_length=2, split="validation",
    )
    try:
        examples = list(dataset)
        assert len(examples) == 3
        assert [row["input_ids"].tolist() for row in examples] == [
            [1, 2], [3, 4], [5, 6],
        ]
        assert all(
            torch.equal(row["labels"], row["input_ids"]) for row in examples
        )
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        ([[1, 8]], "outside"),
        ([[True]], "integers"),
        ([[]], "non-empty"),
        ([], "non-empty"),
    ],
)
def test_writer_rejects_invalid_documents_and_token_ids(
    tmp_path, documents, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        write_packed_binary_shard(
            documents,
            tmp_path,
            shard_id="invalid",
            vocab_size=8,
        )


def test_writer_rejects_unsafe_shard_names(tmp_path):
    with pytest.raises(ValueError, match="shard_id"):
        write_packed_binary_shard(
            [[1]], tmp_path, shard_id="../escape", vocab_size=8,
        )
