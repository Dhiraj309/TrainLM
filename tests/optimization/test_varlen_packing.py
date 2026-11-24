"""CPU tests for Step 7: document-isolated varlen sequence packing."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from trainlm.api import _maybe_pack_token_dataset, compute_varlen_cu_seqlens


class _VarLenDataset(Dataset):
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self._rows[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def test_packed_dataset_emits_position_ids_and_masks_cross_doc_boundary() -> None:
    # Doc 0: [1, 2] -> [1, 2, 0] (pos 0, 1, 2)
    # Doc 1: [3, 4] -> [3, 4, 0] (pos 0, 1, 2)
    # Chunk 0 (len 4): [1, 2, 0, 3], pos=[0, 1, 2, 0], labels=[1, 2, 0, -100]
    ds = _VarLenDataset([[1, 2], [3, 4]])
    packed = _maybe_pack_token_dataset(ds, sequence_length=4, eos_token_id=0)
    assert len(packed) == 1
    row = packed[0]
    assert row["input_ids"].tolist() == [1, 2, 0, 3]
    assert row["position_ids"].tolist() == [0, 1, 2, 0]
    assert row["labels"].tolist() == [1, 2, 0, -100]


def test_compute_varlen_cu_seqlens_from_packed_batch() -> None:
    pos = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    cu_seqlens, max_seqlen = compute_varlen_cu_seqlens(pos)
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 3, 5, 7, 10]
    assert max_seqlen == 3
