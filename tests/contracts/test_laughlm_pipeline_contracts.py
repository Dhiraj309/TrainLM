"""Contract tests for LaughLM subfolder discovery, 6 stage CLIs, and DPO math."""

from __future__ import annotations

import importlib
from pathlib import Path
import struct

import pytest


def test_discover_hub_subfolder_bins_sorted_1_indexed_and_max_shards():
    torch = pytest.importorskip("torch")
    del torch
    from trainlm.data.public import discover_hub_subfolder_bins

    fake_files = [
        "README.md",
        "stage2/stream_stage2_w00_shard_00001.bin",
        "stage1/stream_stage1_w00_shard_00002.bin",
        "stage1/stream_stage1_w00_shard_00001.bin",
        "stage1/notes.txt",
    ]
    found = discover_hub_subfolder_bins(
        "Dhiraj309/laughlm-135m-shards-v2",
        subfolder="stage1",
        max_shards=1,
        list_files_fn=lambda **_: fake_files,
    )
    assert found == ("stage1/stream_stage1_w00_shard_00001.bin",)

    all_stage1 = discover_hub_subfolder_bins(
        "Dhiraj309/laughlm-135m-shards-v2",
        subfolder="stage1",
        list_files_fn=lambda **_: fake_files,
    )
    assert all_stage1 == (
        "stage1/stream_stage1_w00_shard_00001.bin",
        "stage1/stream_stage1_w00_shard_00002.bin",
    )


def test_headerless_u2_token_count_from_file_size(tmp_path: Path):
    torch = pytest.importorskip("torch")
    del torch
    from trainlm.data.public import _inspect_legacy_uint16_bin

    tokens = [0, 42, 49151, 100, 7]
    shard = tmp_path / "stream_stage1_w00_shard_00001.bin"
    shard.write_bytes(struct.pack("<" + "H" * len(tokens), *tokens))
    manifest, validation = _inspect_legacy_uint16_bin(
        shard,
        shard_id="stream_stage1_w00_shard_00001",
        data_path="stage1/stream_stage1_w00_shard_00001.bin",
        header_bytes=0,
        vocab_size=49152,
    )
    assert manifest.header_bytes == 0
    assert manifest.token_dtype == "uint16"
    assert manifest.byte_order == "little"
    assert manifest.token_count == len(tokens)
    assert validation.token_count == len(tokens)
    assert manifest.token_id_max == 49151


def test_laughlm_stage_cli_parsers():
    mods = (
        ("scripts.laughlm.stage0_benchmark", ["--max-shards", "1", "--no-compile"]),
        ("scripts.laughlm.stage1_base_30b", ["--resume", "latest"]),
        ("scripts.laughlm.stage2_anneal_5b", ["--init-checkpoint", "ckpt1"]),
        ("scripts.laughlm.stage3_sft", ["--init-checkpoint", "ckpt2"]),
        ("scripts.laughlm.stage4_dpo", ["--init-checkpoint", "ckpt3"]),
        ("scripts.laughlm.watch_and_upload", ["--checkpoint-dir", "runs/s1", "--once"]),
    )
    for mod_name, argv in mods:
        mod = importlib.import_module(mod_name)
        ns = mod.parse_args(argv)
        assert ns is not None


def test_watch_and_upload_skips_tmp_dirs(tmp_path: Path):
    from scripts.laughlm.watch_and_upload import scan_completed_checkpoints

    (tmp_path / ".checkpoint-001.tmp.123").mkdir()
    (tmp_path / "checkpoint-001.tmp").mkdir()
    (tmp_path / "checkpoint-001").mkdir()
    found = scan_completed_checkpoints(tmp_path)
    assert [p.name for p in found] == ["checkpoint-001"]


def test_dpo_loss_and_sequence_logps():
    torch = pytest.importorskip("torch")
    from trainlm.dpo import compute_dpo_loss, compute_sequence_logps

    logits = torch.randn(2, 5, 16)
    labels = torch.tensor([[-100, -100, 3, 4, 0], [-100, 2, 5, 6, -100]])
    logps = compute_sequence_logps(logits, labels, label_pad_token_id=-100)
    assert logps.shape == (2,)
    assert torch.all(logps <= 0)

    loss, c_rew, r_rew = compute_dpo_loss(
        torch.tensor([-1.0, -2.0]),
        torch.tensor([-3.0, -4.0]),
        torch.tensor([-2.0, -2.5]),
        torch.tensor([-2.5, -3.5]),
        beta=0.1,
    )
    assert loss.ndim == 0 and float(loss) > 0.0
    assert c_rew.shape == (2,) and r_rew.shape == (2,)
