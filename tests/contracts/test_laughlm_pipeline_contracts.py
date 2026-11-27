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
        "Dhiraj45/laughlm-135m-shards-v2",
        subfolder="stage1",
        max_shards=1,
        list_files_fn=lambda **_: fake_files,
    )
    assert found == ("stage1/stream_stage1_w00_shard_00001.bin",)

    all_stage1 = discover_hub_subfolder_bins(
        "Dhiraj45/laughlm-135m-shards-v2",
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


def test_laughlm_stage_cli_parsers_and_defaults():
    mods = (
        ("scripts.laughlm.stage0_benchmark", ["--max-shards", "1", "--no-compile", "--dry-run"]),
        ("scripts.laughlm.stage1_base_30b", ["--resume", "latest", "--dry-run"]),
        ("scripts.laughlm.stage2_anneal_5b", ["--init-checkpoint", "synthetic", "--dry-run"]),
        ("scripts.laughlm.stage3_sft", ["--init-checkpoint", "synthetic", "--dry-run"]),
        ("scripts.laughlm.stage4_dpo", ["--init-checkpoint", "synthetic", "--dry-run"]),
        ("scripts.laughlm.watch_and_upload", ["--checkpoint-dir", "runs/s1", "--once"]),
    )
    for mod_name, argv in mods:
        mod = importlib.import_module(mod_name)
        ns = mod.parse_args(argv)
        assert ns is not None

    from scripts.laughlm.common import (
        DEFAULT_OUTPUT_REPO,
        DEFAULT_SHARD_REPO,
        resolve_logits_chunk_size,
        resolve_loss_rematerialization,
        resolve_optim,
    )
    from scripts.laughlm.stage0_benchmark import (
        parse_args as parse_s0,
        recommended_stage1_grad_accum,
        recommended_stage2_grad_accum,
    )
    from scripts.laughlm.stage1_base_30b import resolve_grad_accum as s1_ga
    from scripts.laughlm.stage2_anneal_5b import resolve_grad_accum as s2_ga
    from scripts.laughlm.watch_and_upload import DEFAULT_HF_REPO

    s0_defaults = parse_s0([])
    assert s0_defaults.micro_batch_sizes == "32,64,96,128"
    assert s0_defaults.torch_compile_mode == "default"
    assert resolve_optim("tpu") == "adamw_bf16"
    assert resolve_optim("cuda") == "adamw_torch"
    assert resolve_logits_chunk_size(16384, accel_kind="tpu") == 2048
    assert resolve_logits_chunk_size(16384, accel_kind="cuda") == 16384
    assert resolve_loss_rematerialization("auto", accel_kind="tpu") == "per_chunk"
    assert resolve_loss_rematerialization("auto", accel_kind="cuda") == "disabled"
    assert recommended_stage1_grad_accum(128) == 2
    assert recommended_stage1_grad_accum(64) == 4
    assert recommended_stage2_grad_accum(64) == 2
    assert recommended_stage2_grad_accum(32) == 4
    assert s1_ga(128, None) == 2
    assert s1_ga(64, None) == 4
    assert s2_ga(64, None) == 2
    assert s2_ga(128, None) == 1

    assert DEFAULT_SHARD_REPO == "Dhiraj45/laughlm-135m-shards-v2"
    assert DEFAULT_OUTPUT_REPO == "Dhiraj45/laughlm-135m-v2"
    assert DEFAULT_HF_REPO == "Dhiraj45/laughlm-135m-v2"

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts" / "laughlm"
    for py_file in scripts_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "adamw_torch_fused" not in text, py_file
        assert "Dhiraj309/laughlm" not in text, py_file
        assert "result.performance" not in text, py_file


def test_watch_and_upload_skips_tmp_and_incomplete_dirs(tmp_path: Path):
    from scripts.laughlm.watch_and_upload import scan_completed_checkpoints

    (tmp_path / ".checkpoint-001.tmp.123").mkdir()
    (tmp_path / "checkpoint-001.tmp").mkdir()
    (tmp_path / "checkpoint-002-empty").mkdir()
    partial = tmp_path / "checkpoint-003-saving"
    partial.mkdir()
    (partial / "trainer_state.pt").write_bytes(b"old")
    (partial / "trainer_state.pt.tmp").write_bytes(b"in-flight")

    ready = tmp_path / "checkpoint-004"
    ready.mkdir()
    (ready / "trainer_state.pt").write_bytes(b"final")

    found = scan_completed_checkpoints(tmp_path)
    assert [p.name for p in found] == ["checkpoint-004"]


def test_laughlm_dry_runs_end_to_end_cpu():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.laughlm import (
        stage0_benchmark,
        stage1_base_30b,
        stage2_anneal_5b,
        stage3_sft,
        stage4_dpo,
    )

    assert stage0_benchmark.main(["--dry-run"]) == 0
    assert stage1_base_30b.main(["--dry-run"]) == 0
    assert stage2_anneal_5b.main(["--init-checkpoint", "synthetic", "--dry-run"]) == 0
    assert stage3_sft.main(["--init-checkpoint", "synthetic", "--dry-run"]) == 0
    assert stage4_dpo.main(["--init-checkpoint", "synthetic", "--dry-run"]) == 0


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
