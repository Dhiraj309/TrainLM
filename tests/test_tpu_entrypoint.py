"""CPU-side regression gates for the notebook launcher and binary preflight."""
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_entrypoint_does_not_import_accelerators(monkeypatch):
    original = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_xla", "transformers", "trainlm"}:
            raise AssertionError(f"Accelerator import at launcher module scope: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    module = load_script("trainlm_tpu_worker")
    assert callable(module.main)


def test_single_vm_launcher_clears_inherited_topology(monkeypatch):
    module = load_script("trainlm_tpu_worker")
    env = {"TPU_PROCESS_ADDRESSES": "local", "PJRT_LOCAL_PROCESS_COUNT": "1",
           "TPU_VISIBLE_CHIPS": "0", "XLA_USE_SPMD": "1", "XLA_USE_BF16": "1"}
    monkeypatch.setattr(module.os, "environ", env)
    module.configure_environment()
    assert env["PJRT_DEVICE"] == "TPU"
    assert env["USE_TF"] == "0"
    assert env["OMP_NUM_THREADS"] == "1"
    for name in ("TPU_PROCESS_ADDRESSES", "PJRT_LOCAL_PROCESS_COUNT",
                 "TPU_VISIBLE_CHIPS", "XLA_USE_SPMD", "XLA_USE_BF16"):
        assert name not in env


def test_raw_cli_requires_explicit_layout(monkeypatch):
    module = load_script("trainlm_tpu_worker")
    monkeypatch.setattr(sys, "argv", ["worker", "--data-mode", "raw", "--bin-path", "tokens.bin"])
    with pytest.raises(SystemExit):
        module.parse_args()
    monkeypatch.setattr(sys, "argv", ["worker", "--probe-only"])
    assert module.parse_args().probe_only


def raw_args(path, **changes):
    values = dict(bin_path=[str(path)], token_dtype="uint16", byte_order="little",
                  header_bytes=4, token_vocab_size=8, expected_token_count=4)
    values.update(changes)
    return SimpleNamespace(**values)


def test_raw_scan_produces_valid_reader_contract(tmp_path):
    from trainlm.data import validate_packed_binary_shard, ContiguousPackedBatchReader
    module = load_script("trainlm_tpu_data")
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"TLM1" + struct.pack("<4H", 1, 2, 3, 4))
    shards = module._raw_shards(raw_args(path))
    observed = validate_packed_binary_shard(shards[0].manifest, path)
    assert observed == shards[0].validation
    with ContiguousPackedBatchReader(shards, batch_size=1, sequence_length=4) as reader:
        assert reader.read_batch(0)["input_ids"].tolist() == [[1, 2, 3, 4]]


def test_raw_scan_rejects_bad_range_and_count(tmp_path):
    module = load_script("trainlm_tpu_data")
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"TLM1" + struct.pack("<4H", 1, 2, 3, 8))
    with pytest.raises(ValueError, match="vocabulary"):
        module._raw_shards(raw_args(path))
    with pytest.raises(ValueError, match="Token count"):
        module._raw_shards(raw_args(path, expected_token_count=5))

