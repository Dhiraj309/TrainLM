"""End-to-end CPU smoke test for LaughLM-Loop: .bin shards -> train -> save/resume -> safetensors -> eval_loop_scaling."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import torch

from scripts.laughlm import eval_loop_scaling
from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.config import LaughLMLoopConfig
from trainlm.data import PackedBinaryShardManifest, PackedBinDataset
from trainlm.model import LaughLMLoopForCausalLM


def _write_synthetic_shard(directory: Path, shard_id: str, tokens: list[int], *, vocab_size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bin_name = f"{shard_id}.bin"
    raw = struct.pack("<" + "H" * len(tokens), *tokens)
    (directory / bin_name).write_bytes(raw)
    manifest = PackedBinaryShardManifest(
        schema_version=1,
        shard_id=shard_id,
        data_path=bin_name,
        compatibility_profile="explicit_v1",
        header_bytes=0,
        token_dtype="uint16",
        byte_order="little",
        token_count=len(tokens),
        token_id_min=min(tokens),
        token_id_max=max(tokens),
        vocab_size=vocab_size,
        file_size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    (directory / f"{shard_id}.manifest.json").write_text(
        manifest.to_json() + "\n", encoding="utf-8"
    )


def _tiny_loop_config() -> LaughLMLoopConfig:
    return LaughLMLoopConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_prelude_layers=1,
        num_core_layers=1,
        num_coda_layers=1,
        num_loop_iterations=4,
        train_min_loops=2,
        train_max_loops=4,
        truncated_bptt_steps=4,
        use_pallas_recurrent_norm=False,
        max_position_embeddings=32,
    )


def test_laughlm_loop_end_to_end_bin_train_resume_export_and_scaling(tmp_path: Path):
    torch.manual_seed(0)
    shard_dir = tmp_path / "shards"
    tokens = [(i * 7 + 3) % 64 for i in range(512)]
    _write_synthetic_shard(shard_dir, "shard_00001", tokens, vocab_size=64)

    dataset = PackedBinDataset.from_directory(shard_dir, sequence_length=16)
    cfg = _tiny_loop_config()
    model = LaughLMLoopForCausalLM(cfg)
    out_dir = tmp_path / "run"

    args = TrainLMTrainingArguments(
        output_dir=str(out_dir),
        accelerator="cpu",
        per_device_train_batch_size=2,
        sequence_length=16,
        max_steps=4,
        logging_steps=2,
        save_strategy="steps",
        save_steps=2,
        bf16=False,
        torch_compile=False,
        loss_implementation="chunked_linear",
        logits_chunk_size=32,
        optim="adamw_torch",
        lr_scheduler_type="wsd",
        warmup_steps=1,
        report_to="none",
    )

    trainer = TrainLMTrainer(model=model, args=args, train_dataset=dataset)
    state = trainer.train()
    assert state.step == 4
    ckpt2 = out_dir / "checkpoint-2"
    ckpt4 = out_dir / "checkpoint-4"
    assert (ckpt2 / "trainer_state.pt").is_file()
    assert (ckpt4 / "trainer_state.pt").is_file()

    # Resume from step 2 -> step 4 cleanly.
    resumed_trainer = TrainLMTrainer(
        model=LaughLMLoopForCausalLM(cfg),
        args=args,
        train_dataset=PackedBinDataset.from_directory(shard_dir, sequence_length=16),
    )
    resumed_state = resumed_trainer.train(resume_from_checkpoint=str(ckpt2))
    assert resumed_state.step == 4

    export_dir = out_dir / "final_export"
    resumed_trainer.save_model(export_dir)
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "config.json").is_file()

    # Evaluate test-time depth scaling R in [1, 2, 4] via eval_loop_scaling helpers.
    eval_model = LaughLMLoopForCausalLM.from_pretrained(export_dir)
    eval_loader = torch.utils.data.DataLoader(
        PackedBinDataset.from_directory(shard_dir, sequence_length=16),
        batch_size=2,
    )
    batches = [next(iter(eval_loader))]
    rows = eval_loop_scaling._evaluate_loop_depths(
        eval_model,
        batches,
        loops_list=[1, 2, 4],
        device=torch.device("cpu"),
    )
    assert [int(r["loops"]) for r in rows] == [1, 2, 4]
    losses = [r["loss"] for r in rows]
    assert len({round(val, 6) for val in losses}) >= 2
    assert all(r["tokens_per_second"] > 0 for r in rows)
