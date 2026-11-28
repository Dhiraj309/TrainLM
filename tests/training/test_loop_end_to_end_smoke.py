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


def test_stage1_stage2_build_training_args_max_steps_none_and_wsd_warmup(tmp_path: Path):
    from scripts.laughlm import stage1_base_30b, stage2_anneal_5b

    s1_args = stage1_base_30b.parse_args(
        [
            "--loop-preset",
            "pilot-1b",
            "--max-tokens",
            "250000000",
            "--micro-batch-size",
            "4",
            "--grad-accum",
            "16",
            "--no-compile",
            "--output-dir",
            str(tmp_path / "s1"),
        ]
    )
    t1 = stage1_base_30b.build_training_arguments(
        s1_args,
        accel_kind="cpu",
        rank=0,
        grad_accum=16,
        remat="auto",
        optim_name="adamw_torch",
        chunk_size=128,
    )
    assert t1.max_steps is None
    assert t1.max_tokens == 250_000_000

    cfg = _tiny_loop_config()
    model = LaughLMLoopForCausalLM(cfg)
    ds = [{"input_ids": torch.zeros(16, dtype=torch.long), "labels": torch.zeros(16, dtype=torch.long)}]
    trainer = TrainLMTrainer(model=model, args=t1, train_dataset=ds)
    sched_cfg = trainer._scheduler_config()
    assert sched_cfg.horizon_tokens == 250_000_000
    assert 0.0 < sched_cfg.warmup_fraction <= 1.0
    # 250M / (4 * 16 * 2048) = 1908 steps; warmup_steps=2000 caps at 1.0,
    # whereas 1B tokens (~7630 steps) keeps warmup_fraction well below 1.0:
    t1_1b = stage1_base_30b.build_training_arguments(
        stage1_base_30b.parse_args(
            [
                "--loop-preset",
                "pilot-1b",
                "--micro-batch-size",
                "4",
                "--grad-accum",
                "16",
                "--no-compile",
                "--output-dir",
                str(tmp_path / "s1_1b"),
            ]
        ),
        accel_kind="cpu",
        rank=0,
        grad_accum=16,
        remat="auto",
        optim_name="adamw_torch",
        chunk_size=128,
    )
    trainer_1b = TrainLMTrainer(model=model, args=t1_1b, train_dataset=ds)
    sched_1b = trainer_1b._scheduler_config()
    assert 0.20 < sched_1b.warmup_fraction < 0.30
    assert sched_1b.stable_fraction > 0.70

    s2_args = stage2_anneal_5b.parse_args(
        ["--init-checkpoint", "runs/s1/final_model", "--no-compile", "--output-dir", str(tmp_path / "s2")]
    )
    t2 = stage2_anneal_5b.build_training_arguments(
        s2_args,
        accel_kind="cpu",
        grad_accum=2,
        remat="auto",
        optim_name="adamw_torch",
        chunk_size=128,
    )
    assert t2.max_steps is None
    assert t2.max_tokens == 5_000_000_000


def test_ddp_safe_host_loop_depth_schedule_and_specializations(tmp_path: Path, monkeypatch):
    from collections import Counter
    import functools
    from trainlm.training.trainer import resolve_scheduled_num_loops

    schedule = (2, 4, 4, 6)
    rank0_seq = [
        resolve_scheduled_num_loops(schedule, step=step, micro_idx=m, grad_accum=16)
        for step in range(2)
        for m in range(16)
    ]
    rank1_seq = [
        resolve_scheduled_num_loops(schedule, step=step, micro_idx=m, grad_accum=16)
        for step in range(2)
        for m in range(16)
    ]
    assert rank0_seq == rank1_seq
    assert Counter(rank0_seq[:16]) == {2: 4, 4: 8, 6: 4}

    cfg = _tiny_loop_config()
    model = LaughLMLoopForCausalLM(cfg)
    observed_loops: list[int | None] = []
    orig_forward = model.model.forward

    @functools.wraps(orig_forward)
    def spy_forward(*args, **kwargs):
        observed_loops.append(kwargs.get("num_loops"))
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(model.model, "forward", spy_forward)
    gen = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, cfg.vocab_size, (16, 16), generator=gen)
    ds = [{"input_ids": row, "labels": row.clone()} for row in tokens]

    trainer = TrainLMTrainer(
        model=model,
        train_dataset=ds,
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "sched"),
            accelerator="cpu",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            sequence_length=16,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            torch_compile=False,
            loss_implementation="chunked_linear",
            logits_chunk_size=32,
            optim="adamw_torch",
            loop_depth_schedule=schedule,
            report_to="none",
        ),
    )
    state = trainer.train()
    assert state.step == 2
    assert observed_loops == [2, 4, 4, 6, 2, 4, 4, 6]

    # Verify compiled microbatch path creates only 3 specializations for (2, 4, 4, 6) over 2 steps.
    observed_loops.clear()
    compile_calls: list[int] = []
    compiled_trainer = TrainLMTrainer(
        model=model,
        train_dataset=ds,
        args=TrainLMTrainingArguments(
            output_dir=str(tmp_path / "sched_compiled"),
            accelerator="cpu",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            sequence_length=16,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            torch_compile=False,
            loss_implementation="chunked_linear",
            logits_chunk_size=32,
            optim="adamw_torch",
            loop_depth_schedule=schedule,
            report_to="none",
        ),
    )

    class _FakeCompiledDevice:
        type = "tpu"

    compiled_trainer.engine.runtime._compile_enabled = True
    monkeypatch.setattr(compiled_trainer.engine.runtime, "_device", _FakeCompiledDevice())
    monkeypatch.setattr(compiled_trainer.engine.runtime, "prepare_batch", lambda batch: batch)
    monkeypatch.setattr(
        compiled_trainer.engine.runtime,
        "compile_training_step",
        lambda fn: (compile_calls.append(len(compile_calls)), fn)[1],
        raising=False,
    )
    compiled_state = compiled_trainer.train()
    assert compiled_state.step == 2
    assert observed_loops == [2, 4, 4, 6, 2, 4, 4, 6]
    assert len(compile_calls) == 3
    assert set(compiled_trainer.engine._compiled_microbatch_step_fns.keys()) == {2, 4, 6}


