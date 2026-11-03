from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD

from trainlm import TrainLMTrainer, TrainLMTrainingArguments
from trainlm.runtime import Runtime
from trainlm.training.callback import TrainerCallback


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class RecordingTokenScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.tokens_seen = None

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step_tokens(self, tokens_seen: int) -> None:
        self.tokens_seen = tokens_seen


class MetricsRecorder(TrainerCallback):
    def __init__(self):
        self.metrics: dict[str, float] = {}

    def on_metrics(self, state, control, metrics):
        self.metrics = dict(metrics)


class GlooRuntime(Runtime):
    """Test-only DDP backend using CPU/Gloo collectives."""

    @property
    def name(self) -> str:
        return "gloo-test"

    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary_process(self) -> bool:
        return self.rank == 0

    @property
    def mean_reduces_gradients_before_clipping(self) -> bool:
        return True

    def prepare_model(self, model: nn.Module) -> nn.Module:
        model = super().prepare_model(model)
        return DistributedDataParallel(model)

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced


def _rank_examples(rank: int) -> list[dict[str, torch.Tensor]]:
    if rank == 0:
        rows = [
            ([1, 2, 3, 4], [1, 2, -100, -100]),  # One supervised target.
            ([1, 2, 3, 4], [1, 2, 3, -100]),  # Two supervised targets.
        ]
    else:
        rows = [
            ([2, 3, 4, 5], [2, 3, 4, -100]),  # Two supervised targets.
            ([2, 3, 4, 5], [2, 3, 4, 5]),  # Three supervised targets.
        ]
    return [
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        for input_ids, labels in rows
    ]


def _token_budget_examples(rank: int) -> list[dict[str, torch.Tensor]]:
    labels = [1, 2, 3, 4] if rank == 0 else [1, 2, -100, -100]
    return [
        {
            "input_ids": torch.tensor([1 + index, 2, 3, 4], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        for index in range(3)
    ]


def _gloo_train_worker(rank: int, init_file: str, output_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(41)
        model = TinyCausalLM()
        learning_rate = 0.05
        metrics_recorder = MetricsRecorder()
        trainer = TrainLMTrainer(
            model=model,
            train_dataset=_rank_examples(rank),
            optimizer=SGD(model.parameters(), lr=learning_rate),
            callbacks=[metrics_recorder],
            runtime=GlooRuntime(device="cpu"),
            args=TrainLMTrainingArguments(
                output_dir=Path(output_dir) / f"rank-{rank}",
                accelerator="cpu",
                max_steps=1,
                sequence_length=4,
                gradient_accumulation_steps=2,
                learning_rate=learning_rate,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                logging_steps=1,
            ),
        )
        trainer.train()
        torch.save(
            {
                name: parameter.detach().cpu()
                for name, parameter in trainer.engine.model.module.state_dict().items()
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
        torch.save(
            {
                "tokens_seen": trainer.engine.state.tokens_seen,
                "samples_seen": trainer.engine.state.samples_seen,
                "global_batch_size": trainer.engine.state.global_batch_size,
                "loss": trainer.engine.state.loss,
                "metrics": metrics_recorder.metrics,
            },
            Path(output_dir) / f"rank-{rank}-metrics.pt",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gloo_token_budget_worker(rank: int, init_file: str, output_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(41)
        model = TinyCausalLM()
        optimizer = SGD(model.parameters(), lr=0.05)
        scheduler = RecordingTokenScheduler(optimizer)
        trainer = TrainLMTrainer(
            model=model,
            train_dataset=_token_budget_examples(rank),
            optimizer=optimizer,
            scheduler=scheduler,
            runtime=GlooRuntime(device="cpu"),
            args=TrainLMTrainingArguments(
                output_dir=Path(output_dir) / f"budget-rank-{rank}",
                accelerator="cpu",
                max_steps=None,
                max_tokens=4,
                sequence_length=4,
                gradient_accumulation_steps=3,
                learning_rate=0.05,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                logging_steps=1,
            ),
        )
        state = trainer.train()
        torch.save(
            {
                "step": state.step,
                "micro_step": state.micro_step,
                "tokens_seen": state.tokens_seen,
                "global_tokens_seen": trainer.engine._global_tokens_seen,
                "scheduled_tokens_seen": scheduler.tokens_seen,
            },
            Path(output_dir) / f"budget-rank-{rank}.pt",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_gloo_update_matches_global_supervised_token_mean(tmp_path: Path):
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _gloo_train_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    torch.manual_seed(41)
    reference = TinyCausalLM()
    optimizer = SGD(reference.parameters(), lr=0.05)
    examples = _rank_examples(0) + _rank_examples(1)
    input_ids = torch.stack([example["input_ids"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    logits = reference(input_ids).logits[:, :-1, :].reshape(-1, 8)
    targets = labels[:, 1:].reshape(-1)
    loss_sum = F.cross_entropy(
        logits,
        targets,
        ignore_index=-100,
        reduction="sum",
    )
    supervised_tokens = targets.ne(-100).sum()
    reference_loss = (loss_sum / supervised_tokens).item()
    (loss_sum / supervised_tokens).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=1.0)
    optimizer.step()

    expected = reference.state_dict()
    rank_metrics = []
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        rank_metrics.append(
            torch.load(tmp_path / f"rank-{rank}-metrics.pt", weights_only=True)
        )
        for name, expected_value in expected.items():
            torch.testing.assert_close(
                actual[name],
                expected_value,
                rtol=1e-6,
                atol=1e-7,
            )
    for report in rank_metrics:
        assert report["global_batch_size"] == 4
        assert report["loss"] == pytest.approx(reference_loss)
        assert report["metrics"]["tokens_seen"] == 8.0
        assert report["metrics"]["samples_seen"] == 4.0
        assert report["metrics"]["loss"] == pytest.approx(reference_loss)
    assert sorted(report["tokens_seen"] for report in rank_metrics) == [3, 5]


def test_gloo_max_tokens_stops_all_ranks_at_same_microbatch(tmp_path: Path):
    init_file = tmp_path / "gloo-budget-init"
    mp.spawn(
        _gloo_token_budget_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    reports = [
        torch.load(tmp_path / f"budget-rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for report in reports:
        assert report["step"] == 1
        assert report["micro_step"] == 1
        assert report["global_tokens_seen"] == 4
        assert report["scheduled_tokens_seen"] == 4
    assert sorted(report["tokens_seen"] for report in reports) == [1, 3]
