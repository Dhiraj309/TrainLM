"""CPU evidence for worker lifecycle wiring, not TPU hardware certification."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from trainlm._tpu_lifecycle import _WorkerBatches, _WorkerCheckpointIO, _WorkerEvaluationTask
from trainlm.config import (
    CheckpointConfig, EvaluationConfig, LoggingConfig, OptimizerConfig, TrainConfig, TrainerConfig,
)
from trainlm.optimization import create_optimizer
from trainlm.data import PartitionedPackedBatchReader, plan_packed_batch_partition
from trainlm.runtime import TorchRuntime
from trainlm.tasks import CausalLMTask
from trainlm.training import Trainer, TrainerCallback
from tests.data.test_partition import _reader


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(8, 128)

    def forward(self, input_ids):
        return {"logits": self.head(self.dropout(self.embedding(input_ids)))}


class StopAtTwo(TrainerCallback):
    def on_step_end(self, state, control):
        if state.step == 2:
            control.request_stop()


def make_engine(tmp_path, partitioned, *, stop=False, context=None, compact=False):
    torch.manual_seed(7)
    batches = _WorkerBatches(partitioned)
    model = TinyModel()
    optimizer = (create_optimizer(model.parameters(), OptimizerConfig(
        learning_rate=0.01, fused=False, mu_dtype="bfloat16", nu_dtype="float32",
    )) if compact else torch.optim.AdamW(model.parameters(), lr=0.01))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    device_rng = {"state": 19}
    checkpoint = _WorkerCheckpointIO(
        tmp_path, batches, context or {"geometry": [2, 4]},
        get_rng=lambda: device_rng["state"],
        set_rng=lambda value: device_rng.update(state=value),
    )
    config = TrainConfig(
        trainer=TrainerConfig(max_steps=4, gradient_accumulation_steps=2),
        checkpoint=CheckpointConfig(output_dir=tmp_path, save_training_every_steps=2),
        evaluation=EvaluationConfig(eval_every_steps=2),
        logging=LoggingConfig(log_every_steps=1),
    )
    evaluation = DataLoader(_WorkerBatches(partitioned), batch_size=None,
                            generator=torch.Generator().manual_seed(13))
    trainer = Trainer(
        config=config, model=model, runtime=TorchRuntime(device="cpu"),
        optimizer=optimizer, scheduler=scheduler, task=CausalLMTask(loss_implementation="causal_lm"),
        train_dataloader=DataLoader(batches, batch_size=None,
                                    generator=torch.Generator().manual_seed(11)),
        eval_dataloader=evaluation, callbacks=[StopAtTwo()] if stop else [],
        checkpoint_saver=checkpoint.save, checkpoint_loader=checkpoint.load,
    )
    return trainer, checkpoint, device_rng


@pytest.mark.parametrize("compact", [False, True])
def test_worker_resume_matches_uninterrupted_dropout_training_and_cadence(tmp_path, compact):
    with _reader(tmp_path / "data", (96,)) as reader:
        partitioned = PartitionedPackedBatchReader(reader, plan_packed_batch_partition(
            reader, split="train", seed=7, epoch=0, world_size=1, rank=0,
        ))
        baseline, _, _ = make_engine(tmp_path / "baseline", partitioned, compact=compact)
        baseline.train()
        stopped, io, _ = make_engine(tmp_path / "interrupted", partitioned, stop=True, compact=compact)
        # Simulate a producer that has prefetched the full partition. The save
        # must still use trainer.micro_step, not producer position.
        list(iter(io.batches))
        stopped.train()
        checkpoint_path = tmp_path / "interrupted" / "checkpoint-2"
        payload = torch.load(checkpoint_path / "rank-0.pt", weights_only=True)
        assert payload["data"]["next_partition_index"] == 4
        assert set(payload["optimizer"]["state"]) == dict(stopped.model.named_parameters()).keys()
        resumed, resumed_io, rng = make_engine(tmp_path / "resumed", partitioned, compact=compact)
        rng["state"] = 0
        resumed.load_checkpoint(checkpoint_path)
        assert rng["state"] == 19
        assert resumed_io.batches.start_index == 4
        resumed.train()
        assert resumed.state.step == baseline.state.step == 4
        assert resumed.state.micro_step == baseline.state.micro_step == 8
        assert resumed.state.tokens_seen == baseline.state.tokens_seen
        assert resumed.scheduler.state_dict() == baseline.scheduler.state_dict()
        for key, expected in baseline.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], expected, rtol=0, atol=0)
        for actual, expected in zip(resumed.optimizer.state.values(), baseline.optimizer.state.values()):
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert (tmp_path / "baseline" / "checkpoint-2" / "commit.json").is_file()
        assert (tmp_path / "baseline" / "checkpoint-4" / "commit.json").is_file()


@pytest.mark.parametrize("damage, message", [
    ("commit", "incomplete"), ("payload", "checksum"),
    ("rank", "missing or truncated"), ("topology", "world size"),
    ("config", "configuration"),
    ("data", "partition plan"),
])
def test_worker_rejects_incomplete_corrupt_or_incompatible_resume(tmp_path, damage, message):
    with _reader(tmp_path / "data", (96,)) as reader:
        partitioned = PartitionedPackedBatchReader(reader, plan_packed_batch_partition(
            reader, split="train", seed=7, epoch=0, world_size=1, rank=0,
        ))
        trainer, _, _ = make_engine(tmp_path / "run", partitioned, stop=True)
        trainer.train()
        path = tmp_path / "run" / "checkpoint-2"
        commit = path / "commit.json"
        if damage == "commit":
            commit.unlink()
        elif damage == "payload":
            data = bytearray((path / "rank-0.pt").read_bytes())
            data[-1] ^= 1
            (path / "rank-0.pt").write_bytes(data)
        elif damage == "rank":
            (path / "rank-0.pt").unlink()
        elif damage == "topology":
            document = json.loads(commit.read_text())
            document["world_size"] = 8
            commit.write_text(json.dumps(document))
        target_partition = partitioned
        if damage == "data":
            target_partition = PartitionedPackedBatchReader(reader, plan_packed_batch_partition(
                reader, split="train", seed=8, epoch=0, world_size=1, rank=0,
            ))
        target, _, _ = make_engine(tmp_path / "target", target_partition,
                                    context={"changed": True} if damage == "config" else None)
        before = {key: value.clone() for key, value in target.model.state_dict().items()}
        with pytest.raises(ValueError, match=message):
            target.load_checkpoint(path)
        for key, expected in before.items():
            torch.testing.assert_close(target.model.state_dict()[key], expected)


def test_worker_evaluation_reduces_weighted_sums_before_metrics():
    # Local losses 2, 4 with weights 2, 1; another replica adds sum=12, weight=3.
    seen = []
    def reduce_sum(value):
        seen.append(value.tolist())
        return value + torch.tensor([12.0, 3.0])
    task = _WorkerEvaluationTask(reduce_sum=reduce_sum)
    results = [SimpleNamespace(loss=torch.tensor(loss),
                               tokens=SimpleNamespace(supervised_tokens=weight))
               for loss, weight in ((2.0, 2), (4.0, 1))]
    metrics = task.aggregate_evaluation_stream(iter(results))
    assert seen == [[8.0, 3.0]]
    assert metrics["eval_loss"] == pytest.approx(20 / 6)


@pytest.mark.parametrize("fail_rank", [None, 1])
def test_replicated_checkpoint_commit_waits_for_all_rank_writes(tmp_path, fail_rank):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2, timeout=10)
    errors = [None, None]
    def exchange(rank, error):
        errors[rank] = error
        barrier.wait()
        result = next((value for value in errors if value is not None), None)
        barrier.wait()
        return result

    engines, writers = [], []
    with _reader(tmp_path / "data", (96,)) as reader:
        for rank in range(2):
            plan = plan_packed_batch_partition(
                reader, split="train", seed=7, epoch=0, world_size=2, rank=rank,
            )
            engine, writer, _ = make_engine(tmp_path / "run", PartitionedPackedBatchReader(reader, plan))
            engine.runtime = SimpleNamespace(rank=rank, world_size=2,
                is_primary_process=rank == 0, state_dict=lambda: {})
            writer.exchange_error = lambda error, rank=rank: exchange(rank, error)
            if rank == fail_rank:
                writer.get_rng = lambda: (_ for _ in ()).throw(OSError("rank write failed"))
            engines.append(engine)
            writers.append(writer)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(writer.save, engine, "checkpoint-0")
                       for writer, engine in zip(writers, engines)]
            if fail_rank is None:
                for future in futures:
                    future.result(timeout=15)
            else:
                for future in futures:
                    with pytest.raises(ValueError, match="rank write failed"):
                        future.result(timeout=15)
        commit = tmp_path / "run" / "checkpoint-0" / "commit.json"
        assert commit.is_file() == (fail_rank is None)
        if fail_rank is None:
            assert len(json.loads(commit.read_text())["ranks"]) == 2


def test_worker_training_body_wires_lifecycle_using_cpu_runtime(tmp_path, monkeypatch):
    """Execute the real body with XLA APIs replaced by CPU-only doubles."""
    import sys
    from types import ModuleType
    from tests.test_tpu_entrypoint import load_script

    xla = ModuleType("torch_xla")
    core = ModuleType("torch_xla.core")
    xm = ModuleType("torch_xla.core.xla_model")
    distributed = ModuleType("torch_xla.distributed")
    pl = ModuleType("torch_xla.distributed.parallel_loader")
    xr = ModuleType("torch_xla.runtime")
    debug = ModuleType("torch_xla.debug")
    debug_metrics = ModuleType("torch_xla.debug.metrics")
    xla.core, core.xla_model = core, xm
    xla.distributed, distributed.parallel_loader = distributed, pl
    xla.runtime, xla.debug, debug.metrics = xr, debug, debug_metrics
    xla.__version__ = "cpu-double"
    xla.device = lambda: torch.device("cpu")
    xla.manual_seed = lambda *args, **kwargs: None
    xla.sync = lambda **kwargs: None
    xr.global_ordinal, xr.world_size = lambda: 0, lambda: 1
    xm.REDUCE_SUM, xm.REDUCE_MAX = "sum", "max"
    xm.all_reduce = lambda operation, value: value
    xm.get_rng_state = lambda device: 7
    xm.set_rng_state = lambda value, device: None
    xm.rendezvous = lambda name: None
    xm.mesh_reduce = lambda name, value, reducer: reducer([value])
    debug_metrics.metrics_report = lambda: "CPU test double"
    for module in (xla, core, xm, distributed, pl, xr, debug, debug_metrics):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    closed = []
    class ParallelLoader:
        def __init__(self, loader, devices, **kwargs):
            self.loader = loader
        def per_device_loader(self, device):
            return self.loader
        def close(self):
            closed.append(True)
    pl.ParallelLoader = ParallelLoader

    module = load_script("trainlm_tpu_training")
    class CPURuntime(TorchRuntime):
        def __init__(self, device, precision, **kwargs):
            super().__init__(device=device, precision=precision)
    monkeypatch.setattr(module, "XlaRuntime", CPURuntime)
    monkeypatch.setattr(module, "_install_xla_attention", lambda model: None)
    def load_model(source):
        model = TinyModel()
        config = SimpleNamespace(use_cache=False, tie_word_embeddings=False,
                                 to_dict=lambda: {"vocab_size": 128})
        model.config = config
        model.get_input_embeddings = lambda: model.embedding
        model.tie_weights = lambda: None
        return SimpleNamespace(model=model, config=config,
            metadata=SimpleNamespace(resolved_revision=None, to_dict=lambda: {}))
    monkeypatch.setattr(module, "load_huggingface_causal_lm", load_model)
    entrypoint = load_script("trainlm_tpu_worker")
    monkeypatch.setattr(sys, "argv", ["worker", "--expected-world-size", "1",
        "--max-steps", "2", "--warmup-steps", "0", "--gradient-accumulation-steps", "1",
        "--micro-batch-per-device", "2", "--sequence-length", "4",
        "--precision", "fp32", "--scheduler", "constant", "--log-every-steps", "1",
        "--save-steps", "1", "--eval-steps", "1", "--eval-manifest-dir", "unused",
        "--output-dir", str(tmp_path / "run")])
    args = entrypoint.parse_args()
    with _reader(tmp_path / "data", (96,)) as reader:
        module.train_fn(0, args, {"train": reader.shards, "evaluation": reader.shards})
        summary = json.loads((tmp_path / "run" / "summary.json").read_text())
        assert summary["steps"] == 2 and len(summary["checkpoints"]) == 2
        metrics = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
        assert [item["step"] for item in metrics if "eval_loss" in item] == [1, 2]
        args.resume_from_checkpoint = str(tmp_path / "run" / "checkpoint-2")
        args.output_dir = str(tmp_path / "resumed")
        args.max_steps = 3
        module.train_fn(0, args, {"train": reader.shards, "evaluation": reader.shards})
        summary = json.loads((tmp_path / "resumed" / "summary.json").read_text())
        assert summary["steps"] == 3 and summary["micro_steps"] == 3
        assert len(summary["checkpoints"]) == 1
        assert summary["resumed_from_checkpoint"] == args.resume_from_checkpoint
    assert len(closed) == 5  # two training loaders and three evaluation passes
