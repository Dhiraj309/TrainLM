"""Private replicated-worker lifecycle hooks; importing this never loads XLA.

This synchronous, same-topology format is deliberately distinct from the M7
sharded resume manifest. Each worker owns its full replica plus RNG/data state.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from trainlm.data import PackedDataCursor, PackedDataCursorState
from trainlm.tasks import CausalLMTask


def _global_stop_steps(max_steps, max_tokens, *, sequence_length, batch_size,
                       accumulation_steps, world_size):
    """Round global supervised-token limits up to a full optimizer update."""
    tokens_per_update = (sequence_length - 1) * batch_size * accumulation_steps * world_size
    token_steps = ((max_tokens + tokens_per_update - 1) // tokens_per_update
                   if max_tokens is not None else None)
    limits = [value for value in (max_steps, token_steps) if value is not None]
    if not limits or min(limits) < 1:
        raise ValueError("Training requires a positive step or token limit.")
    return min(limits)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _WorkerBatches(IterableDataset):
    """Restart at the committed consumption index, independent of prefetch."""

    def __init__(self, reader):
        self.reader = reader
        self.start_index = 0

    def __len__(self):
        return len(self.reader) - self.start_index

    def __iter__(self):
        for index in range(self.start_index, len(self.reader)):
            batch = self.reader.read_batch(index)
            yield {"input_ids": batch["input_ids"], "labels": batch["labels"]}

    def cursor_at(self, consumed):
        if not 0 <= consumed <= len(self.reader):
            raise ValueError("Consumed batch count exceeds the rank-local partition.")
        initial = PackedDataCursor(self.reader).state
        global_index = (
            self.reader.plan.batch_indices[consumed]
            if consumed < len(self.reader) else None
        )
        location = self.reader.reader.locate(global_index) if global_index is not None else None
        return replace(
            initial,
            next_partition_index=consumed,
            batches_consumed=consumed,
            tokens_consumed=consumed * self.reader.reader.layout.tokens_per_batch,
            next_global_batch_index=global_index,
            next_shard_id=location.shard_id if location else None,
            next_local_batch_index=location.local_batch_index if location else None,
            next_token_offset=location.token_offset if location else None,
        )


class _WorkerCheckpointIO:
    """Publish only after every rank has written a checked, atomic payload."""

    def __init__(self, output_dir, batches, context, *, get_rng, set_rng, exchange_error=None):
        self.output_dir = Path(output_dir)
        self.batches = batches
        self.context = context
        self.get_rng = get_rng
        self.set_rng = set_rng
        self.exchange_error = exchange_error
        self.saved_paths = []

    def _collective(self, engine, operation):
        """Bring every worker to the same outcome before the next barrier."""
        if engine.runtime.world_size > 1 and self.exchange_error is None:
            raise ValueError("Multi-rank checkpoint I/O requires collective error exchange.")
        error = None
        result = None
        try:
            result = operation()
        except Exception as exc:
            error = exc
        if self.exchange_error is not None:
            message = self.exchange_error(str(error) if error is not None else None)
            if message is not None:
                raise ValueError(f"Worker checkpoint failed: {message}") from error
        elif error is not None:
            raise error
        return result

    @staticmethod
    def _optimizer_names(engine):
        names = {id(parameter): name for name, parameter in engine.model.named_parameters()}
        return [[names[id(parameter)] for parameter in group["params"]]
                for group in engine.optimizer.param_groups]

    def save(self, engine, destination):
        runtime = engine.runtime
        path = self.output_dir / str(destination or f"checkpoint-{engine.state.step}")
        # Never mix a new transaction with old or interrupted rank files.
        self._collective(engine, lambda: path.mkdir(parents=True, exist_ok=False)
                         if runtime.is_primary_process else None)
        self._collective(engine, lambda: self._write_rank(engine, path))
        self._collective(engine, lambda: self._commit(engine, path)
                         if runtime.is_primary_process else None)
        self.saved_paths.append(str(path))
        return path

    def _write_rank(self, engine, path):
        runtime = engine.runtime
        optimizer = engine.optimizer.state_dict()
        names = self._optimizer_names(engine)
        id_to_name = {index: name for group, group_names in zip(optimizer["param_groups"], names)
                      for index, name in zip(group["params"], group_names)}
        optimizer = {
            "state": {id_to_name[index]: state for index, state in optimizer["state"].items()},
            "param_groups": [{**group, "params": group_names}
                             for group, group_names in zip(optimizer["param_groups"], names)],
        }
        payload = _cpu_tree({
            "schema_version": 1,
            "format": "trainlm_replicated_worker",
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "context": self.context,
            "model": engine.model.state_dict(),
            "optimizer": optimizer,
            "scheduler": engine.scheduler.state_dict(),
            "runtime": dict(runtime.state_dict()),
            "trainer": {key: value for key, value in asdict(engine.state).items()
                        if key not in {"phase", "is_training", "should_stop", "failure"}},
            "cpu_rng": torch.get_rng_state(),
            "device_rng": self.get_rng(),
            "data": self.batches.cursor_at(engine.state.micro_step).to_dict(),
        })
        rank_path = path / f"rank-{runtime.rank}.pt"
        temporary = rank_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(rank_path)
        record = {"rank": runtime.rank, "step": engine.state.step,
                  "micro_step": engine.state.micro_step, "context": self.context,
                  "size": rank_path.stat().st_size, "sha256": _digest(rank_path)}
        (path / f"rank-{runtime.rank}.json").write_text(json.dumps(record), encoding="utf-8")

    def _commit(self, engine, path):
        records = [json.loads((path / f"rank-{rank}.json").read_text())
                   for rank in range(engine.runtime.world_size)]
        for rank, item in enumerate(records):
            if (item["rank"] != rank or item["step"] != engine.state.step
                    or item["micro_step"] != engine.state.micro_step
                    or item["context"] != self.context):
                raise ValueError("Checkpoint replicas disagree on progress or configuration.")
        commit = {"schema_version": 1, "format": "trainlm_replicated_worker",
                  "world_size": engine.runtime.world_size, "ranks": records}
        temporary = path / "commit.json.tmp"
        temporary.write_text(json.dumps(commit, sort_keys=True), encoding="utf-8")
        temporary.replace(path / "commit.json")

    def load(self, engine, source):
        return self._collective(engine, lambda: self._load_rank(engine, source))

    def _load_rank(self, engine, source):
        path = Path(source)
        commit_path = path / "commit.json"
        if not commit_path.is_file():
            raise ValueError("Worker checkpoint is incomplete: commit.json is missing.")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        runtime = engine.runtime
        if (commit.get("schema_version") != 1
                or commit.get("format") != "trainlm_replicated_worker"
                or commit.get("world_size") != runtime.world_size):
            raise ValueError("Worker checkpoint format or world size does not match.")
        records = commit["ranks"]
        if [item["rank"] for item in records] != list(range(runtime.world_size)):
            raise ValueError("Worker checkpoint does not contain every rank.")
        for rank, record in enumerate(records):
            rank_path = path / f"rank-{rank}.pt"
            if not rank_path.is_file() or rank_path.stat().st_size != record["size"]:
                raise ValueError("Worker checkpoint rank payload is missing or truncated.")
            if record["context"] != self.context:
                raise ValueError("Worker checkpoint configuration does not match this run.")
        rank_path = path / f"rank-{runtime.rank}.pt"
        record = records[runtime.rank]
        if _digest(rank_path) != record["sha256"]:
            raise ValueError("Worker checkpoint payload checksum does not match.")
        # The format contains tensors and primitive containers only.
        payload = torch.load(rank_path, map_location="cpu", weights_only=True)
        if (payload["schema_version"] != 1 or payload["format"] != commit["format"]
                or payload["rank"] != runtime.rank or payload["world_size"] != runtime.world_size
                or payload["context"] != self.context):
            raise ValueError("Worker checkpoint payload identity does not match.")
        state = payload["trainer"]
        if (state["step"] != record["step"] or state["micro_step"] != record["micro_step"]
                or state["micro_step"] != state["step"] * engine.config.trainer.gradient_accumulation_steps
                or state["step"] > engine.config.trainer.max_steps):
            raise ValueError("Worker checkpoint progress does not match the training run.")
        cursor = PackedDataCursorState.from_dict(payload["data"])
        PackedDataCursor.from_state(self.batches.reader, cursor)
        if cursor.next_partition_index != state["micro_step"]:
            raise ValueError("Worker checkpoint data cursor does not match consumed batches.")
        optimizer = payload["optimizer"]
        names = self._optimizer_names(engine)
        if [group["params"] for group in optimizer["param_groups"]] != names:
            raise ValueError("Worker checkpoint optimizer parameter ownership changed.")
        current = engine.optimizer.state_dict()
        name_to_id = {name: index for group, group_names in zip(current["param_groups"], names)
                      for index, name in zip(group["params"], group_names)}
        optimizer = {
            "state": {name_to_id[name]: value for name, value in optimizer["state"].items()},
            "param_groups": [{**group, "params": current_group["params"]}
                             for group, current_group in zip(optimizer["param_groups"], current["param_groups"])],
        }
        engine.model.load_state_dict(payload["model"])
        engine.optimizer.load_state_dict(optimizer)
        # AdamW's generic loader casts moments to parameter dtype; restore the
        # worker's explicit compact moment policy before its next update.
        cast_moments = getattr(engine.optimizer, "_cast_moment_state", None)
        if callable(cast_moments):
            cast_moments()
        engine.scheduler.load_state_dict(payload["scheduler"])
        runtime.load_state_dict(payload["runtime"])
        for key, value in state.items():
            setattr(engine.state, key, value)
        self.batches.start_index = cursor.next_partition_index
        engine._train_iterator = None
        torch.set_rng_state(payload["cpu_rng"])
        self.set_rng(payload["device_rng"])
        return path


class _WorkerEvaluationTask(CausalLMTask):
    """Reduce weighted evaluation sums across replicas before host metrics."""

    def __init__(self, *, reduce_sum, **kwargs):
        super().__init__(**kwargs)
        self.reduce_sum = reduce_sum

    def aggregate_evaluation_stream(self, results):
        import math

        numerator = None
        weight = 0
        for result in results:
            count = result.tokens.supervised_tokens
            contribution = result.loss.detach() * count
            numerator = contribution if numerator is None else numerator + contribution
            weight += count
        if numerator is None or weight == 0:
            raise ValueError("Evaluation contains no normalization units.")
        totals = self.reduce_sum(torch.stack((numerator, numerator.new_tensor(weight))))
        loss = (totals[0] / totals[1]).item()
        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return {"eval_loss": loss, "eval_perplexity": perplexity}
