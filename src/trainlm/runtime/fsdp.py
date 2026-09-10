"""Backend-neutral SPMD FSDP mesh and sharding policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import LogicalMesh

CheckpointLayout = Literal["distributed_rank_shards"]


@dataclass(frozen=True, slots=True)
class ParameterShardingRule:
    """Adapter-provided partitioning for parameters matching one suffix."""

    parameter_suffix: str
    partition_spec: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_suffix, str) or not self.parameter_suffix:
            raise ValueError("parameter_suffix cannot be empty.")
        if not self.partition_spec:
            raise ValueError("partition_spec cannot be empty.")
        if any(axis not in {None, "fsdp"} for axis in self.partition_spec):
            raise ValueError("Parameter partitions may only use the FSDP axis.")
        if "fsdp" not in self.partition_spec:
            raise ValueError("A parameter sharding rule must use the FSDP axis.")


@dataclass(frozen=True, slots=True)
class FSDPCheckpointPolicy:
    """Portable checkpoint requirements for a sharded training state."""

    layout: CheckpointLayout = "distributed_rank_shards"
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_rng_state: bool = True
    require_topology_match: bool = True

    def __post_init__(self) -> None:
        if self.layout != "distributed_rank_shards":
            raise ValueError(f"Unsupported FSDP checkpoint layout: {self.layout!r}.")
        for name in (
            "save_optimizer_state",
            "save_scheduler_state",
            "save_rng_state",
            "require_topology_match",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")


@dataclass(frozen=True, slots=True)
class FSDPMeshPolicy:
    """Declarative data/FSDP policy applied before runtime wrapping."""

    data_replicas: int
    fsdp_shards: int
    wrap_module_classes: tuple[str, ...]
    parameter_rules: tuple[ParameterShardingRule, ...]
    checkpoint: FSDPCheckpointPolicy = FSDPCheckpointPolicy()
    shard_optimizer_state: bool = True

    def __post_init__(self) -> None:
        for name in ("data_replicas", "fsdp_shards"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.wrap_module_classes or any(
            not isinstance(name, str) or not name.strip()
            for name in self.wrap_module_classes
        ):
            raise ValueError("wrap_module_classes must contain explicit class names.")
        if len(self.wrap_module_classes) != len(set(self.wrap_module_classes)):
            raise ValueError("wrap_module_classes must be unique.")
        if not self.parameter_rules or any(
            not isinstance(rule, ParameterShardingRule) for rule in self.parameter_rules
        ):
            raise ValueError("parameter_rules must contain explicit sharding rules.")
        suffixes = [rule.parameter_suffix for rule in self.parameter_rules]
        if len(suffixes) != len(set(suffixes)):
            raise ValueError("Parameter sharding suffixes must be unique.")
        if not isinstance(self.checkpoint, FSDPCheckpointPolicy):
            raise TypeError("checkpoint must be an FSDPCheckpointPolicy.")
        if not isinstance(self.shard_optimizer_state, bool):
            raise TypeError("shard_optimizer_state must be boolean.")

    @property
    def world_size(self) -> int:
        return self.data_replicas * self.fsdp_shards

    def logical_mesh(self) -> LogicalMesh:
        return LogicalMesh({"data": self.data_replicas, "fsdp": self.fsdp_shards})

    def validate_application_order(
        self,
        *,
        world_size: int,
        rematerialization_applied: bool,
        model_already_wrapped: bool,
    ) -> None:
        """Reject topology or ordering that would silently change semantics."""

        if world_size != self.world_size:
            raise ValueError(
                f"FSDP mesh requires world_size={self.world_size}; observed {world_size}."
            )
        if not rematerialization_applied:
            raise ValueError("Rematerialization must be applied before FSDP wrapping.")
        if model_already_wrapped:
            raise ValueError("FSDP policy must be applied before model wrapping.")


__all__ = [
    "CheckpointLayout",
    "FSDPCheckpointPolicy",
    "FSDPMeshPolicy",
    "ParameterShardingRule",
]
