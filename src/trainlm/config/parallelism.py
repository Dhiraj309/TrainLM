"""Backend-neutral parallelism policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParallelismConfig:
    """Requested logical parallelism dimensions."""

    data: int = 1
    fsdp: int = 1
    tensor: int = 1
    sequence: int = 1
    pipeline: int = 1

    def __post_init__(self) -> None:
        for name in ("data", "fsdp", "tensor", "sequence", "pipeline"):
            if getattr(self, name) < 1:
                raise ValueError(f"'parallelism.{name}' must be at least 1.")

