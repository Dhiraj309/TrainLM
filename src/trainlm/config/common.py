"""
Shared configuration types.

This module contains reusable configuration objects shared across
multiple TrainLM configuration domains.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntervalConfig:
    """
    Trigger interval.

    At least one interval should typically be configured by the parent
    configuration object.
    """

    steps: int | None = None

    tokens: int | None = None
