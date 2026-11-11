"""Shared fixtures for tests that exercise source-based scratch training."""

from __future__ import annotations

import pytest


@pytest.fixture
def compatible_pretraining_profile(monkeypatch):
    """Keep API behavior tests independent of the developer's installed wheels."""
    def compatible(*, accelerator, provider=None):
        return {
            "schema_version": 1,
            "scope": "installed_metadata_only",
            "status": "metadata_compatible",
            "accelerator": accelerator,
            "requirements": [],
            "limitations": ["metadata_only", "does_not_certify_accelerator_execution"],
        }

    monkeypatch.setattr(
        "trainlm.runtime.pretraining_profile.inspect_pretraining_profile",
        compatible,
    )
    return compatible
