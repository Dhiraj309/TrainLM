import pytest

from trainlm.benchmark import (
    RealShardStabilityEvidence,
    evaluate_real_shard_stability,
)


def evidence(**values):
    defaults = dict(
        completed_updates=200,
        real_data_revision="dataset@abc123",
        shard_count=4,
        evaluation_completed=True,
        checkpoint_resumed=True,
        integrity_checks_passed=True,
        data_cursor_continuous=True,
        export_completed=True,
        unexpected_compile_count=0,
        cpu_fallback_count=0,
        minimum_loss=1.0,
        maximum_loss=10.0,
        minimum_gradient_norm=0.1,
        maximum_gradient_norm=5.0,
    )
    return RealShardStabilityEvidence(**{**defaults, **values})


def test_complete_200_update_run_passes():
    assert evaluate_real_shard_stability(evidence()).passed


def test_all_lifecycle_and_integrity_failures_are_reported():
    result = evaluate_real_shard_stability(
        evidence(
            completed_updates=199,
            shard_count=1,
            evaluation_completed=False,
            checkpoint_resumed=False,
            integrity_checks_passed=False,
            data_cursor_continuous=False,
            export_completed=False,
            unexpected_compile_count=1,
            cpu_fallback_count=1,
        )
    )
    assert not result.passed
    assert len(result.reasons) == 9


def test_non_finite_loss_is_rejected():
    with pytest.raises(ValueError, match="minimum_loss"):
        evidence(minimum_loss=float("nan"))


def test_loss_and_gradient_ranges_must_be_ordered():
    with pytest.raises(ValueError, match="minimum_loss"):
        evidence(minimum_loss=2.0, maximum_loss=1.0)
    with pytest.raises(ValueError, match="minimum_gradient_norm"):
        evidence(minimum_gradient_norm=2.0, maximum_gradient_norm=1.0)
