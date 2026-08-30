import pytest
import torch

from trainlm.monitoring import (
    IntegrityPolicy,
    check_training_integrity,
)


def test_integrity_passes_for_finite_state_and_continuous_tokens():
    report = check_training_integrity(
        loss=torch.tensor(1.0),
        gradients=(torch.ones(2),),
        parameters=(torch.zeros(2),),
        update_norm=0.5,
        expected_tokens=128,
        actual_tokens=128,
        policy=IntegrityPolicy(max_update_norm=1.0),
    )

    assert report.passed
    assert report.violations == ()
    assert set(report.checked) == {
        "loss", "gradients", "parameters", "update_norm", "tokens"
    }


def test_integrity_detects_injected_corruption():
    report = check_training_integrity(
        loss=torch.tensor(float("nan")),
        gradients=(torch.tensor([float("inf")]),),
        parameters=(torch.tensor([float("nan")]),),
        update_norm=2.0,
        expected_tokens=128,
        actual_tokens=64,
        expected_cursor="step-4",
        actual_cursor="step-3",
        policy=IntegrityPolicy(max_update_norm=1.0, require_cursor_continuity=True),
    )

    assert not report.passed
    assert len(report.violations) == 6
    assert report.token_delta_valid is False
    assert report.cursor_continuous is False


def test_integrity_checks_can_be_disabled_for_optional_signals():
    report = check_training_integrity(
        loss=torch.tensor(float("nan")),
        gradients=(torch.tensor([float("nan")]),),
        parameters=(torch.tensor([float("nan")]),),
        expected_tokens=1,
        actual_tokens=2,
        policy=IntegrityPolicy(
            check_loss=False,
            check_gradients=False,
            check_parameters=False,
            require_token_delta=False,
        ),
    )

    assert report.passed
    assert report.checked == ("tokens",)


def test_integrity_validates_policy_and_inputs():
    with pytest.raises(ValueError, match="max_update_norm"):
        IntegrityPolicy(max_update_norm=0)
    with pytest.raises(TypeError, match="IntegrityPolicy"):
        check_training_integrity(loss=1.0, policy=object())
