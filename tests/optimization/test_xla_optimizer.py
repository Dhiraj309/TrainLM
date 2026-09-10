import pytest

from trainlm.optimization import (
    XLAAdamWPolicy,
    XLAOptimizerEvidence,
    evaluate_xla_optimizer_path,
)


def evidence(**values):
    defaults = dict(
        update_matches_reference=True,
        resume_matches_reference=True,
        graph_stable=True,
        cpu_fallback_count=0,
        step_seconds=0.98,
        reference_step_seconds=1.0,
        peak_hbm_gib=8.0,
        reference_peak_hbm_gib=10.0,
    )
    return XLAOptimizerEvidence(**{**defaults, **values})


def test_bf16_fp32_state_path_passes_complete_evidence():
    result = evaluate_xla_optimizer_path(XLAAdamWPolicy(), evidence())
    assert result.passed
    assert result.step_change_fraction == pytest.approx(-0.02)
    assert result.hbm_change_fraction == pytest.approx(-0.2)


def test_failures_are_all_reported():
    result = evaluate_xla_optimizer_path(
        XLAAdamWPolicy(),
        evidence(
            update_matches_reference=False,
            resume_matches_reference=False,
            graph_stable=False,
            cpu_fallback_count=1,
            step_seconds=1.1,
            peak_hbm_gib=10.0,
        ),
    )
    assert not result.passed
    assert len(result.reasons) == 6


def test_second_moment_must_remain_fp32():
    with pytest.raises(ValueError, match="second_moment_dtype"):
        XLAAdamWPolicy(second_moment_dtype="bfloat16")


def test_weight_decay_must_be_decoupled():
    with pytest.raises(ValueError, match="decoupled"):
        XLAAdamWPolicy(decoupled_weight_decay=False)


def test_policy_json_is_stable():
    policy = XLAAdamWPolicy(gradient_clip_norm=1.0, gradient_reduction="mean")
    assert policy.to_json() == policy.to_json()
    assert '"first_moment_dtype": "bfloat16"' in policy.to_json()
