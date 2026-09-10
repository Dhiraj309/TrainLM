import pytest

from trainlm.optimization import (
    RematerializationMeasurement,
    RematerializationPolicy,
    select_rematerialization_policy,
)


def measurement(name, scopes, step, hbm, **values):
    defaults = dict(gradients_match=True, graph_stable=True)
    return RematerializationMeasurement(
        policy=RematerializationPolicy(
            policy_id=name,
            scopes=scopes,
            apply_before_fsdp=scopes != ("none",),
        ),
        step_seconds=step,
        peak_hbm_gib=hbm,
        **{**defaults, **values},
    )


def test_selects_lowest_hbm_policy_inside_slowdown_budget():
    result = select_rematerialization_policy(
        (
            measurement("none", ("none",), 1.0, 12.0),
            measurement("block", ("block",), 1.08, 8.0),
            measurement("attention", ("attention",), 1.04, 9.0),
        ),
        maximum_slowdown=0.10,
    )
    assert result.selected.policy.policy_id == "block"


def test_rejections_explain_parity_graph_and_step_failures():
    result = select_rematerialization_policy(
        (
            measurement("none", ("none",), 1.0, 12.0),
            measurement("bad-gradient", ("mlp",), 1.0, 4.0, gradients_match=False),
            measurement("bad-graph", ("attention",), 1.0, 4.0, graph_stable=False),
            measurement("too-slow", ("block",), 1.2, 4.0),
        )
    )
    assert result.selected.policy.policy_id == "none"
    assert result.rejected == (
        ("bad-gradient", "gradient parity failed"),
        ("bad-graph", "compiled graph is unstable"),
        ("too-slow", "step-time slowdown exceeds budget"),
    )


def test_selection_is_independent_of_measurement_order():
    values = (
        measurement("none", ("none",), 1.0, 12.0),
        measurement("mlp", ("mlp",), 1.05, 8.0),
        measurement("attention", ("attention",), 1.05, 8.0),
    )
    assert select_rematerialization_policy(values) == select_rematerialization_policy(
        tuple(reversed(values))
    )


def test_rematerialization_requires_pre_fsdp_application():
    with pytest.raises(ValueError, match="before FSDP"):
        RematerializationPolicy("block", ("block",), apply_before_fsdp=False)


def test_exactly_one_baseline_is_required():
    with pytest.raises(ValueError, match="Exactly one"):
        select_rematerialization_policy(
            (measurement("block", ("block",), 1.0, 8.0),)
        )
