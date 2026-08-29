import pytest

from trainlm.runtime import (
    AccumulationEvidence,
    select_accumulation_plan,
)


def _plan(**kwargs):
    values = {
        "micro_batch": 2,
        "sequence_length": 2048,
        "accumulation_steps": 32,
        "evidence": AccumulationEvidence(backend="pytorch-xla"),
    }
    values.update(kwargs)
    return select_accumulation_plan(**values)


def test_auto_uses_safe_microstep_without_complete_evidence():
    plan = _plan()

    assert plan.selected == "microstep"
    assert plan.fallback_from is None
    assert "evidence" in plan.reason


def test_auto_selects_unrolled_only_with_compile_dispatch_and_hbm_evidence():
    plan = _plan(
        evidence=AccumulationEvidence(
            backend="pytorch-xla",
            compile_supported=True,
            dispatch_supported=True,
            hbm_headroom=True,
        )
    )

    assert plan.selected == "unrolled"
    assert plan.micro_batch == 2
    assert plan.sequence_length == 2048
    assert plan.accumulation_steps == 32


def test_explicit_unsupported_strategy_records_safe_fallback():
    plan = _plan(requested="xla_loop")

    assert plan.selected == "microstep"
    assert plan.fallback_from == "xla_loop"


def test_required_strategy_rejects_missing_evidence():
    with pytest.raises(RuntimeError, match="fallbacks are disabled"):
        _plan(requested="unrolled", allow_fallbacks=False)
