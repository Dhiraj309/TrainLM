import pytest
import torch

from trainlm.checkpoint import evaluate_round_trip


def _report():
    state = {"layer.weight": torch.tensor([1.0, 2.0])}
    return evaluate_round_trip(
        reference_state=state,
        resumed_state={"layer.weight": torch.tensor([1.0, 2.0 + 1e-7])},
        exported_state={"layer.weight": torch.tensor([1.0, 2.0 - 1e-7])},
        reference_next_update=torch.tensor([0.2, 0.3]),
        resumed_next_update=torch.tensor([0.2, 0.3 + 1e-7]),
        reference_logits=torch.tensor([[0.1, 0.2]]),
        exported_logits=torch.tensor([[0.1, 0.2 - 1e-7]]),
    )


def test_round_trip_report_accepts_equivalent_resume_and_export():
    report = _report()

    assert report.passed
    assert report.missing_state_keys == ()
    assert report.unexpected_state_keys == ()


def test_round_trip_report_rejects_missing_or_numerically_different_state():
    state = {"layer.weight": torch.tensor([1.0])}
    report = evaluate_round_trip(
        reference_state=state,
        resumed_state={},
        exported_state={"other.weight": torch.tensor([1.0])},
        reference_next_update=torch.tensor([0.0]),
        resumed_next_update=torch.tensor([0.0]),
        reference_logits=torch.tensor([0.0]),
        exported_logits=torch.tensor([0.0]),
    )

    assert not report.passed
    assert report.missing_state_keys == ("layer.weight",)
    assert report.unexpected_state_keys == ("other.weight",)


def test_round_trip_report_rejects_shape_mismatch():
    state = {"layer.weight": torch.tensor([1.0, 2.0])}
    report = evaluate_round_trip(
        reference_state=state,
        resumed_state={"layer.weight": torch.tensor([1.0])},
        exported_state=state,
        reference_next_update=torch.tensor([0.0]),
        resumed_next_update=torch.tensor([0.0]),
        reference_logits=torch.tensor([0.0]),
        exported_logits=torch.tensor([0.0]),
    )

    assert not report.passed


def test_round_trip_inputs_are_validated():
    with pytest.raises(TypeError, match="reference_state"):
        evaluate_round_trip(
            reference_state=object(),
            resumed_state={},
            exported_state={},
            reference_next_update=torch.tensor([0.0]),
            resumed_next_update=torch.tensor([0.0]),
            reference_logits=torch.tensor([0.0]),
            exported_logits=torch.tensor([0.0]),
        )

    with pytest.raises(ValueError, match="tolerance"):
        evaluate_round_trip(
            reference_state={},
            resumed_state={},
            exported_state={},
            reference_next_update=torch.tensor([0.0]),
            resumed_next_update=torch.tensor([0.0]),
            reference_logits=torch.tensor([0.0]),
            exported_logits=torch.tensor([0.0]),
            tolerance=0,
        )
