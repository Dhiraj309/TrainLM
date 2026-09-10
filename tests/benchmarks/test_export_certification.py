import pytest

from trainlm.benchmark import PlainHFExportEvidence, evaluate_plain_hf_export


def evidence(**values):
    defaults = dict(
        transformers_version="4.56.0",
        trainlm_installed=False,
        canonical_state_dict_only=True,
        tied_aliases_preserved=True,
        logits_max_abs_error=1e-6,
        loss_abs_error=1e-7,
    )
    return PlainHFExportEvidence(**{**defaults, **values})


def test_clean_transformers_only_reload_passes():
    assert evaluate_plain_hf_export(evidence()).passed


def test_all_interoperability_failures_are_reported():
    result = evaluate_plain_hf_export(
        evidence(
            trainlm_installed=True,
            canonical_state_dict_only=False,
            tied_aliases_preserved=False,
            missing_keys=("model.embed_tokens.weight",),
            unexpected_keys=("model.packed_qkv.weight",),
            logits_max_abs_error=1e-3,
            loss_abs_error=1e-3,
        )
    )
    assert not result.passed
    assert len(result.reasons) == 7


def test_boundary_errors_are_accepted():
    result = evaluate_plain_hf_export(
        evidence(logits_max_abs_error=1e-5, loss_abs_error=1e-5),
        tolerance=1e-5,
    )
    assert result.passed


def test_non_finite_error_is_rejected():
    with pytest.raises(ValueError, match="logits_max_abs_error"):
        evidence(logits_max_abs_error=float("nan"))
