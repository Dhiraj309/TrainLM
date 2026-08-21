from pathlib import Path


SCOPE_DOCUMENT = Path(__file__).parents[2] / "docs" / "SCOPE.md"


def _scope_text() -> str:
    return SCOPE_DOCUMENT.read_text(encoding="utf-8")


def test_dense_ar_scope_document_is_present_and_normative():
    text = _scope_text()

    assert "Status:** Normative for dense-AR V1" in text
    assert "Meaning of “any autoregressive model”" in text
    assert "Explicitly outside dense-AR V1" in text
    assert "Changing this contract" in text


def test_scope_defines_distinct_support_levels():
    text = _scope_text()

    for support_level in ("### Compatible", "### Optimized", "### Certified"):
        assert support_level in text

    assert "Generic fallback is a required compatibility mechanism" in text
    assert "Only Certified models may be described" in text


def test_scope_keeps_structurally_distinct_dense_ar_representatives():
    text = _scope_text()

    representative_families = (
        "GPT-2",
        "OPT",
        "GPT-NeoX",
        "BLOOM",
        "Falcon",
        "Phi",
        "Llama",
        "Mistral",
        "Qwen",
        "Gemma",
    )

    for family in representative_families:
        assert family in text


def test_scope_preserves_plain_hf_and_remote_code_contracts():
    text = _scope_text()

    assert "AutoModelForCausalLM" in text
    assert "does not require TrainLM to reload" in text
    assert "trust_remote_code=False" in text
