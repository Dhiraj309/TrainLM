import json
from pathlib import Path

import pytest

from trainlm.benchmark import calculate_causal_lm_flops, calculate_mfu


REPOSITORY_ROOT = Path(__file__).parents[2]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "manifests"
    / "laughlm_135m_v5e8_v1.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_laughlm_flop_breakdown_matches_locked_geometry():
    manifest = _manifest()
    model = manifest["model"]
    architecture = manifest["architecture"]

    flops = calculate_causal_lm_flops(
        parameter_count=model["expected_parameter_count"],
        vocabulary_size=model["vocab_size"],
        hidden_size=model["hidden_size"],
        num_hidden_layers=model["num_hidden_layers"],
        sequence_length=model["max_sequence_length"],
        tie_word_embeddings=architecture["tie_word_embeddings"],
    )

    assert flops.non_embedding_flops_per_token == 817_993_728
    assert flops.output_projection_flops_per_token == 197_001_216
    assert flops.logits_inclusive_flops_per_token == 1_014_994_944


def test_laughlm_mfu_is_reproduced_from_locked_manifest():
    manifest = _manifest()
    model = manifest["model"]
    architecture = manifest["architecture"]
    runtime = manifest["runtime"]
    metrics = manifest["reference_metrics"]
    flops = calculate_causal_lm_flops(
        parameter_count=model["expected_parameter_count"],
        vocabulary_size=model["vocab_size"],
        hidden_size=model["hidden_size"],
        num_hidden_layers=model["num_hidden_layers"],
        sequence_length=model["max_sequence_length"],
        tie_word_embeddings=architecture["tie_word_embeddings"],
    )

    non_embedding_mfu = calculate_mfu(
        tokens_per_second=metrics["device_tokens_per_second"],
        flops_per_token=flops.non_embedding_flops_per_token,
        peak_flops_per_device=197_000_000_000_000,
        device_count=runtime["device_count"],
    )
    logits_inclusive_mfu = calculate_mfu(
        tokens_per_second=metrics["device_tokens_per_second"],
        flops_per_token=flops.logits_inclusive_flops_per_token,
        peak_flops_per_device=197_000_000_000_000,
        device_count=runtime["device_count"],
    )

    assert non_embedding_mfu == pytest.approx(0.530969279)
    assert logits_inclusive_mfu == pytest.approx(0.658845068)
    assert round(non_embedding_mfu, 3) == metrics["non_embedding_mfu"]
    assert round(logits_inclusive_mfu, 3) == metrics["logits_inclusive_mfu"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"tokens_per_second": 0.0},
        {"flops_per_token": -1.0},
        {"peak_flops_per_device": float("inf")},
        {"device_count": 0},
    ],
)
def test_mfu_rejects_invalid_inputs(arguments):
    values = {
        "tokens_per_second": 1.0,
        "flops_per_token": 1.0,
        "peak_flops_per_device": 1.0,
        "device_count": 1,
    }
    values.update(arguments)

    with pytest.raises(ValueError):
        calculate_mfu(**values)
