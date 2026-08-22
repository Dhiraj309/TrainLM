import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "manifests"
    / "laughlm_135m_v5e8_v1.json"
)
LOCK_PATH = MANIFEST_PATH.with_suffix(".lock.json")

KNOWN_LOCKS = {
    ("laughlm-135m-v5e8", 1): (
        "a7a78b4b3fd2da14b4314944d67e8a9576237769a557cc0805e98c71697bbfe1"
    ),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_matches_immutable_version_lock():
    manifest = _read_json(MANIFEST_PATH)
    lock = _read_json(LOCK_PATH)
    identity = (manifest["manifest_id"], manifest["manifest_version"])
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()

    assert lock["immutable"] is True
    assert lock["manifest_id"] == identity[0]
    assert lock["manifest_version"] == identity[1]
    assert lock["manifest_path"] == MANIFEST_PATH.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    assert digest == lock["sha256"]
    assert digest == KNOWN_LOCKS[identity]

    decision_path = REPOSITORY_ROOT / lock["decision_record"]
    assert decision_path.is_file()
    decision = decision_path.read_text(encoding="utf-8")
    assert f"Manifest version:** {identity[1]}" in decision
    assert digest in decision


def test_manifest_locks_required_parity_sections():
    manifest = _read_json(MANIFEST_PATH)
    required_sections = {
        "reference",
        "model",
        "architecture",
        "initialization",
        "loss",
        "optimizer",
        "scheduler",
        "batch",
        "precision",
        "parallelism",
        "runtime",
        "data",
        "reference_metrics",
        "acceptance_thresholds",
    }

    assert required_sections <= manifest.keys()
    assert manifest["status"] == "locked_reference"
    assert manifest["reference"]["revision"] == "0705d255faab"
    assert manifest["runtime"]["accelerator_type"] == "v5e-8"


def test_locked_model_geometry_has_expected_parameter_count():
    manifest = _read_json(MANIFEST_PATH)
    model = manifest["model"]
    architecture = manifest["architecture"]

    hidden_size = model["hidden_size"]
    attention_heads = model["num_attention_heads"]
    key_value_heads = model["num_key_value_heads"]
    head_size, remainder = divmod(hidden_size, attention_heads)
    assert remainder == 0

    query_width = attention_heads * head_size
    key_value_width = key_value_heads * head_size
    embedding_parameters = model["vocab_size"] * hidden_size
    attention_parameters = hidden_size * (
        query_width + key_value_width + key_value_width + hidden_size
    )
    mlp_parameters = 3 * hidden_size * model["intermediate_size"]
    block_norm_parameters = 2 * hidden_size
    final_norm_parameters = hidden_size
    output_head_parameters = (
        0
        if architecture["tie_word_embeddings"]
        else model["vocab_size"] * hidden_size
    )
    calculated_parameters = (
        embedding_parameters
        + model["num_hidden_layers"]
        * (
            attention_parameters
            + mlp_parameters
            + block_norm_parameters
        )
        + final_norm_parameters
        + output_head_parameters
    )

    assert architecture["use_bias"] is False
    assert architecture["attention_variant"] == "mha"
    assert attention_heads == key_value_heads
    assert calculated_parameters == 135_611_392
    assert calculated_parameters == model["expected_parameter_count"]


def test_locked_batch_has_expected_tokens_per_optimizer_update():
    batch = _read_json(MANIFEST_PATH)["batch"]
    calculated_tokens = (
        batch["sequence_length"]
        * batch["micro_batch_per_device"]
        * batch["gradient_accumulation_steps"]
        * batch["data_parallel_replicas"]
    )

    assert calculated_tokens == 1_048_576
    assert calculated_tokens == batch["expected_tokens_per_optimizer_update"]


def test_locked_reference_and_release_thresholds_are_consistent():
    manifest = _read_json(MANIFEST_PATH)
    reference = manifest["reference_metrics"]
    thresholds = manifest["acceptance_thresholds"]

    assert reference["global_tokens_per_second"] == 1_014_000
    assert reference["non_embedding_mfu"] == 0.531
    assert thresholds["hard_90_percent"][
        "minimum_global_tokens_per_second"
    ] == 912_600
    assert thresholds["preferred_95_percent"][
        "minimum_global_tokens_per_second"
    ] == 963_300
    assert thresholds["hard_90_percent"]["minimum_non_embedding_mfu"] == 0.478
    assert thresholds["preferred_95_percent"][
        "minimum_non_embedding_mfu"
    ] == 0.504
