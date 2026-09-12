import json
from pathlib import Path

import pytest

from trainlm.release import (
    evaluate_explanation_support,
    load_support_manifest,
)


MANIFEST = Path(__file__).resolve().parents[2] / "support" / "dense_ar_v1.json"


def test_published_support_manifest_is_valid_and_honest():
    manifest = load_support_manifest(MANIFEST)

    assert manifest.schema_version == 1
    assert manifest.torchtpu == {"status": "deferred", "milestone": "M15"}
    assert {item["backend"] for item in manifest.hardware} == {
        "cpu",
        "cuda",
        "xla",
    }
    assert all(
        item["support_level"] != "certified" for item in manifest.providers
    )


def test_manifest_agrees_with_local_and_tpu_explanations():
    manifest = load_support_manifest(MANIFEST)
    local = {
        "backend": "cpu",
        "selected_path": "huggingface_model",
        "certification": "compatible",
    }
    tpu = {
        "backend": "xla",
        "selected_path": "tpu_coordinator",
        "certification": "unverified",
    }

    assert evaluate_explanation_support(manifest, local).agrees
    assert evaluate_explanation_support(manifest, tpu).agrees


def test_manifest_rejects_overclaim_and_unknown_execution_path():
    manifest = load_support_manifest(MANIFEST)
    result = evaluate_explanation_support(
        manifest,
        {
            "backend": "xla",
            "selected_path": "unpublished_kernel",
            "certification": "certified",
        },
    )

    assert not result.agrees
    assert any("absent" in reason for reason in result.reasons)
    assert any("exceeds hardware" in reason for reason in result.reasons)


def test_loader_rejects_top_level_and_nested_schema_drift(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "support.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest keys"):
        load_support_manifest(path)

    payload.pop("unexpected")
    payload["hardware"][0]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"hardware\[0\] keys"):
        load_support_manifest(path)


def test_loader_rejects_malformed_json_and_wrong_container_types(tmp_path):
    path = tmp_path / "support.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid support manifest"):
        load_support_manifest(path)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["providers"] = "portable_chunked_linear_ce"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="providers must be a JSON array"):
        load_support_manifest(path)
