import pytest

from trainlm.checkpoint import plan_canonical_hf_export

from .test_export_contract import artifact, export_manifest


def test_canonical_export_plan_requires_reversed_transforms_and_safetensors():
    plan = plan_canonical_hf_export(export_manifest())

    assert plan.export_id == "hf-step-100"
    assert plan.target_parameter_layout == "huggingface"
    assert plan.safe_serialization is True
    assert plan.required_roles == ("config", "model_weights")
    assert plan.forbidden_roles == ()
    assert plan.reversed_transform_ids == ("pack-qkv",)


def test_canonical_export_plan_rejects_incomplete_manifest():
    with pytest.raises(RuntimeError, match="incomplete"):
        plan_canonical_hf_export(export_manifest(status="staging", artifacts=()))


def test_canonical_export_plan_is_guarded_by_manifest_contract():
    with_runtime = export_manifest().artifacts + (
        artifact("optimizer", "optimizer", "optimizer.pt", "torch_state"),
    )
    with pytest.raises(ValueError, match="training-only"):
        plan_canonical_hf_export(export_manifest(artifacts=with_runtime))

    with pytest.raises(ValueError, match="safetensors"):
        plan_canonical_hf_export(
            export_manifest(
                artifacts=(
                    artifact("config", "config", "config.json", "json"),
                    artifact(
                        "weights",
                        "model_weights",
                        "model.bin",
                        "pytorch_pickle",
                    ),
                )
            )
        )


def test_canonical_export_plan_requires_manifest_type():
    with pytest.raises(TypeError, match="HFExportManifest"):
        plan_canonical_hf_export(object())
