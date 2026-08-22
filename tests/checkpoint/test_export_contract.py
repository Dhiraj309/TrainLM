import pytest

from trainlm.checkpoint import (
    ArtifactRecord,
    AtomicCommit,
    ExportLayout,
    HFExportManifest,
    TiedWeight,
)


def artifact(artifact_id, role, path, format):
    return ArtifactRecord(
        artifact_id=artifact_id,
        role=role,
        path=path,
        format=format,
        sha256="c" * 64,
        size_bytes=256,
    )


def export_manifest(*, status="complete", artifacts=None, layout=None):
    return HFExportManifest(
        schema_version=1,
        export_id="hf-step-100",
        created_at="2026-08-22T12:00:00Z",
        model_type="llama",
        architecture="LlamaForCausalLM",
        config_class="LlamaConfig",
        transformers_version="5.15.0",
        dtype="bfloat16",
        safe_serialization=True,
        source_checkpoint_id="step-100",
        layout=layout or ExportLayout(
            source_parameter_layout="packed-qkv",
            target_parameter_layout="huggingface",
            reversed_transform_ids=("pack-qkv",),
        ),
        tied_weights=(TiedWeight("model.embed_tokens.weight", "lm_head.weight"),),
        artifacts=artifacts if artifacts is not None else (
            artifact("config", "config", "config.json", "json"),
            artifact("weights", "model_weights", "model.safetensors", "safetensors"),
            artifact("index", "weight_index", "model.safetensors.index.json", "json"),
        ),
        commit=AtomicCommit(status, "export-transaction", "directory_rename"),
    )


def test_complete_hf_export_round_trips_and_is_loadable():
    manifest = export_manifest()
    restored = HFExportManifest.from_json(manifest.to_json())

    restored.assert_loadable()
    assert restored == manifest
    assert restored.layout.target_parameter_layout == "huggingface"
    assert restored.tied_weights[0].alias_name == "lm_head.weight"


def test_incomplete_export_is_ignored():
    manifest = export_manifest(status="failed", artifacts=())

    assert manifest.is_loadable is False
    with pytest.raises(RuntimeError, match="incomplete"):
        manifest.assert_loadable()


def test_complete_export_rejects_runtime_state_and_unreversed_transforms():
    with_runtime = export_manifest().artifacts + (
        artifact("optimizer", "optimizer", "optimizer.pt", "torch_state"),
    )
    with pytest.raises(ValueError, match="training-only"):
        export_manifest(artifacts=with_runtime)

    layout = ExportLayout(
        source_parameter_layout="packed-qkv",
        target_parameter_layout="huggingface",
        remaining_transform_ids=("pack-qkv",),
    )
    with pytest.raises(ValueError, match="cannot retain"):
        export_manifest(layout=layout)


def test_complete_export_requires_config_and_safetensors_weights():
    weights_only = (
        artifact("weights", "model_weights", "model.safetensors", "safetensors"),
    )
    with pytest.raises(ValueError, match="config"):
        export_manifest(artifacts=weights_only)

    pickle_weights = (
        artifact("config", "config", "config.json", "json"),
        artifact("weights", "model_weights", "pytorch_model.bin", "pytorch_pickle"),
    )
    with pytest.raises(ValueError, match="safetensors"):
        export_manifest(artifacts=pickle_weights)
