import pytest

from trainlm.checkpoint import (
    ArtifactRecord,
    AtomicCommit,
    DataCursor,
    LayoutState,
    MeshAxis,
    ResumeManifest,
    ResumeTopology,
    StateDescriptor,
    TrainingProgress,
)


def artifact(artifact_id, role, path, *, rank=None, shard=None):
    shard_fields = {}
    if shard is not None:
        shard_fields = {
            "shard_group": role,
            "shard_index": shard,
            "shard_count": 2,
        }
    return ArtifactRecord(
        artifact_id=artifact_id,
        role=role,
        path=path,
        format="safetensors" if role == "model" else "torch_state",
        sha256="a" * 64,
        size_bytes=128,
        rank=rank,
        **shard_fields,
    )


def resume_manifest(*, status="complete", artifacts=None, cursors=None):
    artifacts = artifacts if artifacts is not None else (
        artifact("model-0", "model", "model/rank-0.safetensors", shard=0),
        artifact("model-1", "model", "model/rank-1.safetensors", shard=1),
        artifact("optimizer-0", "optimizer", "optimizer/rank-0.pt", shard=0),
        artifact("optimizer-1", "optimizer", "optimizer/rank-1.pt", shard=1),
        artifact("scheduler", "scheduler", "scheduler/state.json"),
        artifact("trainer", "trainer", "trainer/state.json"),
        artifact("runtime", "runtime", "runtime/state.json"),
        artifact("rng-0", "rng", "rng/rank-0.pt", rank=0),
        artifact("rng-1", "rng", "rng/rank-1.pt", rank=1),
        artifact("data-0", "data", "data/rank-0-worker-0.json", rank=0),
        artifact("data-1", "data", "data/rank-1-worker-0.json", rank=1),
    )
    cursors = cursors if cursors is not None else (
        DataCursor(0, 0, "dataset-sha", "data-0", 0, 2, 50, 128, 1, 1000),
        DataCursor(1, 0, "dataset-sha", "data-1", 0, 3, 10, 64, 1, 1000),
    )
    return ResumeManifest(
        schema_version=1,
        checkpoint_id="step-100",
        created_at="2026-08-22T12:00:00Z",
        framework_version="0.1.0",
        framework_revision="revision-sha",
        progress=TrainingProgress(100, 3200, 1, 104857600, 2000),
        topology=ResumeTopology(
            backend="pytorch-xla",
            world_size=2,
            precision="bf16",
            state_layout="sharded",
            mesh=(MeshAxis("data", 2),),
        ),
        layout=LayoutState("b" * 64, "plan-001", "packed-qkv", ("pack-qkv",)),
        states=(
            StateDescriptor(
                "model", "LlamaForCausalLM", 1, "sharded",
                "canonical_parameter_name", ("model-0", "model-1"),
            ),
            StateDescriptor(
                "optimizer", "AdamW", 1, "sharded",
                "canonical_parameter_name", ("optimizer-0", "optimizer-1"),
            ),
            StateDescriptor(
                "scheduler", "WSD", 1, "replicated",
                "state_dict_key", ("scheduler",),
            ),
            StateDescriptor(
                "trainer", "TrainLMTrainer", 1, "replicated",
                "field_name", ("trainer",),
            ),
            StateDescriptor(
                "runtime", "pytorch-xla", 1, "replicated",
                "field_name", ("runtime",),
            ),
            StateDescriptor(
                "rng", "torch+xla", 1, "per_rank", "rank",
                ("rng-0", "rng-1"),
            ),
            StateDescriptor(
                "data", "native_memmap", 1, "per_worker",
                "replica_worker", ("data-0", "data-1"),
            ),
        ),
        data_cursors=cursors,
        artifacts=artifacts,
        commit=AtomicCommit(status, "transaction-001", "directory_rename"),
    )


def test_complete_resume_manifest_round_trips_and_is_resumable():
    manifest = resume_manifest()
    restored = ResumeManifest.from_json(manifest.to_json())

    restored.assert_resumable()
    assert restored == manifest
    assert restored.progress.tokens_seen == 104857600
    assert restored.topology.mesh[0].name == "data"
    assert restored.layout.applied_transform_ids == ("pack-qkv",)


def test_incomplete_transaction_is_serializable_but_never_resumable():
    manifest = resume_manifest(status="staging", artifacts=(), cursors=())

    assert manifest.is_resumable is False
    with pytest.raises(RuntimeError, match="incomplete"):
        manifest.assert_resumable()


def test_complete_resume_requires_every_state_role_and_exact_cursor():
    without_rng = tuple(
        item for item in resume_manifest().artifacts if item.role != "rng"
    )
    with pytest.raises(ValueError, match="rng"):
        resume_manifest(artifacts=without_rng)

    inexact = (
        DataCursor(0, 0, "dataset-sha", "data-0", 0, 2, 50, 128, 1, 1000),
        DataCursor(
            1, 0, "dataset-sha", "data-1", 0, 3, 10, 64, 1, 1000, exact=False
        ),
    )
    with pytest.raises(ValueError, match="exact data cursors"):
        resume_manifest(cursors=inexact)


def test_complete_resume_rejects_incomplete_shard_group():
    incomplete = tuple(
        item for item in resume_manifest().artifacts if item.artifact_id != "model-1"
    )
    with pytest.raises(ValueError, match="complete and contiguous"):
        resume_manifest(artifacts=incomplete)


def test_resume_topology_requires_a_supported_layout_and_explicit_mesh():
    with pytest.raises(ValueError, match="state_layout"):
        ResumeTopology("pytorch-xla", 1, "bf16", "unknown", (MeshAxis("data", 1),))

    with pytest.raises(ValueError, match="MeshAxis"):
        ResumeTopology("pytorch-xla", 1, "bf16", "canonical", ())


def test_artifact_integrity_fields_are_strictly_typed():
    with pytest.raises(ValueError, match="sha256"):
        ArtifactRecord("model", "model", "model.safetensors", "safetensors", 1, 1)

    with pytest.raises(ValueError, match="required"):
        ArtifactRecord(
            "model", "model", "model.safetensors", "safetensors",
            "a" * 64, 1, required=1,
        )
