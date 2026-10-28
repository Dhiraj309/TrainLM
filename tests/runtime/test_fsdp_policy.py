import json

import pytest

from trainlm.runtime import FSDPMeshPolicy, ParameterShardingRule


def policy():
    return FSDPMeshPolicy(
        data_replicas=4,
        fsdp_shards=2,
        wrap_module_classes=("DecoderLayer",),
        parameter_rules=(
            ParameterShardingRule(".weight", ("fsdp", None)),
            ParameterShardingRule(".bias", ("fsdp",)),
        ),
    )


def test_policy_builds_data4_fsdp2_mesh_and_checkpoint_contract():
    selected = policy()
    mesh = selected.logical_mesh()

    assert mesh.axis_sizes == {"data": 4, "fsdp": 2}
    assert mesh.size == selected.world_size == 8
    assert selected.shard_optimizer_state
    assert selected.checkpoint.layout == "distributed_rank_shards"
    assert selected.checkpoint.require_topology_match


def test_policy_requires_rematerialization_before_wrapping():
    selected = policy()
    selected.validate_application_order(
        world_size=8,
        rematerialization_applied=True,
        model_already_wrapped=False,
    )
    with pytest.raises(ValueError, match="Rematerialization"):
        selected.validate_application_order(
            world_size=8,
            rematerialization_applied=False,
            model_already_wrapped=False,
        )
    with pytest.raises(ValueError, match="before model wrapping"):
        selected.validate_application_order(
            world_size=8,
            rematerialization_applied=True,
            model_already_wrapped=True,
        )


def test_policy_rejects_topology_and_ambiguous_parameter_rules():
    with pytest.raises(ValueError, match="world_size=8"):
        policy().validate_application_order(
            world_size=4,
            rematerialization_applied=True,
            model_already_wrapped=False,
        )
    with pytest.raises(ValueError, match="must use the FSDP axis"):
        ParameterShardingRule(".weight", (None, None))
    with pytest.raises(ValueError, match="suffixes must be unique"):
        FSDPMeshPolicy(
            data_replicas=4,
            fsdp_shards=2,
            wrap_module_classes=("DecoderLayer",),
            parameter_rules=(
                ParameterShardingRule(".weight", ("fsdp", None)),
                ParameterShardingRule(".weight", (None, "fsdp")),
            ),
        )


def test_policy_manifest_round_trip_preserves_nested_contracts():
    selected = policy()

    restored = FSDPMeshPolicy.from_manifest(selected.to_manifest())

    assert restored == selected
    assert restored.logical_mesh().axis_sizes == {"data": 4, "fsdp": 2}


def test_policy_manifest_rejects_schema_drift_and_invalid_arrays():
    manifest = json.loads(policy().to_manifest())
    manifest["policy"]["unexpected"] = True
    with pytest.raises(ValueError, match="policy keys"):
        FSDPMeshPolicy.from_manifest(manifest)

    manifest = json.loads(policy().to_manifest())
    manifest["policy"]["parameter_rules"][0]["partition_spec"] = "fsdp"
    with pytest.raises(ValueError, match="partition_spec must be an array"):
        FSDPMeshPolicy.from_manifest(manifest)


def test_policy_manifest_rejects_malformed_json_and_boolean_version():
    with pytest.raises(ValueError, match="Invalid FSDP policy manifest"):
        FSDPMeshPolicy.from_manifest("not-json")

    manifest = json.loads(policy().to_manifest())
    manifest["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version=1"):
        FSDPMeshPolicy.from_manifest(manifest)
