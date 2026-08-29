import pytest

from trainlm.checkpoint import plan_distributed_resume

from .test_resume_contract import resume_manifest


def test_distributed_resume_plan_requires_complete_state_ownership():
    plan = plan_distributed_resume(resume_manifest())

    assert plan.backend == "pytorch-xla"
    assert plan.world_size == 2
    assert plan.direct_shard_io is True
    assert plan.state_names == (
        "model",
        "optimizer",
        "scheduler",
        "trainer",
        "runtime",
        "rng",
        "data",
    )


def test_distributed_resume_plan_can_disable_direct_shard_io():
    plan = plan_distributed_resume(resume_manifest(), direct_shard_io=False)

    assert plan.direct_shard_io is False


def test_distributed_resume_plan_rejects_incomplete_transactions():
    with pytest.raises(RuntimeError, match="incomplete"):
        plan_distributed_resume(
            resume_manifest(status="staging", artifacts=(), cursors=())
        )


def test_distributed_resume_plan_rejects_topology_mismatch():
    with pytest.raises(ValueError, match="backend"):
        plan_distributed_resume(resume_manifest(), backend="cpu")
    with pytest.raises(ValueError, match="world_size"):
        plan_distributed_resume(resume_manifest(), world_size=8)


def test_distributed_resume_plan_requires_manifest_type():
    with pytest.raises(TypeError, match="ResumeManifest"):
        plan_distributed_resume(object())
