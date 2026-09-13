from pathlib import Path


TRAINING_SCRIPT = Path("scripts/trainlm_tpu_training.py")


def test_prefetch_geometry_uses_the_worker_microbatch_argument():
    source = TRAINING_SCRIPT.read_text(encoding="utf-8")

    assert "args.micro_batch_per_device" in source
    assert "args.per_device_batch_size" not in source


def test_evaluation_uses_current_partition_plan_contract():
    source = TRAINING_SCRIPT.read_text(encoding="utf-8")

    assert "eval_partition.batch_indices" in source
    assert "eval_partition.assignments" not in source


def test_model_preflight_uses_a_bounded_diagnostic_shape():
    source = TRAINING_SCRIPT.read_text(encoding="utf-8")

    assert "preflight_sequence_length = min(args.sequence_length, 16)" in source
    assert '"sequence_length": preflight_sequence_length' in source
