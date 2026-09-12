from pathlib import Path


TRAINING_SCRIPT = Path("scripts/trainlm_tpu_training.py")


def test_prefetch_geometry_uses_the_worker_microbatch_argument():
    source = TRAINING_SCRIPT.read_text(encoding="utf-8")

    assert "args.micro_batch_per_device" in source
    assert "args.per_device_batch_size" not in source
