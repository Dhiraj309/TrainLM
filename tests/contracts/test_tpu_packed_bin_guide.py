from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "tutorials" / "TPU_PACKED_BIN_PRETRAINING.md"


def test_guide_covers_secure_public_workflow_and_current_export_boundary():
    text = GUIDE.read_text(encoding="utf-8")
    for required in (
        "HF_TOKEN",
        "immutable commit",
        "shard_range=(0, 8)",
        'split="train"',
        'split="validation"',
        "trainer.explain",
        "resume_from_checkpoint",
        "trainer.save_model",
        "NotImplementedError",
    ):
        assert required in text
    assert "from trainlm import" in text
    assert "from trainlm." not in text
    assert "subprocess" not in text
    assert "PJRT_" not in text


def test_guide_contains_no_hugging_face_access_token_literal():
    text = GUIDE.read_text(encoding="utf-8")
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    assert token_pattern.search(text) is None
