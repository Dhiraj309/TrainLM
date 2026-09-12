from pathlib import Path


HANDOFF = Path("docs/HANDOFF.md")


def test_handoff_separates_implemented_evidence_pending_and_planned_work():
    text = HANDOFF.read_text(encoding="utf-8")

    for required in (
        "## Status legend",
        "## Product boundary",
        "## Implemented public training and data path",
        "## Implemented runtime, checkpoint, and optimization foundations",
        "## Immediate execution-preserving backlog (priority order)",
        "## Architecture-changing backlog (TrainLM Lab, not optimizer transforms)",
        "## Telemetry and self-tuning backlog",
        "## Next owner-run TPU procedure",
        "| `[x]` |",
        "| `[~]` |",
        "| `[ ]` |",
    ):
        assert required in text


def test_handoff_does_not_claim_unmeasured_tpu_certification():
    text = HANDOFF.read_text(encoding="utf-8")

    assert "| `[ ]` | Performance certification |" in text
    assert "No current result may be marked certified" in text
    assert "do not mark a TPU or" in text
    assert "performance item complete from software tests alone" in text
