from pathlib import Path

from trainlm.tasks import LanguageModelTask


def test_trainer_contains_no_causal_lm_batch_semantics():
    repository_root = Path(__file__).parents[2]
    trainer_source = (
        repository_root / "src" / "trainlm" / "training" / "trainer.py"
    ).read_text(encoding="utf-8")

    assert "input_ids" not in trainer_source
    assert "attention_mask" not in trainer_source
    assert "labels" not in trainer_source
    assert "CausalLMTask" not in trainer_source
    assert LanguageModelTask is not None
