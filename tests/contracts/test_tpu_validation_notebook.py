import json
from pathlib import Path


NOTEBOOK = Path("notebooks/TrainLM_TPU_Validation.ipynb")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    code_cells = [
        cell for cell in payload["cells"] if cell["cell_type"] == "code"
    ]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_tpu_validation_notebook_uses_only_the_public_training_boundary():
    source = _source()

    for required in (
        "TrainLMTrainer",
        "TrainLMTrainingArguments",
        "PackedBinDataset.from_directory",
        "trainer.explain",
        "python -m trainlm train",
        "--dry-run",
        "save_steps",
        "eval_steps",
        "resume_from_checkpoint",
        "checkpoint-4",
    ):
        assert required in source

    for private_detail in (
        "trainlm_tpu_worker.py",
        "_TPUCoordinator",
        "_TPURunRequest",
        "PJRT_",
        "torch_xla.launch",
        "subprocess",
    ):
        assert private_detail not in source


def test_tpu_validation_notebook_has_explicit_cost_and_evidence_gates():
    source = _source()

    assert 'TRAINLM_RUN_TPU' in source
    assert 'RUN_TPU = False' in source
    assert "steady_global_supervised_tokens_per_second" in source
    assert "performance_certified" in source
    assert "Target-hardware acceptance checklist" in source
