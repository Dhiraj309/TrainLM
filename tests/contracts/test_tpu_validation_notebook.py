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


def test_tpu_notebook_demonstrates_the_public_hf_like_workflow():
    source = _source()

    for required in (
        "TrainLMTrainer.from_pretrained",
        "TrainLMTrainingArguments",
        "PackedBinDataset.from_directory",
        "result = trainer.train()",
        'trainer.explain(format="text")',
        "eval_steps=2",
        "save_steps=2",
        "resume_from_checkpoint=OUTPUT_DIR",
    ):
        assert required in source


def test_tpu_notebook_hides_orchestration_and_topology_inputs():
    source = _source()

    for private_detail in (
        "trainlm_tpu_worker.py",
        "_TPUCoordinator",
        "_TPURunRequest",
        "expected_world_size",
        "PJRT_",
        "torch_xla.launch",
        "subprocess",
        "--expected-world-size",
    ):
        assert private_detail not in source
