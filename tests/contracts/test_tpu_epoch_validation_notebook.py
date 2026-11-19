import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/TrainLM_TPU_Epoch_Eval_Save_V5e1.ipynb")


def test_v5e1_epoch_validation_notebook_is_runnable_and_scoped():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for required in (
        "TrainLM_TPU_Epoch_Eval_Save_V5e1.ipynb",
        "ct5lp-hightpu-1t",
        "accelerator=BACKEND",
        'BACKEND = "auto"',
        'BACKEND == "native_tpu"',
        'BACKEND == "tpu"',
        'eval_strategy="epoch"',
        'save_strategy="epoch"',
        '"evaluation-only", eval_strategy="epoch", save_strategy="no"',
        '"checkpoint-only", eval_strategy="no", save_strategy="epoch"',
        "tests/test_api_tpu_coordinator.py",
        "tests/training/test_trainer.py",
        "checkpoint-3",
        "evaluation_completed",
        "legacy_xla_path_hardware_validated",
        '"performance_certified": False',
    ):
        assert required in source
    assert "pip install" not in source
    assert "torch_tpu" in source
