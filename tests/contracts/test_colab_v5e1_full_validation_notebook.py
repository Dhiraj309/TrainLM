import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/TrainLM_Colab_v5e1_Full_Validation.ipynb")


def test_colab_v5e1_full_validation_notebook_is_runnable_and_complete():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")

    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for required in (
        "TrainLM_Colab_v5e1_Full_Validation.ipynb",
        "ct5lp-hightpu-1t",
        "--no-deps",
        "RUN_T1_COMPILE",
        "RUN_T1_LIFECYCLE",
        "RUN_GPT2_HF_MATRIX",
        "RUN_LLAMA_HF_MATRIX",
        "RUN_CLI_WORKFLOWS",
        "PRECISION_MODE",
        "TEST_PRETRAIN_OPTIONS",
        "TEST_PRETRAIN_COMPILE",
        "TEST_VM_RECOVERY",
        "TEST_FINETUNING_OPTIONS",
        "TEST_FINETUNING_STREAMING",
        "TEST_EPOCH_AND_CHUNKED_LOSS",
        "HUB_REPO_ID",
        "examples/torchtpu_t1_v5e1_probe.py",
        "examples/torchtpu_t1_lifecycle_probe.py",
        "examples/validate_hf_pretraining.py",
        "training_plan(inspect_model=True)",
        "include_resume_state=True",
        "restored_output_dir=",
        'loss_implementation="chunked_linear"',
        "LinearCausalLMTrainingView",
        "restore_finetuning_config",
        "streaming=True",
        'report_to="jsonl"',
        "trainlm_v5e1_full_validation_report.json",
    ):
        assert required in source, f"Missing {required!r} in {NOTEBOOK}"
