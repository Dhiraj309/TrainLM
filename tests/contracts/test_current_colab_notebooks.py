import json
from pathlib import Path
import re


ACTIVE_NOTEBOOKS = (
    Path("notebooks/TorchTPU_T1_v5e1_Probe.ipynb"),
    Path("notebooks/TorchTPU_T1_v5e8_Distributed_Probe.ipynb"),
    Path("notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb"),
    Path("notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb"),
    Path("notebooks/TrainLM_Text_Fine_Tuning.ipynb"),
)
STALE_BRANCHES = (
    "docs/01-t1-v5e1-probe",
    "feat/01-native-torchtpu-single-device-training",
    "feat/02-packed-pretraining-data",
)


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in cells
        if cell["cell_type"] == "code"
    )
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_active_colab_notebooks_clone_merged_main():
    for path in ACTIVE_NOTEBOOKS:
        source = _notebook_source(path)
        if path.name.startswith("TorchTPU_T1_"):
            assert 'PROBE_BRANCH = "main"' in source, path
        elif path.name == "TrainLM_Text_Fine_Tuning.ipynb":
            assert 'TRAINLM_REF = "feat/01-pretrained-finetuning-api"' in source, path
        else:
            assert re.search(r'"--branch",\s*"main"', source), path
        for stale_branch in STALE_BRANCHES:
            assert stale_branch not in source, (path, stale_branch)


def test_probe_notebook_badges_open_main_versions():
    for path in ACTIVE_NOTEBOOKS[:2]:
        source = _notebook_source(path)
        assert "colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/" in source


def test_from_scratch_notebook_installs_only_missing_pretraining_extras():
    notebook = json.loads(
        Path("notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb").read_text(
            encoding="utf-8"
        )
    )
    setup = "".join(notebook["cells"][1]["source"])

    compile(setup, "TrainLM_Native_TorchTPU_From_Scratch.ipynb:setup", "exec")
    assert 'Requirement("datasets>=3,<5")' in setup
    assert 'Requirement("tokenizers>=0.21,<1")' in setup
    assert "missing_pretraining_requirements" in setup
    assert "incompatible_pretraining_requirements" in setup
    assert '"pip", "install"' in setup
    assert '"--quiet", "--disable-pip-version-check"' in setup
    assert "the setup cell leaves installed versions unchanged" in setup
    assert "trainlm[dev]" not in setup


def test_text_finetuning_notebook_is_copyable_and_uses_the_public_api():
    path = Path("notebooks/TrainLM_Text_Fine_Tuning.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(
        cell["execution_count"] is None and not cell["outputs"]
        for cell in code_cells
    )
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")

    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'TRAINLM_REF = "feat/01-pretrained-finetuning-api"' in source
    assert "--no-deps" in source
    assert "run_finetuning(" in source
    assert "restore_finetuning_config(" in source
    assert 'if RESUME_FROM_CHECKPOINT == "latest":' in source
    assert "max_steps=MAX_STEPS" in source
    assert "config.args.max_steps" in source
    assert 'RESUME_FROM_CHECKPOINT = ""' in source
    assert "resume_from_checkpoint=RESUME_FROM_CHECKPOINT or None" in source
    assert "SAVE_STEPS = 100" in source
    assert "save_steps=SAVE_STEPS" in source
    assert "above the saved step" in source
    assert "persistent storage" in source
    assert 'ACCELERATOR = "cpu"' in source
    assert '"native_tpu"' in source
    assert "files.download" in source
    assert "finetuning-summary.json" in source
    assert "json.loads(summary_path.read_text" in source
    assert 'report["status"] == "completed"' in source
    assert "run_status=failed" in source
    assert '"torch"' not in source


def test_from_scratch_notebook_exposes_opt_in_gradient_checkpointing():
    notebook = json.loads(
        Path("notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb").read_text(
            encoding="utf-8"
        )
    )
    settings = "".join(notebook["cells"][5]["source"])
    run = "".join(notebook["cells"][7]["source"])

    assert "GRADIENT_CHECKPOINTING = False" in settings
    assert "gradient_checkpointing=GRADIENT_CHECKPOINTING" in run
    assert "show_progress=True" in run
    assert "lower activation memory" in _notebook_source(
        Path("notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb")
    )
    compile(settings, "from-scratch:advanced-settings", "exec")
    compile(run, "from-scratch:run", "exec")


def test_from_scratch_notebook_labels_smoke_and_starter_budgets():
    notebook = json.loads(
        Path("notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb").read_text(
            encoding="utf-8"
        )
    )
    instructions = "".join(notebook["cells"][2]["source"])
    settings = "".join(notebook["cells"][3]["source"])

    assert 'TRAINING_PRESET = "quick_smoke (10k tokens)"' in settings
    assert '"quick_smoke (10k tokens)": 10_000' in settings
    assert '"starter_pretraining (1M tokens)": 1_000_000' in settings
    assert "CUSTOM_TARGET_SUPERVISED_TOKENS" in settings
    assert "resuming_saved_target_tokens" in settings
    assert "not useful-model training" in instructions
    compile(settings, "from-scratch:run-settings", "exec")


def test_xla_validation_notebook_is_labeled_historical():
    source = _notebook_source(Path("notebooks/TrainLM_TPU_Validation.ipynb"))
    assert "Historical PyTorch/XLA validation notebook" in source
    assert "does not validate the current native TorchTPU path" in source
