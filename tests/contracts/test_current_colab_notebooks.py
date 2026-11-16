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
            assert "git+https://github.com/Dhiraj309/TrainLM.git@main" in source, path
        elif path.name == "TrainLM_Native_TorchTPU_From_Scratch.ipynb":
            assert "git+https://github.com/Dhiraj309/TrainLM.git@main" in source, path
        else:
            assert re.search(r'"--branch",\s*"main"', source), path
        for stale_branch in STALE_BRANCHES:
            assert stale_branch not in source, (path, stale_branch)


def test_probe_notebook_badges_open_main_versions():
    for path in ACTIVE_NOTEBOOKS[:2]:
        source = _notebook_source(path)
        assert "colab.research.google.com/github/Dhiraj309/TrainLM/blob/main/" in source


def _code_cells(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in cells)
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        compile(source, f"{path.name}:cell-{index}", "exec")
        assert len(source.splitlines()) <= 30, f"{path.name}:cell-{index} is too long"
    return notebook, cells


def test_training_notebooks_keep_the_common_path_short_and_copyable():
    for name, api_name in (
        ("TrainLM_Native_TorchTPU_From_Scratch.ipynb", "run_pretraining"),
        ("TrainLM_Text_Fine_Tuning.ipynb", "run_finetuning"),
    ):
        path = Path("notebooks") / name
        notebook, cells = _code_cells(path)
        source = _notebook_source(path)
        assert len(notebook["cells"]) <= 5
        assert api_name in source
        assert "--no-deps" in source
        assert "git+https://github.com/Dhiraj309/TrainLM.git@main" in source
        assert "files.download" in source
        assert "restore_finetuning_config" not in source
        assert "RESUME_FROM_CHECKPOINT" not in source

    scratch = _notebook_source(
        Path("notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb")
    )
    assert 'accelerator="native_tpu"' in scratch
    assert "10,000-token" in scratch
    assert 'model_type="gpt2"' in scratch
    assert "Hugging Face GPT-2" in scratch
    assert "GPT2LMHeadModel" in scratch
    assert "does not load a pretrained GPT-2 checkpoint" in scratch
    finetuning = _notebook_source(Path("notebooks/TrainLM_Text_Fine_Tuning.ipynb"))
    assert 'accelerator="auto"' in finetuning
    assert "Native TorchTPU fine-tuning is not validated yet" in finetuning


def test_python_quickstart_examples_are_small_public_api_examples():
    for name, api_name in (
        ("pretrain_from_scratch.py", "run_pretraining"),
        ("finetune_text.py", "run_finetuning"),
    ):
        path = Path("examples") / name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert len(source.splitlines()) <= 30
        assert api_name in source
        assert "10_000" in source or "max_steps=3" in source
    scratch = Path("examples/pretrain_from_scratch.py").read_text(encoding="utf-8")
    assert 'model_type="gpt2"' in scratch
    assert "randomly initialized hugging face gpt-2" in scratch.lower()


def test_legacy_smollm2_yaml_is_identified_as_xla():
    source = Path("examples/smollm2_135m_tpu_pretraining.yaml").read_text(
        encoding="utf-8"
    )
    assert "Historical PyTorch/XLA" in source
    assert "not native TorchTPU" in source


def test_xla_validation_notebook_is_labeled_historical():
    source = _notebook_source(Path("notebooks/TrainLM_TPU_Validation.ipynb"))
    assert "Historical PyTorch/XLA validation notebook" in source
    assert "does not validate the current native TorchTPU path" in source


def test_notebook_catalog_classifies_every_notebook():
    catalog = Path("notebooks/README.md").read_text(encoding="utf-8")
    for path in Path("notebooks").glob("*.ipynb"):
        assert path.name in catalog, f"Unclassified notebook: {path.name}"
