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
        "!git clone --branch milestone/m10-m12-kernels-parity --single-branch https://github.com/Dhiraj309/TrainLM.git",
        "%cd TrainLM",
        'git fetch origin "$BRANCH"',
        'git checkout -B "$BRANCH" "origin/$BRANCH"',
        "git rev-parse HEAD",
        "%pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel tensorflow-rocm tf-keras keras torchvision torchaudio",
        "Restart Session",
        "TensorFlow is still importable",
        '"initialization": "config"',
        '"model_type": "llama"',
        '"vocab_size": 32064',
        '"num_hidden_layers": 8',
        "sequence_length = 2048",
        '"per_device_train_batch_size": 4',
        '"gradient_accumulation_steps": 16',
        "TrainLMTrainer.from_config",
        "PackedBinDataset.from_hub",
        "shard_range=(0, TRAIN_SHARD_STOP)",
        "cache_dir=DATA_CACHE_DIR",
        "It does not download during training",
        "result = trainer.train()",
        "Stale notebook/trainer configuration detected",
        "Throughput candidate verified:",
        "terminates its private worker process group",
        'trainer.explain(format="text")',
        '"max_steps": 100',
        '"warmup_steps": 5',
        '"lr_scheduler_type": "wsd"',
        "Evaluation and checkpointing are intentionally disabled",
        "[Notebook 1/8]",
        "[Notebook 8/8]",
        "10-second heartbeats",
    ):
        assert required in source

    # The checkout bootstrap must run before package installation so a fresh
    # Kaggle session never installs a stale copy of the repository.
    assert source.index("!git clone --branch") < source.index("%pip uninstall")
    assert source.index('git checkout -B "$BRANCH"') < source.index("%pip uninstall")
    assert '"eval_steps"' not in source
    assert '"save_steps"' not in source
    assert "eval_dataset=" not in source
    assert "HuggingFaceTB/SmolLM2-135M-Instruct" not in source


def test_tpu_notebook_hides_orchestration_and_topology_inputs():
    source = _source()

    assert source.index("%pip uninstall") < source.index("from trainlm import")

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

    assert "microsoft/Phi-3.5-mini-instruct" not in source
