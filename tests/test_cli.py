"""Public CLI delegates exclusively to the HF-like facade."""

from types import SimpleNamespace

from trainlm import cli


def test_train_command_builds_public_datasets_and_delegates(
    monkeypatch, tmp_path, capsys
):
    config = tmp_path / "train.yaml"
    config.write_text(
        "api_version: '1'\n"
        "model: local-model\n"
        "training_args:\n"
        "  sequence_length: 128\n"
        "  seed: 17\n",
        encoding="utf-8",
    )
    calls = []

    def dataset_from_directory(path, **kwargs):
        calls.append(("dataset", path, kwargs))
        return f"dataset:{path.name}"

    class Trainer:
        def train(self, *, resume_from_checkpoint=None):
            calls.append(("train", resume_from_checkpoint))
            return SimpleNamespace(to_dict=lambda: {"step": 12})

    def trainer_from_config(path, *, train_dataset, eval_dataset):
        calls.append(("trainer", path, train_dataset, eval_dataset))
        return Trainer()

    monkeypatch.setattr(
        cli.PackedBinDataset, "from_directory", dataset_from_directory
    )
    monkeypatch.setattr(cli.TrainLMTrainer, "from_config", trainer_from_config)

    result = cli.main(
        (
            "train",
            "--config",
            str(config),
            "--train-manifest-dir",
            str(tmp_path / "train"),
            "--eval-manifest-dir",
            str(tmp_path / "eval"),
            "--resume-from-checkpoint",
            str(tmp_path / "checkpoint-12"),
        )
    )

    assert result == 0
    assert calls == [
        (
            "dataset",
            tmp_path / "train",
            {"sequence_length": 128, "split": "train", "seed": 17},
        ),
        (
            "dataset",
            tmp_path / "eval",
            {"sequence_length": 128, "split": "validation"},
        ),
        (
            "trainer",
            {
                "api_version": "1",
                "model": "local-model",
                "training_args": {"sequence_length": 128, "seed": 17},
            },
            "dataset:train",
            "dataset:eval",
        ),
        ("train", tmp_path / "checkpoint-12"),
    ]
    assert capsys.readouterr().out == '{"step": 12}\n'
