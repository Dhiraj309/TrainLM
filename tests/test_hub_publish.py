import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from trainlm import (
    HubPublishResult,
    TrainLMFineTuningResult,
    TrainLMPretrainingResult,
)


_COMMIT = "a" * 40


def _export(tmp_path: Path, *, readme: str | None = None) -> Path:
    export = tmp_path / "model"
    export.mkdir()
    (export / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    (export / "model.safetensors").write_bytes(b"model weights")
    (export / "tokenizer.json").write_text("{}")
    (export / "report.json").write_text("private run details")
    (export / "optimizer.pt").write_bytes(b"private optimizer state")
    checkpoint = export / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint weights")
    if readme is not None:
        (export / "README.md").write_text(readme)
    return export


def _fake_hub(monkeypatch, *, existing_private=None, upload_error=None):
    state = {"create": [], "upload": [], "existing_private": existing_private}

    class RepositoryNotFoundError(Exception):
        pass

    class HfApi:
        endpoint = "https://huggingface.co"

        def repo_info(self, repo_id, *, repo_type):
            assert repo_type == "model"
            if state["existing_private"] is None:
                raise RepositoryNotFoundError(repo_id)
            return SimpleNamespace(private=state["existing_private"])

        def create_repo(self, repo_id, *, repo_type, private):
            state["create"].append((repo_id, repo_type, private))
            return SimpleNamespace()

        def upload_folder(self, **kwargs):
            state["upload"].append(kwargs)
            if upload_error is not None:
                raise upload_error
            return SimpleNamespace(oid=_COMMIT, commit_url=None)

    hub = ModuleType("huggingface_hub")
    hub.HfApi = HfApi
    utils = ModuleType("huggingface_hub.utils")
    utils.RepositoryNotFoundError = RepositoryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return state


def test_pretraining_result_publishes_allowlisted_model_and_private_card(
    tmp_path, monkeypatch
):
    export = _export(tmp_path)
    state = _fake_hub(monkeypatch)
    result = TrainLMPretrainingResult(
        report={
            "status": "passed",
            "model": {
                "requested_model_type": "gpt2",
                "model_class": "GPT2LMHeadModel",
                "requested_initialization": "random",
            },
        },
        report_path=tmp_path / "report.json",
        export_path=export,
    )

    published = result.push_to_hub("dignity045/trainlm-test")

    assert isinstance(published, HubPublishResult)
    assert published.repo_id == "dignity045/trainlm-test"
    assert published.repo_url == "https://huggingface.co/dignity045/trainlm-test"
    assert published.revision == _COMMIT
    assert published.commit_url.endswith(f"/commit/{_COMMIT}")
    assert published.private is True
    assert state["create"] == [("dignity045/trainlm-test", "model", True)]
    uploaded = set(state["upload"][0]["allow_patterns"])
    assert uploaded == {
        "README.md",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    card = (export / "README.md").read_text()
    assert "library_name: trainlm" in card
    assert "GPT2LMHeadModel" in card
    assert "dataset license" in card
    assert "report.json" not in uploaded
    assert "optimizer.pt" not in uploaded
    assert "checkpoints/checkpoint-1/model.safetensors" not in uploaded


def test_finetuning_result_preserves_existing_readme_and_requires_visibility_match(
    tmp_path, monkeypatch
):
    export = _export(tmp_path, readme="# Maintainer model card\n")
    state = _fake_hub(monkeypatch, existing_private=False)
    result = TrainLMFineTuningResult(
        model_name_or_path="org/base-model",
        model_revision=None,
        dataset_id="private/dataset",
        dataset_revision=None,
        model_path=export,
        step=1,
        train_loss=1.0,
        evaluation={},
        resolved_model_revision="b" * 40,
    )

    with pytest.raises(ValueError, match="already public"):
        result.push_to_hub("dignity045/model", private=True)
    assert not state["upload"]
    assert not state["create"]

    published = result.push_to_hub("dignity045/model", private=False)
    assert published.private is False
    assert "README.md" in state["upload"][0]["allow_patterns"]
    assert (export / "README.md").read_text() == "# Maintainer model card\n"
    assert "private/dataset" not in (export / "README.md").read_text()


def test_publisher_rejects_custom_remote_code_and_missing_weights(
    tmp_path, monkeypatch
):
    export = _export(tmp_path)
    (export / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling.Model"}})
    )
    state = _fake_hub(monkeypatch)
    result = TrainLMPretrainingResult(
        report={"status": "passed"},
        report_path=tmp_path / "report.json",
        export_path=export,
    )
    with pytest.raises(ValueError, match="custom remote model code"):
        result.push_to_hub("dignity045/model")
    assert not state["upload"]

    (export / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    (export / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="supported .safetensors"):
        result.push_to_hub("dignity045/model")
    assert not state["upload"]


def test_failed_upload_keeps_local_model_export(tmp_path, monkeypatch):
    export = _export(tmp_path)
    _fake_hub(monkeypatch, upload_error=OSError("offline"))
    result = TrainLMFineTuningResult(
        model_name_or_path="org/base-model",
        model_revision=None,
        dataset_id=None,
        dataset_revision=None,
        model_path=export,
        step=1,
        train_loss=None,
        evaluation={},
    )

    with pytest.raises(OSError, match="offline"):
        result.push_to_hub("dignity045/model")
    assert (export / "model.safetensors").is_file()
    assert (export / "config.json").is_file()
