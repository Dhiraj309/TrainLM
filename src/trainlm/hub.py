"""Explicit, allowlisted publication of portable model exports to the HF Hub."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


_WEIGHT_NAME = re.compile(
    r"(?:model(?:-\d+-of-\d+)?\.safetensors|"
    r"pytorch_model(?:-\d+-of-\d+)?\.bin|"
    r"model\.safetensors\.index\.json|pytorch_model\.bin\.index\.json)\Z"
)
_MODEL_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "spiece.model",
        "tokenizer.tiktoken",
        "chat_template.jinja",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
        "feature_extractor_config.json",
        "audio_preprocessor_config.json",
        "video_preprocessor_config.json",
        "normalizer.json",
    }
)
_HUB_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class HubPublishResult:
    """The Hugging Face repository and immutable commit created by a publish."""

    repo_id: str
    repo_url: str
    revision: str
    commit_url: str
    private: bool


def _is_model_artifact(path: Path) -> bool:
    return path.name == "README.md" or path.name in _MODEL_FILES or bool(
        _WEIGHT_NAME.fullmatch(path.name)
    )


def _export_files(export_dir: Path) -> list[str]:
    if not export_dir.is_dir():
        raise ValueError(f"Model export directory does not exist: {export_dir}")
    config_path = export_dir / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError("Model export must contain a regular config.json file.")
    readme_path = export_dir / "README.md"
    if readme_path.is_symlink() or (readme_path.exists() and not readme_path.is_file()):
        raise ValueError("Model export README.md must be a regular file.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Model export config.json is not valid JSON.") from exc
    if isinstance(config, dict) and config.get("auto_map"):
        raise ValueError(
            "This export references custom remote model code. TrainLM's Hub "
            "publisher does not upload or enable that code; review and publish "
            "it separately if needed."
        )
    tokenizer_config_path = export_dir / "tokenizer_config.json"
    if tokenizer_config_path.is_file() and not tokenizer_config_path.is_symlink():
        try:
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Model export tokenizer_config.json is not valid JSON."
            ) from exc
        if isinstance(tokenizer_config, dict) and tokenizer_config.get("auto_map"):
            raise ValueError(
                "This export references custom remote tokenizer code. TrainLM's "
                "Hub publisher does not upload or enable that code; review and "
                "publish it separately if needed."
            )

    files = []
    has_weights = False
    for path in sorted(export_dir.iterdir()):
        if (
            not path.is_file()
            or path.is_symlink()
            or not _is_model_artifact(path)
        ):
            continue
        relative = path.relative_to(export_dir).as_posix()
        files.append(relative)
        has_weights = has_weights or bool(_WEIGHT_NAME.fullmatch(path.name))
    if not has_weights:
        raise ValueError(
            "Model export must contain a supported .safetensors or PyTorch "
            ".bin weight file."
        )
    return files


def _model_card(metadata: Mapping[str, Any]) -> str:
    workflow = metadata.get("workflow")
    model_type = metadata.get("model_type")
    model_class = metadata.get("model_class")
    initialization = metadata.get("initialization")
    base_model = metadata.get("base_model")
    base_revision = metadata.get("base_model_revision")

    lines = ["---", "library_name: trainlm", "pipeline_tag: text-generation"]
    if isinstance(base_model, str) and _HUB_MODEL_ID.fullmatch(base_model):
        lines.append(f"base_model: {json.dumps(base_model)}")
    lines.extend(
        [
            "---",
            "",
            "# Model card",
            "",
            "Exported with [TrainLM](https://github.com/Dhiraj309/TrainLM).",
        ]
    )
    if isinstance(workflow, str) and workflow:
        lines.extend(["", f"Training workflow: {workflow}."])
    if isinstance(model_class, str) and model_class:
        lines.append(f"Model class: `{model_class}`.")
    if isinstance(model_type, str) and model_type:
        lines.append(f"Model type: `{model_type}`.")
    if isinstance(initialization, str) and initialization:
        lines.append(f"Initialization: `{initialization}`.")
    if isinstance(base_model, str) and _HUB_MODEL_ID.fullmatch(base_model):
        lines.append(
            f"Base model: [`{base_model}`](https://huggingface.co/{base_model})."
        )
    if isinstance(base_revision, str) and re.fullmatch(r"[0-9a-f]{40}", base_revision):
        lines.append(f"Base model revision: `{base_revision}`.")
    lines.extend(
        [
            "",
            "Review and add the applicable model and dataset license information "
            "before making this repository public. TrainLM's software license "
            "does not determine the license for model weights or training data.",
            "",
        ]
    )
    return "\n".join(lines)


def _publish_model_export(
    export_path: str | Path,
    repo_id: str,
    *,
    private: bool,
    commit_message: str | None,
    metadata: Mapping[str, Any],
) -> HubPublishResult:
    """Publish only model artifacts after explicit visibility verification."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty Hugging Face model ID.")
    if type(private) is not bool:
        raise TypeError("private must be a boolean.")
    if commit_message is not None and (
        not isinstance(commit_message, str) or not commit_message.strip()
    ):
        raise ValueError("commit_message must be a non-empty string when provided.")

    export_dir = Path(export_path).expanduser()
    files = _export_files(export_dir)

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "Hub publishing requires huggingface_hub. "
            "Install TrainLM's core dependencies."
        ) from exc

    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id, repo_type="model")
    except RepositoryNotFoundError:
        actual_private = private
        repository_missing = True
    else:
        repository_missing = False
        actual_private = getattr(repo_info, "private", None)
        if not isinstance(actual_private, bool):
            raise RuntimeError(
                "Hugging Face did not return repository visibility; refusing to upload."
            )
        if actual_private != private:
            requested = "private" if private else "public"
            existing = "private" if actual_private else "public"
            raise ValueError(
                f"Repository {repo_id!r} is already {existing}; requested {requested}. "
                "TrainLM will not change an existing repository's visibility."
            )

    readme = export_dir / "README.md"
    if not readme.exists():
        readme.write_text(_model_card(metadata), encoding="utf-8")
        files.append("README.md")
    if repository_missing:
        api.create_repo(repo_id, repo_type="model", private=private)

    endpoint = str(getattr(api, "endpoint", "https://huggingface.co")).rstrip("/")
    repo_url = f"{endpoint}/{repo_id}"
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        allow_patterns=sorted(set(files)),
        commit_message=commit_message or "Upload model export from TrainLM",
    )
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(
            "Hugging Face upload completed without returning a full commit SHA."
        )
    commit_url = getattr(commit, "commit_url", None)
    if not isinstance(commit_url, str) or not commit_url:
        commit_url = f"{repo_url}/commit/{revision}"
    return HubPublishResult(
        repo_id=repo_id,
        repo_url=repo_url,
        revision=revision,
        commit_url=commit_url,
        private=actual_private,
    )


__all__ = ["HubPublishResult"]
