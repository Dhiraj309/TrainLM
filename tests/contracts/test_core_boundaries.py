"""Source-level dependency boundaries for TrainLM's portable core."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "trainlm"
CORE_PACKAGES = (
    "training",
    "tasks",
    "optimization",
    "checkpoint",
    "runtime",
)
FORBIDDEN_IMPORT_ROOTS = {
    "torch_xla",
    "jax",
    "flax",
}
FORBIDDEN_MODEL_NAMESPACES = (
    "trainlm.model",
    "transformers.models",
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return tuple(imported)


def test_portable_core_has_no_accelerator_or_model_family_imports():
    violations = []

    for package in CORE_PACKAGES:
        for path in sorted((SOURCE_ROOT / package).rglob("*.py")):
            for module in _imports(path):
                root = module.split(".", maxsplit=1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS or module.startswith(
                    FORBIDDEN_MODEL_NAMESPACES
                ):
                    relative_path = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative_path}: {module}")

    assert not violations, "Portable-core dependency violations:\n" + "\n".join(
        violations
    )


def test_checkpoint_contracts_remain_framework_independent():
    forbidden = FORBIDDEN_IMPORT_ROOTS | {"torch", "transformers"}
    violations = []

    for path in sorted((SOURCE_ROOT / "checkpoint").rglob("*.py")):
        for module in _imports(path):
            if module.split(".", maxsplit=1)[0] in forbidden:
                relative_path = path.relative_to(REPOSITORY_ROOT)
                violations.append(f"{relative_path}: {module}")

    assert not violations, "Checkpoint contract dependency violations:\n" + "\n".join(
        violations
    )
