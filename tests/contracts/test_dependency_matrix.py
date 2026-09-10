import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).parents[2]
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
MATRIX_PATH = REPOSITORY_ROOT / "compatibility" / "dependency_matrix_v1.json"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _constraints(relative_path: str) -> dict[str, str]:
    lines = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    packages = {}
    for line in lines:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, version = requirement.split("==", maxsplit=1)
        normalized_name = name.lower().replace("_", "-")
        assert normalized_name not in packages
        packages[normalized_name] = version
    return packages


def test_core_dependencies_are_portable_and_transformers_v5():
    project = _pyproject()["project"]

    assert project["requires-python"] == ">=3.10"
    assert set(project["dependencies"]) == {
        "huggingface-hub>=1.0,<2",
        "torch>=2.9,<2.14",
        "transformers>=5.0,<6",
    }
    assert all("xla" not in dependency.lower() for dependency in project["dependencies"])
    assert all("jax" not in dependency.lower() for dependency in project["dependencies"])


def test_tpu_extras_lock_matched_xla_and_pallas_release_families():
    extras = _pyproject()["project"]["optional-dependencies"]

    assert set(extras["tpu-xla"]) == {
        "torch==2.9.0",
        "torch-xla[tpu]==2.9.0",
        "transformers==5.15.0",
    }
    assert set(extras["tpu-pallas"]) == {
        "torch==2.9.0",
        "torch-xla[pallas,tpu]==2.9.0",
        "transformers==5.15.0",
    }


def test_matrix_profiles_match_their_constraint_files():
    matrix = _matrix()

    assert matrix["schema_version"] == 1
    assert matrix["core_contract"] == {
        "python": ">=3.10",
        "huggingface-hub": ">=1.0,<2",
        "torch": ">=2.9,<2.14",
        "transformers": ">=5.0,<6",
        "policy": (
            "CPU compatibility CI covers the minimum and current profiles. "
            "TPU certification uses exact backend profiles."
        ),
    }
    for profile in matrix["profiles"].values():
        expected_packages = {
            name.lower().replace("_", "-"): version
            for name, version in profile["packages"].items()
        }
        assert _constraints(profile["constraints"]) == expected_packages


def test_xla_package_metadata_pins_are_preserved_by_tpu_profiles():
    profiles = _matrix()["profiles"]
    xla = profiles["tpu-xla-stable"]
    pallas = profiles["tpu-xla-pallas"]

    assert xla["status"] == "resolution_target_pending_m5_tpu_validation"
    assert xla["packages"] == {
        "huggingface-hub": "1.16.4",
        "torch": "2.9.0",
        "torch-xla": "2.9.0",
        "transformers": "5.15.0",
        "libtpu": "0.0.21",
    }
    assert pallas["status"] == (
        "resolution_target_pending_m5_and_kernel_validation"
    )
    assert pallas["packages"] == {
        **xla["packages"],
        "jax": "0.7.1",
        "jaxlib": "0.7.1",
    }


def test_dependency_matrix_uses_only_versioned_official_sources():
    sources = _matrix()["official_sources"]

    assert len(sources) >= 5
    assert all(source["name"] for source in sources)
    assert all(source["url"].startswith("https://") for source in sources)
    assert any("pypi.org/pypi/torch-xla/2.9.0/json" in source["url"] for source in sources)
