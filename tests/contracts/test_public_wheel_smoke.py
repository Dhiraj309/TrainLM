from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_installed_wheel_exposes_public_surface_outside_checkout(tmp_path: Path):
    wheel_dir = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
    )
    wheel = next(wheel_dir.glob("trainlm-*.whl"))
    site_packages = tmp_path / "site-packages"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(site_packages),
            str(wheel),
        ],
        check=True,
    )
    smoke_dir = tmp_path / "outside-checkout"
    smoke_dir.mkdir()
    smoke_code = f"""
import sys
sys.path.insert(0, {str(site_packages)!r})
from pathlib import Path
import trainlm
assert Path(trainlm.__file__).resolve().is_relative_to(Path({str(site_packages)!r}))
assert trainlm.PUBLIC_API_VERSION == '1'
assert set(trainlm.__all__) == {{
    'DEPRECATED_CONFIG_KEYS',
    'HuggingFaceShardSourceConfig',
    'HuggingFaceShardSpec',
    'PUBLIC_API_VERSION',
    'PackedBinDataset',
    'TrainLMTrainer',
    'TrainLMTrainingArguments',
}}
args = trainlm.TrainLMTrainingArguments(max_steps=1, accelerator='cpu')
assert args.max_steps == 1
"""
    subprocess.run(
        [sys.executable, "-I", "-c", smoke_code],
        check=True,
        cwd=smoke_dir,
    )
