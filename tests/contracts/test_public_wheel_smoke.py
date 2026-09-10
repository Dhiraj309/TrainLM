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
    environment = tmp_path / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
        check=True,
    )
    python = environment / "bin" / "python"
    wheel = next(wheel_dir.glob("trainlm-*.whl"))
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
    )
    smoke_dir = tmp_path / "outside-checkout"
    smoke_dir.mkdir()
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
from pathlib import Path
import trainlm
assert 'site-packages' in Path(trainlm.__file__).as_posix()
assert trainlm.PUBLIC_API_VERSION == '1'
assert set(trainlm.__all__) == {
    'DEPRECATED_CONFIG_KEYS',
    'HuggingFaceShardSourceConfig',
    'HuggingFaceShardSpec',
    'PUBLIC_API_VERSION',
    'PackedBinDataset',
    'TrainLMTrainer',
    'TrainLMTrainingArguments',
}
args = trainlm.TrainLMTrainingArguments(max_steps=1, accelerator='cpu')
assert args.max_steps == 1
""",
        ],
        check=True,
        cwd=smoke_dir,
    )
