import argparse
import sys
from pathlib import Path

# Ensure package is importable when running as a script
# (safe even if installed in editable mode)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trainlm.config.loader import load_config
from trainlm.data.dataloader import build_train_dataloader
from trainlm.train.trainer import Trainer


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TrainLM - Training Entrypoint")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config YAML",
    )

    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Optional override config YAML",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory to resume from",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of training steps",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    config = load_config(args.config, args.override)

    # --------------------------------------------------------
    # Compute global batch size
    # --------------------------------------------------------
    import jax

    num_devices = jax.local_device_count()
    global_batch_size = (
        num_devices * config.runtime.micro_batch_per_device
    )

    print(f"[entry] devices: {num_devices}")
    print(f"[entry] global_batch_size: {global_batch_size}")

    # --------------------------------------------------------
    # Build dataloader
    # --------------------------------------------------------
    if not config.data.sources:
        raise ValueError("No dataset sources provided in config.data.sources")

    dataloader = build_train_dataloader(
        paths=config.data.sources,
        seq_len=config.runtime.seq_len,
        global_batch_size=global_batch_size,
        seed=42,
    )

    # --------------------------------------------------------
    # Build trainer
    # --------------------------------------------------------
    trainer = Trainer(
        config=config,
        resume_dir=args.resume,
    )

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------
    trainer.train(
        dataloader=dataloader,
        num_steps=args.steps,
    )


if __name__ == "__main__":
    main()
