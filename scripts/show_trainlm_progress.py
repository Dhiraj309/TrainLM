"""Display the single live TrainLM progress document in a notebook.

Usage from Kaggle/Jupyter:
    %run scripts/show_trainlm_progress.py --output-dir /kaggle/working/run --watch

Run the watcher in a second cell or notebook while the training process owns
the TPU. Without ``--watch`` it renders one snapshot and exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def _display(markdown: str) -> None:
    try:
        from IPython.display import Markdown, clear_output, display
    except ImportError:
        print(markdown, end="")
        return
    clear_output(wait=True)
    display(Markdown(markdown))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    path = args.output_dir / "progress.md"
    while True:
        if path.is_file():
            _display(path.read_text(encoding="utf-8"))
            text = path.read_text(encoding="utf-8")
            if not args.watch or any(
                f"**Status:** `{status}`" in text
                for status in ("finalized", "failed")
            ):
                return
        elif not args.watch:
            raise SystemExit(f"Progress document does not exist yet: {path}")
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
