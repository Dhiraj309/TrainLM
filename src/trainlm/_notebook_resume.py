"""Private resume examples for the raw-text pretraining notebook report."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import TrainLMTrainer
    from .pretraining import TrainLMPretrainingConfig


def build_resume_report(
    config: TrainLMPretrainingConfig,
    *,
    trainer: TrainLMTrainer | None,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture a runnable public-API example without loading checkpoint state.

    The example targets the recorded checkpoint, preserves every config value,
    and requires the original run artifacts. A previously requested budget
    extension is retried only while its configuration publication is pending;
    a committed extension instead receives an ordinary exact-resume example.
    """
    settings = asdict(config)
    settings["output_dir"] = str(Path(config.output_dir).resolve())
    result = {
        "status": "unavailable",
        "reason": (
            "checkpoint_lookup_failed"
            if checkpoint.get("lookup_error_type") else "no_checkpoint"
        ),
        "pretraining_config": settings,
        "checkpoint_path": None,
        "extend_training_budget": False,
        "script": None,
        "code": None,
    }
    latest_path = checkpoint.get("latest_path")
    if trainer is None or not latest_path or checkpoint.get("lookup_error_type"):
        return result

    checkpoint_path = str(Path(latest_path).resolve())
    # This state stays pending until a compatible checkpoint and its run
    # configuration are published. The original notebook flag can remain True
    # after that point, so it must not be reused for a completed extension.
    extension_pending = (
        getattr(trainer, "_pending_scratch_run_configuration", None) is not None
    )
    code = (
        "# Restore the original run directory before running this example.\n"
        "# Match the TrainLM source revision and package versions in report.json.\n"
        "# The downloaded model export alone is not a training checkpoint.\n"
        "from pathlib import Path\n"
        "from trainlm import TrainLMPretrainingConfig, TrainLMTrainer\n\n"
        f"settings = {pformat(settings, sort_dicts=False)}\n"
        "config = TrainLMPretrainingConfig(**settings)\n"
        f"checkpoint = Path({checkpoint_path!r})\n"
        "if (\n"
        "    not checkpoint.is_dir()\n"
        "    or not (config.output_dir / 'run_config.json').is_file()\n"
        "    or not (config.output_dir / 'preparation_cache.json').is_file()\n"
        "):\n"
        "    raise FileNotFoundError(\n"
        "        'Restore the full original run directory at its recorded path.'\n"
        "    )\n"
        "trainer = TrainLMTrainer.from_scratch(\n"
        f"    config, extend_training_budget={extension_pending!r},\n"
        ")\n"
        "state = trainer.train(resume_from_checkpoint=checkpoint)\n"
        "print(f'resumed_step={state.step} supervised_tokens={state.tokens_seen}')\n"
    )
    result.update({
        "status": "available",
        "reason": None,
        "checkpoint_path": checkpoint_path,
        "extend_training_budget": extension_pending,
        "script": "resume.py",
        "code": code,
    })
    return result
