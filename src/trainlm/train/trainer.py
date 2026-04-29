import json
import time
from pathlib import Path
from typing import Optional, Any

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from trainlm.config.schema import TrainConfig
from trainlm.model.model_factory import build_model
from trainlm.train.optimizer import build_optimizer
from trainlm.train.scheduler import build_scheduler, compute_total_steps
from trainlm.train.step import create_train_step, create_eval_step
from trainlm.train.state import TrainState


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _scalar(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(jax.device_get(x))
        except Exception:
            return float("nan")


def _unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _replicate(tree, devices):
    return jax.device_put_replicated(tree, devices)


def _extract_input_ids(batch: Any) -> jnp.ndarray:
    if isinstance(batch, dict):
        return batch["input_ids"]
    return batch


def _reshape_for_pmap(batch, num_devices, micro_batch):
    global_batch, seq_len = batch.shape
    expected = num_devices * micro_batch

    if global_batch != expected:
        raise ValueError(
            f"Batch mismatch: got {global_batch}, expected {expected}"
        )

    return batch.reshape(num_devices, micro_batch, seq_len)


def _build_pmap_batch(micro_batches, num_devices, micro_batch):
    per_step = [
        _reshape_for_pmap(b, num_devices, micro_batch)
        for b in micro_batches
    ]

    stacked = jnp.stack(per_step, axis=0)
    return jnp.swapaxes(stacked, 0, 1)


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class Trainer:

    def __init__(
        self,
        config: TrainConfig,
        resume_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.config = config
        self.num_devices = jax.local_device_count()
        self.devices = jax.local_devices()

        print(f"[trainer] devices: {self.num_devices}")

        self.rng = jax.random.PRNGKey(seed)

        # 🔥 LAZY INIT (critical)
        self.model = None
        self.state = None
        self.train_step = None
        self.eval_step = None
        self.schedule = None
        self.optimizer = None

        # --------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------
        self.checkpoint_dir = Path(
            resume_dir or config.runtime.checkpoint_dir
        ).resolve()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = config.runtime.checkpoint_interval
        self.max_to_keep = getattr(
            config.runtime, "checkpoint_max_to_keep", 3
        )

        # Save config
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)

    # ------------------------------------------------------------
    # Setup (LAZY INIT)
    # ------------------------------------------------------------

    def setup(self):
        print("[setup] building model...")

        self.model, params = build_model(
            model_cfg=self.config.model,
            parallel_cfg=self.config.parallelism,
            checkpoint_dir=None,
        )

        print("[setup] building optimizer...")

        self.schedule = build_scheduler(
            self.config,
            num_devices=self.num_devices,
        )

        self.optimizer = build_optimizer(
            self.config.optimizer,
            self.schedule,
            params,
        )

        print("[setup] initializing optimizer state...")

        opt_state = self.optimizer.init(params)

        state = TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            tx=self.optimizer,
            rng_key=self.rng,
            tokens_processed=0,
        )

        print("[setup] replicating state...")
        self.state = _replicate(state, self.devices)

        print("[setup] creating train step...")
        self.train_step = create_train_step(
            model=self.model,
            grad_accum=self.config.runtime.gradient_accumulation,
        )

        self.eval_step = create_eval_step(self.model)

        print("[setup] done")

    # ------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------

    def _save_checkpoint(self):
        state = _unreplicate(self.state)

        checkpoints.save_checkpoint(
            ckpt_dir=str(self.checkpoint_dir),
            target=state,
            step=int(state.step),
            keep=self.max_to_keep,
            overwrite=False,
        )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    def train(self, dataloader, num_steps=None):

        # 🔥 ensure setup is called only when needed
        if self.state is None:
            self.setup()

        cfg = self.config

        if num_steps is None:
            num_steps = compute_total_steps(cfg.runtime, self.num_devices)

        global_batch = self.num_devices * cfg.runtime.micro_batch_per_device
        tokens_per_step = (
            cfg.runtime.seq_len
            * global_batch
            * cfg.runtime.gradient_accumulation
        )

        print("\n" + "=" * 72)
        print(f"steps: {num_steps:,}")
        print(f"tokens/step: {tokens_per_step:,}")
        print("=" * 72 + "\n")

        data_iter = iter(dataloader)

        for _ in range(num_steps):

            step_start = time.time()

            micro_batches = []
            for _ in range(cfg.runtime.gradient_accumulation):
                batch = next(data_iter)
                batch = _extract_input_ids(batch)
                batch = jnp.asarray(batch, dtype=jnp.int32)
                micro_batches.append(batch)

            batch = _build_pmap_batch(
                micro_batches,
                self.num_devices,
                cfg.runtime.micro_batch_per_device,
            )

            self.state, metrics = self.train_step(self.state, batch)

            metrics = _unreplicate(metrics)
            state = _unreplicate(self.state)

            step_time = time.time() - step_start

            steps_per_sec = 1.0 / step_time
            tokens_per_sec = tokens_per_step * steps_per_sec

            if int(state.step) % cfg.runtime.log_interval == 0:

                lr = _scalar(self.schedule(int(state.step)))

                print(
                    f"step={int(state.step):>7} "
                    f"loss={_scalar(metrics['loss']):.4f} "
                    f"grad_norm={_scalar(metrics['grad_norm']):.3f} "
                    f"lr={lr:.6g} "
                    f"{tokens_per_sec/1000:.1f}k tok/s "
                    f"{steps_per_sec:.1f} step/s "
                    f"{step_time:.4f}s"
                )

            if int(state.step) % cfg.runtime.checkpoint_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        print("[trainer] done")