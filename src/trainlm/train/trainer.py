import json
import time
from pathlib import Path
from typing import Iterator, Optional, Any

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


def _unreplicate_pytree(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _replicate_pytree(tree, devices):
    return jax.device_put_replicated(tree, devices)


def _extract_input_ids(batch: Any) -> jnp.ndarray:
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("Batch dict must contain key 'input_ids'.")
        return batch["input_ids"]
    return batch


def _reshape_for_pmap(batch, num_devices, micro_batch_per_device):
    global_batch, seq_len = batch.shape
    expected = num_devices * micro_batch_per_device

    if global_batch != expected:
        raise ValueError(
            f"Batch size mismatch: got {global_batch}, expected {expected}"
        )

    return batch.reshape(num_devices, micro_batch_per_device, seq_len)


def _build_pmap_batch(micro_batches, num_devices, micro_batch_per_device):
    per_step = [
        _reshape_for_pmap(b, num_devices, micro_batch_per_device)
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

        print(f"[trainer] local devices: {self.num_devices}")

        self.rng = jax.random.PRNGKey(seed)

        # --------------------------------------------------------
        # ✅ FIX: HF model init (NO .init)
        # --------------------------------------------------------
        self.model, params = build_model(
            model_cfg=config.model,
            parallel_cfg=config.parallelism,
            checkpoint_dir=None,
        )

        # --------------------------------------------------------
        # Scheduler + optimizer
        # --------------------------------------------------------
        self.schedule = build_scheduler(config, num_devices=self.num_devices)
        self.optimizer = build_optimizer(
            config.optimizer,
            self.schedule,
            params,
        )

        opt_state = self.optimizer.init(params)

        # --------------------------------------------------------
        # Training state (NO apply_fn)
        # --------------------------------------------------------
        self.state = TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            tx=self.optimizer,
            rng_key=self.rng,
            tokens_processed=0,
        )

        self.state = _replicate_pytree(self.state, self.devices)

        # --------------------------------------------------------
        # PMAP steps
        # --------------------------------------------------------
        self.train_step = create_train_step(
            model=self.model,  # ✅ pass model explicitly
            grad_accum=config.runtime.gradient_accumulation,
            num_devices=self.num_devices,
        )

        self.eval_step = create_eval_step(model=self.model)

        # --------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------
        self.checkpoint_dir = Path(resume_dir or config.runtime.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = config.runtime.checkpoint_interval
        self.max_to_keep = getattr(config.runtime, "checkpoint_max_to_keep", 3)

        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)

        # restored = self._restore_checkpoint()
        # if restored is not None:
            # self.state = _replicate_pytree(restored, self.devices)
            #print(f"[trainer] resumed from step {restored.step}")

        restored = None  # TEMP disable restore (fix OOM)

    # ------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------

    def _restore_checkpoint(self):
        target = _unreplicate_pytree(self.state)

        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(self.checkpoint_dir),
            target=target,
        )

        if restored is None:
            return None

        # Already correct type (normal case)
        if isinstance(restored, TrainState):
            return restored

        # Legacy dict checkpoint (fallback)
        if isinstance(restored, dict):
            try:
                return TrainState(**restored)
            except Exception:
                return None

        return None

    def _save_checkpoint(self):
        state_to_save = _unreplicate_pytree(self.state)

        checkpoints.save_checkpoint(
            ckpt_dir=str(self.checkpoint_dir),
            target=state_to_save,
            step=int(state_to_save.step),
            keep=self.max_to_keep,
            overwrite=False,
        )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    def train(self, dataloader, num_steps=None, eval_dataloader=None):

        cfg = self.config

        if num_steps is None:
            num_steps = compute_total_steps(cfg.runtime, self.num_devices)

        global_batch_size = self.num_devices * cfg.runtime.micro_batch_per_device
        tokens_per_step = (
            cfg.runtime.seq_len
            * global_batch_size
            * cfg.runtime.gradient_accumulation
        )

        print("\n" + "=" * 72)
        print(f"Training for {num_steps:,} steps")
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

            metrics = _unreplicate_pytree(metrics)

            step_time = time.time() - step_start
            host_state = _unreplicate_pytree(self.state)

            if int(host_state.step) % cfg.runtime.log_interval == 0:
                lr = _scalar(self.schedule(int(host_state.step)))

                print(
                    f"step={int(host_state.step)} "
                    f"loss={_scalar(metrics['loss']):.4f} "
                    f"lr={lr:.6g} "
                    f"time={step_time:.4f}s"
                )

            if int(host_state.step) % cfg.runtime.checkpoint_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        print("[trainer] training complete")
