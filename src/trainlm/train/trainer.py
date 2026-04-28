import json
import time
from pathlib import Path
from typing import Iterator, Optional, Any, Dict

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
    """
    Convert a JAX scalar or array to a Python float.
    """
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
    """
    Take the first replica from a replicated pytree.
    """
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _replicate_pytree(tree, devices):
    """
    Replicate a pytree across local devices.
    """
    return jax.device_put_replicated(tree, devices)


def _extract_input_ids(batch: Any) -> jnp.ndarray:
    """
    Accept either:
      - raw array: [batch, seq]
      - dict with key 'input_ids'
    """
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("Batch dict must contain key 'input_ids'.")
        return batch["input_ids"]
    return batch


def _reshape_for_pmap(
    batch: jnp.ndarray,
    num_devices: int,
    micro_batch_per_device: int,
) -> jnp.ndarray:
    """
    Reshape a host batch into:
        [num_devices, micro_batch_per_device, seq_len]
    """
    if batch.ndim != 2:
        raise ValueError(
            f"Expected batch ndim=2 [batch, seq], got shape={batch.shape}"
        )

    global_batch, seq_len = batch.shape
    expected = num_devices * micro_batch_per_device

    if global_batch != expected:
        raise ValueError(
            f"Batch size mismatch: got {global_batch}, expected {expected} "
            f"(num_devices={num_devices}, micro_batch_per_device={micro_batch_per_device})."
        )

    return batch.reshape(num_devices, micro_batch_per_device, seq_len)


def _build_pmap_batch(
    micro_batches: list,
    num_devices: int,
    micro_batch_per_device: int,
) -> jnp.ndarray:
    """
    Given grad_accum host batches shaped:
        [global_batch, seq_len]

    produce:
        [num_devices, grad_accum, micro_batch_per_device, seq_len]
    """
    per_step_batches = [
        _reshape_for_pmap(b, num_devices, micro_batch_per_device)
        for b in micro_batches
    ]

    stacked = jnp.stack(per_step_batches, axis=0)
    # [grad_accum, num_devices, micro_batch, seq] -> [num_devices, grad_accum, micro_batch, seq]
    return jnp.swapaxes(stacked, 0, 1)


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

class Trainer:
    """
    Config-driven trainer for Flax/HF models.
    """

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
        # Build model and initialize parameters
        # --------------------------------------------------------
        self.model, params = build_model(
            model_cfg=config.model,
            parallel_cfg=config.parallelism,
            checkpoint_dir=None,
        )

        dummy_batch = jnp.zeros(
            (config.runtime.micro_batch_per_device, config.runtime.seq_len),
            dtype=jnp.int32,
        )

        init_variables = self.model.init(self.rng, dummy_batch)
        params = init_variables["params"]

        # --------------------------------------------------------
        # Scheduler and optimizer
        # --------------------------------------------------------
        self.schedule = build_scheduler(config, num_devices=self.num_devices)
        self.optimizer = build_optimizer(
            config.optimizer,
            self.schedule,
            params,
        )

        opt_state = self.optimizer.init(params)

        # --------------------------------------------------------
        # Training state (host copy)
        # --------------------------------------------------------
        self.state = TrainState(
            step=0,
            params=params,
            opt_state=opt_state,
            tx=self.optimizer,
            apply_fn=self.model.apply,
            rng_key=self.rng,
            tokens_processed=0,
        )

        # --------------------------------------------------------
        # Replicate state for pmap
        # --------------------------------------------------------
        self.state = _replicate_pytree(self.state, self.devices)

        # --------------------------------------------------------
        # PMapped steps
        # --------------------------------------------------------
        self.train_step = create_train_step(
            grad_accum=config.runtime.gradient_accumulation,
            num_devices=self.num_devices,
        )
        self.eval_step = create_eval_step()

        # --------------------------------------------------------
        # Runtime / checkpointing
        # --------------------------------------------------------
        self.checkpoint_dir = Path(resume_dir or config.runtime.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = config.runtime.checkpoint_interval
        self.max_to_keep = getattr(config.runtime, "checkpoint_max_to_keep", 3)

        # Snapshot config for reproducibility
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)

        # --------------------------------------------------------
        # Optional restore
        # --------------------------------------------------------
        restored = self._restore_checkpoint()
        if restored is not None:
            self.state = _replicate_pytree(restored, self.devices)
            print(f"[trainer] resumed from step {restored.step}")

    # ------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------

    def _restore_checkpoint(self) -> Optional[TrainState]:
        """
        Restore the latest checkpoint if present.
        """
        target = _unreplicate_pytree(self.state)

        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(self.checkpoint_dir),
            target=target,
        )

        if restored is None:
            return None

        if isinstance(restored, dict) and "step" not in restored:
            return None

        return restored

    def _save_checkpoint(self):
        """
        Save unreplicated state to disk.
        """
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

    def train(
        self,
        dataloader: Iterator,
        num_steps: Optional[int] = None,
        eval_dataloader: Optional[Iterator] = None,
    ):
        """
        Train the model.

        Expected dataloader batches:
            [num_devices * micro_batch_per_device, seq_len]
        """

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
        print(f"Training for {num_steps:,} optimizer steps")
        print(f"Devices               : {self.num_devices}")
        print(f"Global batch size     : {global_batch_size}")
        print(f"Tokens per step       : {tokens_per_step:,}")
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
                micro_batches=micro_batches,
                num_devices=self.num_devices,
                micro_batch_per_device=cfg.runtime.micro_batch_per_device,
            )

            if int(_unreplicate_pytree(self.state).step) == 0:
                print(f"[debug] pmap batch shape: {batch.shape}")

            self.state, metrics = self.train_step(self.state, batch)

            # Metrics are replicated; take first replica.
            metrics = _unreplicate_pytree(metrics)

            step_time = time.time() - step_start
            host_state = _unreplicate_pytree(self.state)

            if int(host_state.step) % cfg.runtime.log_interval == 0:
                lr = _scalar(self.schedule(int(host_state.step)))

                print(
                    f"step={int(host_state.step):>8} "
                    f"loss={_scalar(metrics['loss']):.4f} "
                    f"grad_norm={_scalar(metrics['grad_norm']):.4f} "
                    f"lr={lr:.6g} "
                    f"tokens={int(host_state.tokens_processed):,} "
                    f"step_time={step_time:.4f}s"
                )

            if int(host_state.step) % cfg.runtime.checkpoint_interval == 0:
                self._save_checkpoint()

                print(f"[trainer] checkpoint saved at step {int(host_state.step)}")

            if eval_dataloader is not None and int(host_state.step) % cfg.runtime.eval_interval == 0:
                self._run_eval(eval_dataloader)

        self._save_checkpoint()
        print("[trainer] final checkpoint saved")

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def _run_eval(self, eval_dataloader: Iterator):
        """
        Minimal evaluation loop.
        """
        print(f"[eval] step={int(_unreplicate_pytree(self.state).step)}")

        data_iter = iter(eval_dataloader)
        batch = next(data_iter)
        batch = _extract_input_ids(batch)
        batch = jnp.asarray(batch, dtype=jnp.int32)

        batch = _reshape_for_pmap(
            batch=batch,
            num_devices=self.num_devices,
            micro_batch_per_device=self.config.runtime.micro_batch_per_device,
        )

        metrics = self.eval_step(self.state, batch)
        metrics = _unreplicate_pytree(metrics)

        print(
            f"[eval] loss={_scalar(metrics['loss']):.4f} "
            f"cross_entropy={_scalar(metrics.get('cross_entropy')):.4f} "
            f"z_loss={_scalar(metrics.get('z_loss')):.4f}"
            )
