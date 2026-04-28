from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import jax
import jax.numpy as jnp

from trainlm.train.state import TrainState


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _extract_input_ids(batch: Any) -> jnp.ndarray:
    """
    Accept either:
      - raw array: [batch, seq_len]
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


def _unreplicate_pytree(tree):
    """
    Take the first replica from a replicated pytree.
    """
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _to_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert JAX scalars/arrays into Python floats.
    """
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            out[key] = float(jax.device_get(value))
        except Exception:
            out[key] = float("nan")
    return out


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def run_evaluation(
    eval_step,
    state: TrainState,
    eval_dataloader: Iterator,
    num_devices: int,
    micro_batch_per_device: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run evaluation over a dataloader and return averaged metrics.

    Expected eval batches:
        [num_devices * micro_batch_per_device, seq_len]
    """
    data_iter = iter(eval_dataloader)

    totals: Dict[str, float] = {}
    count = 0

    while True:
        if max_batches is not None and count >= max_batches:
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            break

        batch = _extract_input_ids(batch)
        batch = jnp.asarray(batch, dtype=jnp.int32)
        batch = _reshape_for_pmap(
            batch=batch,
            num_devices=num_devices,
            micro_batch_per_device=micro_batch_per_device,
        )

        metrics = eval_step(state, batch)
        metrics = _unreplicate_pytree(metrics)
        metrics = _to_float_dict(metrics)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value

        count += 1

    if count == 0:
        return {}

    averaged = {key: value / count for key, value in totals.items()}

    if "loss" in averaged:
        averaged["perplexity"] = float(jnp.exp(jnp.minimum(jnp.array(averaged["loss"]), 20.0)))

    return averaged
