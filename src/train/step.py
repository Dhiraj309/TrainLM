from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import optax

from train.loss import shift_tokens, compute_loss
from train.state import TrainState


Params = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ------------------------------------------------------------
# TRAIN STEP
# ------------------------------------------------------------

def create_train_step(
    grad_accum: int,
    axis_name: str = "batch",
    num_devices: int = 1,
) -> Callable:
    """
    Create a pmapped training step.

    Expected batch shape inside each device:
        (grad_accum, micro_batch, seq_len)
    """

    def train_step(state: TrainState, batch: Batch):
        """
        One optimizer step on one pmapped replica.
        """

        # Split RNG once for this step, then once per micro-step.
        step_rng, new_rng = jax.random.split(state.rng_key)
        micro_rngs = jax.random.split(step_rng, grad_accum)

        def loss_fn(params: Params, micro_batch: jnp.ndarray, rng: jnp.ndarray):
            inputs, targets = shift_tokens(micro_batch)

            outputs = state.apply_fn(
                {"params": params},
                inputs,
                train=True,
                dropout_rng=rng,
            )

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss, _ = compute_loss(logits, targets)
            return loss

        def scan_fn(carry, inputs):
            grads_accum = carry
            micro_batch, rng = inputs

            loss, grads = jax.value_and_grad(loss_fn)(
                state.params,
                micro_batch,
                rng,
            )

            grads_accum = jax.tree_util.tree_map(
                lambda a, b: a + b,
                grads_accum,
                grads,
            )

            return grads_accum, loss

        grads_init = jax.tree_util.tree_map(jnp.zeros_like, state.params)

        grads_accum, losses = jax.lax.scan(
            scan_fn,
            grads_init,
            (batch, micro_rngs),
        )

        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(losses)

        # Sync across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)
        grad_norm = jax.lax.pmean(optax.global_norm(grads), axis_name=axis_name)

        # Apply update through TrainState
        new_state = state.replace(rng_key=new_rng)
        new_state = new_state.apply_gradients(grads)

        tokens_this_step = grad_accum * batch.shape[1] * batch.shape[2] * num_devices
        new_state = new_state.update_tokens(tokens_this_step)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "tokens_processed": jnp.array(new_state.tokens_processed),
        }

        return new_state, metrics

    return jax.pmap(
        train_step,
        axis_name=axis_name,
        donate_argnums=(0, 1),
    )


# ------------------------------------------------------------
# EVAL STEP
# ------------------------------------------------------------

def create_eval_step(axis_name: str = "batch") -> Callable:
    """
    Create a pmapped evaluation step.
    """

    def eval_step(state: TrainState, batch: Batch):
        inputs, targets = shift_tokens(batch)

        outputs = state.apply_fn(
            {"params": state.params},
            inputs,
            train=False,
        )

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss, metrics = compute_loss(logits, targets)

        metrics = dict(metrics)
        metrics["loss"] = loss
        return metrics

    return jax.pmap(eval_step, axis_name=axis_name)
