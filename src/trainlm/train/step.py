import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable

from trainlm.train.loss import shift_tokens, compute_loss


Params = Any
Batch = jnp.ndarray


# ------------------------------------------------------------
# TRAIN STEP
# ------------------------------------------------------------

def create_train_step(
    model,
    grad_accum: int,
    num_devices: int,
    axis_name: str = "batch",
) -> Callable:

    def loss_fn(params: Params, micro_batch: Batch):
        inputs, targets = shift_tokens(micro_batch)

        outputs = model(
            input_ids=inputs,
            params=params,
            train=True,
        )

        logits = outputs.logits
        loss, _ = compute_loss(logits, targets)

        return loss

    def train_step(state, batch):
        params = state.params
        opt_state = state.opt_state

        grads_accum = jax.tree_util.tree_map(jnp.zeros_like, params)

        def scan_fn(carry, micro_batch):
            grads_accum = carry

            loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)

            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g,
                grads_accum,
                grads,
            )

            return grads_accum, loss

        grads_accum, losses = jax.lax.scan(
            scan_fn,
            grads_accum,
            batch,
        )

        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(losses)

        # sync across devices
        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        updates, new_opt_state = state.tx.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        new_state = state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
        )

        metrics = {
            "loss": loss,
        }

        return new_state, metrics

    return jax.pmap(
        train_step,
        axis_name=axis_name,
        donate_argnums=(0,),
    )


# ------------------------------------------------------------
# EVAL STEP
# ------------------------------------------------------------

def create_eval_step(model):

    def eval_step(state, batch):
        inputs, targets = shift_tokens(batch)

        outputs = model(
            input_ids=inputs,
            params=state.params,
            train=False,
        )

        logits = outputs.logits
        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.pmap(eval_step, axis_name="batch")
