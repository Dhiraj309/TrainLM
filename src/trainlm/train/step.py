import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable

from trainlm.train.loss import shift_tokens, compute_loss


Params = Any
Batch = jnp.ndarray


# ------------------------------------------------------------
# TRAIN STEP (PMAP + TRUE GRAD ACCUMULATION)
# ------------------------------------------------------------

def create_train_step(
    model,
    grad_accum: int,
    axis_name: str = "batch",
) -> Callable:

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------
    def loss_fn(params: Params, micro_batch: Batch, rng):

        inputs, targets = shift_tokens(micro_batch)

        outputs = model(
            input_ids=inputs,
            params=params,
            train=True,
            dropout_rng=rng,
        )

        logits = outputs.logits
        loss, _ = compute_loss(logits, targets)

        return loss

    # --------------------------------------------------------
    # Train step (per device)
    # --------------------------------------------------------
    def train_step(state, batch):

        params = state.params
        opt_state = state.opt_state

        # --------------------------------------------
        # RNG split (important for dropout correctness)
        # --------------------------------------------
        state, step_rng = state.next_rng()

        # --------------------------------------------
        # Init gradient accumulator
        # --------------------------------------------
        grads_accum = jax.tree_util.tree_map(jnp.zeros_like, params)

        # --------------------------------------------
        # Gradient accumulation via scan
        # --------------------------------------------
        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry

            rng, subkey = jax.random.split(rng)

            loss, grads = jax.value_and_grad(loss_fn)(
                params,
                micro_batch,
                subkey,
            )

            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g,
                grads_accum,
                grads,
            )

            return (grads_accum, rng), loss

        (grads_accum, _), losses = jax.lax.scan(
            scan_fn,
            (grads_accum, step_rng),
            batch,
        )

        # --------------------------------------------
        # Average gradients
        # --------------------------------------------
        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(losses)

        # --------------------------------------------
        # Cross-device sync
        # --------------------------------------------
        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        # --------------------------------------------
        # Gradient norm (for logging / debugging)
        # --------------------------------------------
        grad_norm = optax.global_norm(grads)
        grad_norm = jax.lax.pmean(grad_norm, axis_name)

        # --------------------------------------------
        # Optimizer step
        # --------------------------------------------
        updates, new_opt_state = state.tx.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        # --------------------------------------------
        # Tokens processed (correct accounting)
        # --------------------------------------------
        tokens_in_step = (
            batch.shape[-1] # seq_len
            * batch.shape[1] # grad_accum
            * batch.shape[2] # micro_batch
        )

        new_state = state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
            tokens_processed=state.tokens_processed + tokens_in_step,
        )

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
        }

        return new_state, metrics

    # ------------------------------------------------------------
    # PMAP wrapper
    # ------------------------------------------------------------
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