from flax import struct
from typing import Any, Callable

import optax


# ------------------------------------------------------------
# TrainState
# ------------------------------------------------------------

@struct.dataclass
class TrainState:
    """
    Full training state (Flax-compatible, checkpointable).
    """

    step: int

    params: Any
    opt_state: optax.OptState

    tx: optax.GradientTransformation
    apply_fn: Callable

    rng_key: Any

    tokens_processed: int = 0

    # --------------------------------------------------------
    # Apply gradients
    # --------------------------------------------------------

    def apply_gradients(self, grads):
        """
        Apply gradients and return updated state.
        """

        updates, new_opt_state = self.tx.update(
            grads,
            self.opt_state,
            self.params,
        )

        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

    # --------------------------------------------------------
    # Update tokens processed
    # --------------------------------------------------------

    def update_tokens(self, num_tokens: int):
        return self.replace(
            tokens_processed=self.tokens_processed + num_tokens
        )

    # --------------------------------------------------------
    # RNG handling
    # --------------------------------------------------------

    def next_rng(self):
        """
        Split RNG key for next step.
        """
        import jax

        new_key, subkey = jax.random.split(self.rng_key)
        return self.replace(rng_key=new_key), subkey
