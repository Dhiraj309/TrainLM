from flax import struct
from typing import Any

import optax
import jax


@struct.dataclass
class TrainState:
    """
    Training state for distributed (pmap) training.

    Notes
    -----
    - params / opt_state are replicated across devices
    - tx is static (not part of pytree)
    - step is global optimizer step
    """

    # ------------------------------------------------------------
    # Core state
    # ------------------------------------------------------------

    step: int

    params: Any
    opt_state: optax.OptState

    # Optimizer (static, not replicated as pytree node)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # RNG
    rng_key: Any

    # Tracking
    tokens_processed: int = 0

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    def update_tokens(self, num_tokens: int):
        """
        Update token counter.
        """
        return self.replace(
            tokens_processed=self.tokens_processed + num_tokens
        )

    def next_rng(self):
        """
        Split RNG safely.
        """
        new_key, subkey = jax.random.split(self.rng_key)
        return self.replace(rng_key=new_key), subkey
