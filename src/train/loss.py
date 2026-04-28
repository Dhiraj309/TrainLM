from typing import Optional, Dict, Tuple

import jax
import jax.numpy as jnp
import optax


# ------------------------------------------------------------
# Token shifting for causal LM
# ------------------------------------------------------------

def shift_tokens(input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shift token IDs for causal language modeling.

    input_ids: [batch, seq_len]
    returns:
        inputs  = [batch, seq_len - 1]
        targets = [batch, seq_len - 1]
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    return inputs, targets


# ------------------------------------------------------------
# Cross-entropy loss
# ------------------------------------------------------------

def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Token-level cross-entropy averaged over valid positions.

    logits:  [batch, seq_len, vocab]
    targets: [batch, seq_len]
    mask:    [batch, seq_len] with 1.0 for valid tokens, 0.0 for ignored tokens
    """
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets,
    )

    if mask is not None:
        per_token_loss = per_token_loss * mask
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        loss = jnp.sum(per_token_loss) / denom
    else:
        loss = jnp.mean(per_token_loss)

    return loss


# ------------------------------------------------------------
# Z-loss regularization
# ------------------------------------------------------------

def z_loss(
    logits: jnp.ndarray,
    coeff: float = 1e-4,
) -> jnp.ndarray:
    """
    Z-loss regularization to control logit magnitude.
    """
    if coeff == 0.0:
        return jnp.array(0.0, dtype=logits.dtype)

    log_z = jax.nn.logsumexp(logits, axis=-1)
    return coeff * jnp.mean(jnp.square(log_z))


# ------------------------------------------------------------
# Combined training loss
# ------------------------------------------------------------

def compute_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    zloss_coeff: float = 1e-4,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Returns:
        total_loss, metrics
    """
    ce = cross_entropy_loss(logits, targets, mask)
    zl = z_loss(logits, zloss_coeff)
    total = ce + zl

    metrics = {
        "cross_entropy": ce,
        "z_loss": zl,
        "total": total,
    }

    return total, metrics
