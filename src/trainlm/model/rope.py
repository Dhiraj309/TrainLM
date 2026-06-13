import jax.numpy as jnp

from trainlm.model.config import LlamaConfig

def rotate_half(x: jnp.ndarray):
    x1, x2 = jnp.split(
        x,
        2,
        axis=-1,
    )

    return jnp.concatenate(
        (-x2, x1),
        axis=-1
    )

def build_rope(
    config: LlamaConfig,
    position_ids: jnp.ndarray,
    dtype: jnp.dtype,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    head_dim = config.head_dim

    theta = config.rope_parameters[
        "rope_theta"
    ]

    inv_freq = 1.0 / (
        theta ** (
            jnp.arange(
                0,
                head_dim,
                2, dtype=jnp.float32,
            )
            / head_dim
        )
    )

    freqs = jnp.einsum(
        "bt,d->btd",
        position_ids.astype(jnp.float32),
        inv_freq,
    )

    emb = jnp.concatenate(
        (freqs, freqs),
        axis=-1
    )

    cos = jnp.cos(emb).astype(dtype)
    sin = jnp.sin(emb).astype(dtype)

    return cos, sin

def apply_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]

    q = (
        q * cos
        + rotate_half(q) * sin
    )

    k = (
        k * cos
        + rotate_half(q) * sin
    )

    return q, k