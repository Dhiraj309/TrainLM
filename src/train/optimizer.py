from typing import Any, Callable

import optax
from flax import traverse_util


# ------------------------------------------------------------
# Weight decay mask (STATIC TREE)
# ------------------------------------------------------------

def create_weight_decay_mask(params: Any) -> Any:
    """
    Build a mask tree for weight decay.

    True  = apply weight decay
    False = exclude

    Excludes:
      - bias
      - norm scale (RMSNorm / LayerNorm)
      - embeddings (optional safety)
    """

    flat = traverse_util.flatten_dict(params)

    no_decay = {"bias", "scale", "embedding"}

    mask_flat = {
        k: (k[-1] not in no_decay)
        for k in flat
    }

    return traverse_util.unflatten_dict(mask_flat)


# ------------------------------------------------------------
# AdamW (manual, frontier-style)
# ------------------------------------------------------------

def build_adamw(
    optimizer_cfg,
    schedule: Callable,
    params: Any,
) -> optax.GradientTransformation:
    """
    AdamW with:
    - gradient clipping
    - Adam moments
    - masked weight decay
    - LR schedule
    """

    mask = create_weight_decay_mask(params)

    return optax.chain(
        # 1. Gradient clipping
        optax.clip_by_global_norm(optimizer_cfg.grad_clip),

        # 2. Adam moments
        optax.scale_by_adam(
            b1=optimizer_cfg.beta1,
            b2=optimizer_cfg.beta2,
            eps=optimizer_cfg.eps,
        ),

        # 3. Weight decay (masked)
        optax.add_decayed_weights(
            weight_decay=optimizer_cfg.weight_decay,
            mask=mask,
        ),

        # 4. LR schedule
        optax.scale_by_learning_rate(schedule),
    )


# ------------------------------------------------------------
# Adafactor
# ------------------------------------------------------------

def build_adafactor(
    optimizer_cfg,
    schedule: Callable,
) -> optax.GradientTransformation:

    return optax.chain(
        optax.clip_by_global_norm(optimizer_cfg.grad_clip),
        optax.adafactor(
            learning_rate=schedule,
        ),
    )


# ------------------------------------------------------------
# Lion
# ------------------------------------------------------------

def build_lion(
    optimizer_cfg,
    schedule: Callable,
) -> optax.GradientTransformation:

    return optax.chain(
        optax.clip_by_global_norm(optimizer_cfg.grad_clip),
        optax.lion(
            learning_rate=schedule,
            b1=optimizer_cfg.beta1,
            b2=optimizer_cfg.beta2,
        ),
    )


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_optimizer(
    optimizer_cfg,
    schedule: Callable,
    params: Any,
) -> optax.GradientTransformation:
    """
    Build optimizer.

    NOTE:
    params required for weight decay mask construction.
    """

    opt_type = optimizer_cfg.type

    if opt_type == "adamw":
        return build_adamw(optimizer_cfg, schedule, params)

    if opt_type == "adafactor":
        return build_adafactor(optimizer_cfg, schedule)

    if opt_type == "lion":
        return build_lion(optimizer_cfg, schedule)

    if opt_type == "muon":
        raise NotImplementedError("Muon not implemented")

    raise ValueError(f"Unknown optimizer type: {opt_type}")
