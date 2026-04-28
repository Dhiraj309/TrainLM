from typing import Callable

import optax


# ------------------------------------------------------------
# Utility: compute total training steps
# ------------------------------------------------------------

def compute_total_steps(runtime_cfg, num_devices: int) -> int:
    """
    Derive total optimizer steps from runtime config.

    tokens_per_step =
        seq_len × micro_batch_per_device × gradient_accumulation × num_devices

    total_steps = total_tokens // tokens_per_step
    """
    tokens_per_step = (
        runtime_cfg.seq_len
        * runtime_cfg.micro_batch_per_device
        * runtime_cfg.gradient_accumulation
        * num_devices
    )

    if tokens_per_step <= 0:
        raise ValueError(
            f"Invalid tokens_per_step={tokens_per_step}. "
            "Check seq_len, micro_batch_per_device, gradient_accumulation, and num_devices."
        )

    return max(runtime_cfg.total_tokens // tokens_per_step, 1)


# ------------------------------------------------------------
# Cosine decay with warmup
# ------------------------------------------------------------

def build_cosine_scheduler(config, num_devices: int) -> Callable:
    """
    Warmup -> cosine decay.
    """
    total_steps = compute_total_steps(config.runtime, num_devices)
    lr = config.optimizer.learning_rate
    min_ratio = config.scheduler.min_lr_ratio
    warmup = getattr(config.scheduler, "warmup_steps", None)

    if warmup is None:
        warmup_fraction = getattr(config.scheduler, "warmup_fraction", 0.01)
        warmup = int(total_steps * warmup_fraction)

    warmup = max(warmup, 0)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=lr * min_ratio,
    )


# ------------------------------------------------------------
# Linear warmup -> linear decay
# ------------------------------------------------------------

def build_linear_scheduler(config, num_devices: int) -> Callable:
    """
    Warmup -> linear decay to zero.
    """
    total_steps = compute_total_steps(config.runtime, num_devices)
    lr = config.optimizer.learning_rate
    warmup = getattr(config.scheduler, "warmup_steps", None)

    if warmup is None:
        warmup_fraction = getattr(config.scheduler, "warmup_fraction", 0.01)
        warmup = int(total_steps * warmup_fraction)

    warmup = max(warmup, 0)
    decay_steps = max(total_steps - warmup, 1)

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=0.0,
        transition_steps=decay_steps,
    )

    return optax.join_schedules(
        schedules=[warmup_sched, decay_sched],
        boundaries=[warmup],
    )


# ------------------------------------------------------------
# Inverse square root
# ------------------------------------------------------------

def build_rsqrt_scheduler(config, num_devices: int) -> Callable:
    """
    Warmup then inverse square root decay.
    """
    lr = config.optimizer.learning_rate
    warmup = getattr(config.scheduler, "warmup_steps", None)

    if warmup is None:
        total_steps = compute_total_steps(config.runtime, num_devices)
        warmup_fraction = getattr(config.scheduler, "warmup_fraction", 0.01)
        warmup = int(total_steps * warmup_fraction)

    warmup = max(warmup, 1)

    def schedule(step: int):
        step = max(step, 1)
        scale = min(step ** -0.5, step * (warmup ** -1.5))
        return lr * scale

    return schedule


# ------------------------------------------------------------
# WSD: Warmup -> Stable -> Decay
# ------------------------------------------------------------

def build_wsd_scheduler(config, num_devices: int) -> Callable:
    """
    Warmup-Stable-Decay schedule.
    """
    lr = config.optimizer.learning_rate
    min_ratio = config.scheduler.min_lr_ratio
    total_steps = compute_total_steps(config.runtime, num_devices)

    warmup = getattr(config.scheduler, "warmup_steps", None)
    warmup_fraction = getattr(config.scheduler, "warmup_fraction", None)

    if warmup is None:
        if warmup_fraction is None:
            raise ValueError(
                "WSD scheduler requires either warmup_steps or warmup_fraction."
            )
        warmup = int(total_steps * warmup_fraction)

    warmup = max(warmup, 0)

    stable_fraction = getattr(config.scheduler, "stable_fraction", 0.88)
    if not (0.0 < stable_fraction < 1.0):
        raise ValueError(
            f"scheduler.stable_fraction must be in (0, 1), got {stable_fraction}."
        )

    stable_steps = int(total_steps * stable_fraction)
    decay_steps = getattr(config.scheduler, "decay_steps", None)

    if decay_steps is not None:
        decay_steps = int(decay_steps)
        stable_steps = total_steps - warmup - decay_steps
    else:
        decay_steps = total_steps - warmup - stable_steps

    if stable_steps < 0:
        raise ValueError(
            f"Computed stable_steps is negative ({stable_steps}). "
            "Check stable_fraction or decay_steps."
        )

    if decay_steps < 0:
        raise ValueError(
            f"Computed decay_steps is negative ({decay_steps}). "
            "Reduce warmup or stable_fraction."
        )

    if warmup + stable_steps >= total_steps:
        raise ValueError(
            f"Invalid schedule:\n"
            f"  warmup: {warmup}\n"
            f"  stable: {stable_steps}\n"
            f"  total:  {total_steps}\n"
            f"-> No room for decay."
        )

    min_lr = lr * min_ratio

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    stable_sched = optax.constant_schedule(lr)

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=min_lr,
        transition_steps=max(decay_steps, 1),
    )

    return optax.join_schedules(
        schedules=[warmup_sched, stable_sched, decay_sched],
        boundaries=[warmup, warmup + stable_steps],
    )


# ------------------------------------------------------------
# Main factory
# ------------------------------------------------------------

def build_scheduler(config, num_devices: int) -> Callable:
    """
    Build learning-rate schedule from trainlm.config.
    """
    sched_type = config.scheduler.type

    if sched_type == "cosine":
        return build_cosine_scheduler(config, num_devices)

    if sched_type == "linear":
        return build_linear_scheduler(config, num_devices)

    if sched_type == "rsqrt":
        return build_rsqrt_scheduler(config, num_devices)

    if sched_type == "wsd":
        return build_wsd_scheduler(config, num_devices)

    raise ValueError(
        f"Unknown scheduler type: '{sched_type}'. "
        f"Valid options: cosine, linear, rsqrt, wsd."
    )
