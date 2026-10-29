"""CPU-safe optimization preflight shared by the launcher and TPU workers."""

from pathlib import Path

from trainlm.config import ModelSourceConfig
from trainlm.optimization.integration import (
    OptimizationSession, load_optimization_options, reject_unavailable_strict_options,
    write_optimization_report,
)


def model_source(args):
    if args.model_id:
        return ModelSourceConfig(provider="huggingface", initialization="pretrained",
            name_or_path=args.model_id, revision=args.model_revision or None,
            trust_remote_code=args.trust_remote_code, dtype="float32", use_safetensors=True)
    return ModelSourceConfig(provider="huggingface", initialization="config", model_type="llama", dtype="float32",
        config_overrides={"vocab_size": 32064, "hidden_size": 1024, "intermediate_size": 2816,
            "num_hidden_layers": 8, "num_attention_heads": 8, "num_key_value_heads": 8,
            "max_position_embeddings": args.sequence_length, "tie_word_embeddings": True,
            "use_cache": False, "_attn_implementation": "sdpa"})


def prepare_session(model, args):
    options = load_optimization_options(args.optimization_factory, policy=args.optimization_policy)
    session = OptimizationSession(model, options, backend="xla", precision=args.precision,
        batch_size=args.micro_batch_per_device, sequence_length=args.sequence_length,
        accumulation_steps=args.gradient_accumulation_steps, policy=args.optimization_policy,
        allow_fallbacks=(None if args.optimization_fallbacks is None else args.optimization_fallbacks == "allow"),
        source_provider="huggingface", world_size=args.expected_world_size,
        eval_batch_size=args.eval_batch_size if args.eval_manifest_dir else None)
    if args.optimization_plan_id is not None and session.plan.plan_id != args.optimization_plan_id:
        raise ValueError("Worker optimization plan differs from host preflight.")
    return session


def host_preflight(args):
    options = load_optimization_options(args.optimization_factory, policy=args.optimization_policy)
    reject_unavailable_strict_options(options, policy=args.optimization_policy,
        allow_fallbacks=(None if args.optimization_fallbacks is None else args.optimization_fallbacks == "allow"),
        precision=args.precision)
    from trainlm.model import load_huggingface_causal_lm
    loaded = load_huggingface_causal_lm(model_source(args))
    if hasattr(loaded.model.config, "use_cache"):
        loaded.model.config.use_cache = False
    session = prepare_session(loaded.model, args)
    session.apply(loaded.model)
    if session.applied is not None:
        session.applied.revert()
    report = session.report()
    report["status"] = "host_preflight_validated"
    write_optimization_report(Path(args.output_dir) / "optimization_preflight.json", report)
    return report


def apply_worker_profile(session, model, *, get_rng, set_rng):
    if not session.plan.transformations:
        return
    # Native module constructors may initialize temporary tensors on XLA even
    # when the validated weights overwrite them. Keep training's device seed.
    seed = get_rng()
    try:
        session.apply(model)
    finally:
        set_rng(seed)
