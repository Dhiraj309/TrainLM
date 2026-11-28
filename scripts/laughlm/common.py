"""Shared configuration and checkpoint helpers for LaughLM 135M stages."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

DEFAULT_SHARD_REPO = "Dhiraj45/laughlm-135m-shards-v2"
DEFAULT_OUTPUT_REPO = "Dhiraj45/laughlm-135m-v2"
DEFAULT_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
VOCAB_SIZE = 49152
EOS_TOKEN_ID = 0
BOS_TOKEN_ID = 0


def build_135m_config(
    *,
    max_position_embeddings: int = 4096,
    rope_theta: float = 100000.0,
) -> Any:
    """Return the locked 30-layer deep-narrow 135M SmolLM2-compatible config."""
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.041666666666666664,
        rms_norm_eps=1e-5,
        use_cache=False,
        tie_word_embeddings=True,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=EOS_TOKEN_ID,
        rope_theta=rope_theta,
    )


def load_or_init_135m_model(
    *,
    init_checkpoint: str | Path | None = None,
    max_position_embeddings: int = 4096,
    rope_theta: float = 100000.0,
) -> Any:
    """Initialize a fresh 135M model or load weights from a snapshot or TrainLM checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM

    if init_checkpoint is not None:
        ckpt_path = Path(init_checkpoint)
        model_dir = ckpt_path / "model" if (ckpt_path / "model").is_dir() else ckpt_path
        trainer_state = ckpt_path / "trainer_state.pt"
        if (model_dir / "config.json").is_file():
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))
        elif trainer_state.is_file():
            cfg = build_135m_config(
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
            )
            model = AutoModelForCausalLM.from_config(cfg)
            payload = torch.load(trainer_state, map_location="cpu", weights_only=False)
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in payload["model"].items()
            }
            model.load_state_dict(state_dict, strict=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))
        model.config.max_position_embeddings = max(
            int(getattr(model.config, "max_position_embeddings", 0) or 0),
            max_position_embeddings,
        )
        model.config.rope_theta = float(rope_theta)
        model.config.vocab_size = VOCAB_SIZE
        model.config.eos_token_id = EOS_TOKEN_ID
        model.config.use_cache = False
        return model

    cfg = build_135m_config(
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
    )
    return AutoModelForCausalLM.from_config(cfg)


def build_135m_loop_config(
    *,
    max_position_embeddings: int = 2048,
    rope_theta: float = 100000.0,
    use_pallas_recurrent_norm: bool = True,
    deep_init_scale: bool = False,
    loop_depth_schedule: tuple[int, ...] | None = None,
    pack_doc_boundary_mask: bool = False,
    use_value_residual: bool = False,
    use_qk_norm: bool = False,
    use_attn_output_gate: bool = False,
) -> Any:
    """Return the LaughLM-Loop-135M preset aligned for v5e-1 MXUs."""
    from trainlm.config import LaughLMLoopConfig

    return LaughLMLoopConfig.laughlm_loop_135m(
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        use_pallas_recurrent_norm=use_pallas_recurrent_norm,
        deep_init_scale=deep_init_scale,
        loop_depth_schedule=loop_depth_schedule,
        pack_doc_boundary_mask=pack_doc_boundary_mask,
        use_value_residual=use_value_residual,
        use_qk_norm=use_qk_norm,
        use_attn_output_gate=use_attn_output_gate,
    )


def load_or_init_loop_135m_model(
    *,
    init_checkpoint: str | Path | None = None,
    max_position_embeddings: int = 2048,
    rope_theta: float = 100000.0,
    use_pallas_recurrent_norm: bool = True,
    deep_init_scale: bool = False,
    loop_depth_schedule: tuple[int, ...] | None = None,
    pack_doc_boundary_mask: bool = False,
    use_value_residual: bool = False,
    use_qk_norm: bool = False,
    use_attn_output_gate: bool = False,
) -> Any:
    """Initialize a fresh LaughLM-Loop-135M model or load from checkpoint."""
    from trainlm.model import LaughLMLoopForCausalLM

    if init_checkpoint is not None:
        ckpt_path = Path(init_checkpoint)
        model_dir = ckpt_path / "model" if (ckpt_path / "model").is_dir() else ckpt_path
        model = LaughLMLoopForCausalLM.from_pretrained(str(model_dir))
        model.config.use_pallas_recurrent_norm = bool(use_pallas_recurrent_norm)
        if pack_doc_boundary_mask:
            model.config.pack_doc_boundary_mask = True
        if loop_depth_schedule is not None:
            model.config.loop_depth_schedule = tuple(loop_depth_schedule)
        model.config.use_cache = False
        return model

    cfg = build_135m_loop_config(
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        use_pallas_recurrent_norm=use_pallas_recurrent_norm,
        deep_init_scale=deep_init_scale,
        loop_depth_schedule=loop_depth_schedule,
        pack_doc_boundary_mask=pack_doc_boundary_mask,
        use_value_residual=use_value_residual,
        use_qk_norm=use_qk_norm,
        use_attn_output_gate=use_attn_output_gate,
    )
    return LaughLMLoopForCausalLM(cfg)


def load_tokenizer(tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> Any:
    """Load the SmolLM2 tokenizer with verified vocab_size=49152 and eos_token_id=0."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if getattr(tok, "eos_token_id", None) is None:
        tok.eos_token_id = EOS_TOKEN_ID
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    if len(tok) != VOCAB_SIZE:
        raise ValueError(f"Expected vocab_size={VOCAB_SIZE}, got {len(tok)}.")
    if int(tok.eos_token_id) != EOS_TOKEN_ID:
        raise ValueError(f"Expected eos_token_id={EOS_TOKEN_ID}, got {tok.eos_token_id}.")
    return tok


def make_synthetic_dataset(
    num_examples: int,
    seq_len: int,
    *,
    vocab_size: int = VOCAB_SIZE,
) -> list[dict[str, Any]]:
    """Build deterministic CPU synthetic token batches for zero-cost dry runs."""
    import torch

    gen = torch.Generator().manual_seed(42)
    ids = torch.randint(0, vocab_size, (num_examples, seq_len), generator=gen)
    return [{"input_ids": ids[i], "labels": ids[i]} for i in range(num_examples)]


def detect_accelerator_kind() -> str:
    """Return 'cuda', 'tpu', or 'cpu' without calling internal TorchTPU APIs."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_tpu  # noqa: F401
    except ModuleNotFoundError:
        torch_tpu = None
    if torch_tpu is not None:
        try:
            from trainlm.runtime import discover_tpu

            report = discover_tpu(torch, verify_allocation=False, include_properties=False)
            if report.get("device_type") == "tpu" and report.get("status") in {
                "ready",
                "available_unverified",
            }:
                return "tpu"
        except Exception:
            pass
    return "cpu"


def resolve_optim(accel_kind: str) -> str:
    """Select the fastest validated optimizer for the active accelerator."""
    if accel_kind == "tpu":
        return "adamw_bf16"
    return "adamw_torch"


def resolve_logits_chunk_size(requested: int, *, accel_kind: str) -> int:
    """Bound per-chunk logits memory on 16 GB TPUs while keeping large chunks on GPUs."""
    if accel_kind == "tpu" and requested >= 16384:
        return 2048
    return requested


def resolve_loss_rematerialization(
    requested: str,
    *,
    cuda_available: bool = False,
    accel_kind: str | None = None,
) -> str:
    """Disable per-chunk remat on high-VRAM CUDA/ROCm GPUs; keep per_chunk on TPU."""
    if requested != "auto":
        return requested
    kind = accel_kind or ("cuda" if cuda_available else "cpu")
    if kind == "cuda":
        return "disabled"
    if kind == "tpu":
        return "per_chunk"
    return "auto"


def atomic_save_model_snapshot(model: Any, tokenizer: Any | None, target_dir: Path) -> Path:
    """Save unwrapped model (+ optional tokenizer) to `.tmp` and atomically rename."""
    unwrapped = getattr(model, "_orig_mod", model)
    unwrapped = getattr(unwrapped, "module", unwrapped)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target_dir.parent / f".{target_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        unwrapped.save_pretrained(tmp_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_dir)
        if target_dir.exists():
            backup = target_dir.parent / f".{target_dir.name}.bak.{os.getpid()}"
            os.replace(target_dir, backup)
            os.replace(tmp_dir, target_dir)
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return target_dir


def extract_cli_option(argv: list[str] | None, names: tuple[str, ...]) -> str | None:
    """Read an option value from ``argv`` without importing heavy ML packages."""
    if not argv:
        return None
    for idx, token in enumerate(argv):
        for name in names:
            if token == name and idx + 1 < len(argv):
                return argv[idx + 1]
            if token.startswith(f"{name}="):
                return token.split("=", 1)[1]
    return None


def configure_preimport_tpu_compile_cache(
    cache_dir: str | Path | None = None,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Export XLA/Tier-3 compile cache env vars before importing ``torch`` or ``torch_tpu``."""
    env = os.environ if base_env is None else base_env
    raw = (
        str(cache_dir).strip()
        if cache_dir is not None and str(cache_dir).strip()
        else (
            env.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
            or env.get("XLA_COMPILATION_CACHE_PATH")
            or env.get("TORCH_TPU_CACHE_DIR")
            or str(Path.home() / ".cache" / "trainlm" / "tpu_compile")
        )
    )
    if raw.startswith("gs://"):
        resolved = raw
    else:
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        resolved = str(path)
    env.setdefault("TORCH_TPU_CACHE_DIR", resolved)
    env.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", resolved)
    env.setdefault("XLA_COMPILATION_CACHE_PATH", resolved)
    return env


def resolve_shard_range(
    shard_start: int = 0,
    shard_stop: int | None = None,
    max_shards: int | None = None,
) -> tuple[int, int] | None:
    """Resolve end-exclusive ``(start, stop)`` shard indices while preserving ``--max-shards``."""
    if isinstance(shard_start, bool) or not isinstance(shard_start, int) or shard_start < 0:
        raise ValueError("shard_start must be a non-negative integer.")
    if shard_stop is not None:
        if isinstance(shard_stop, bool) or not isinstance(shard_stop, int) or shard_stop <= shard_start:
            raise ValueError("shard_stop must be greater than shard_start.")
        return (shard_start, shard_stop)
    if max_shards is not None:
        if isinstance(max_shards, bool) or not isinstance(max_shards, int) or max_shards < 1:
            raise ValueError("max_shards must be a positive integer.")
        if shard_start > 0:
            return (shard_start, shard_start + max_shards)
        return None
    if shard_start > 0:
        raise ValueError("Specify --shard-stop or --max-shards when --shard-start > 0.")
    return None


def parse_loop_depth_schedule(raw: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated loop depths (e.g. ``'2,4,4,6'``) with static cache bounds."""
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("loop_depth_schedule cannot be empty.")
    values = tuple(int(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError("loop_depth_schedule values must be positive integers.")
    if len(set(values)) > 6:
        raise ValueError(
            "loop_depth_schedule supports at most 6 unique depths to bound XLA compilation cache."
        )
    return values

