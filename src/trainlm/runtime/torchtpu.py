"""Native TorchTPU execution backend (single-device and multi-chip distributed)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import os
from typing import Any

import torch
from torch import nn

from .base import BackendDiagnostics, Precision
from .runtime import TorchRuntime
from .torchtpu_discovery import discover_tpu


def _device_rng_getter() -> Callable[[], Any] | None:
    """Resolve TorchTPU RNG capture through the public API for this PyTorch."""
    accelerator = getattr(torch, "accelerator", None)
    random_api = getattr(accelerator, "random", None)
    getter = getattr(random_api, "get_rng_state", None)
    if callable(getter):
        return getter

    getter = getattr(accelerator, "get_rng_state", None)
    if callable(getter):
        return getter

    if random_api is None:
        getter = getattr(getattr(torch, "tpu", None), "get_rng_state", None)
        if callable(getter):
            return getter
    return None


def _device_rng_setter() -> Callable[[Any], None] | None:
    """Resolve TorchTPU RNG restore through a supported public API."""
    accelerator = getattr(torch, "accelerator", None)
    random_api = getattr(accelerator, "random", None)
    setter = getattr(random_api, "set_rng_state", None)
    if callable(setter):
        return setter

    setter = getattr(getattr(torch, "tpu", None), "set_rng_state", None)
    if callable(setter):
        return setter

    setter = getattr(accelerator, "set_rng_state", None)
    if callable(setter):
        return setter
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} must be an integer.") from exc


def _resolve_tpu_device(local_rank: int | None) -> torch.device:
    if local_rank is None:
        return torch.device("tpu")
    try:
        return torch.device("tpu", local_rank)
    except (TypeError, ValueError):
        return torch.device("tpu")


class TorchTPURuntime(TorchRuntime):
    """Native TorchTPU runtime supporting single-chip and multi-chip (v5e-8/v6e-8) launches."""

    def __init__(
        self,
        *,
        precision: Precision = "fp32",
        seed: int = 42,
        compile_enabled: bool = False,
        allow_excess_precision: bool | None = None,
        compile_cache_dir: str | None = None,
        distributed_strategy: str = "auto",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if allow_excess_precision is not None and not isinstance(
            allow_excess_precision, bool
        ):
            raise TypeError("allow_excess_precision must be boolean or None.")
        if compile_cache_dir is not None and (
            not isinstance(compile_cache_dir, str) or not compile_cache_dir.strip()
        ):
            raise ValueError("compile_cache_dir must be a non-empty path string or None.")
        if distributed_strategy not in {"auto", "ddp", "fsdp2", "none"}:
            raise ValueError(f"Unsupported distributed_strategy: {distributed_strategy!r}")

        super().__init__(
            device="cpu", precision=precision, compile_enabled=compile_enabled
        )
        self._seed = seed
        self._allow_excess_precision = allow_excess_precision
        self._compile_cache_dir = compile_cache_dir
        self._distributed_strategy = distributed_strategy
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0
        self._dist_backend: str | None = None
        self._topology_mesh: Any = None

        tpu_mod = getattr(torch, "tpu", None)
        tpu_dist = getattr(tpu_mod, "distributed", None)
        dist_mod = getattr(torch, "distributed", None)
        dist_is_avail = bool(
            dist_mod is not None
            and callable(getattr(dist_mod, "is_available", None))
            and dist_mod.is_available()
        )
        dist_initialized = bool(
            dist_is_avail
            and callable(getattr(dist_mod, "is_initialized", None))
            and dist_mod.is_initialized()
        )

        env_world_size = _env_int("WORLD_SIZE", 1)
        env_local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
        env_rank = _env_int("RANK", 0)
        env_local_rank = _env_int("LOCAL_RANK", env_rank)
        is_multi_rank_launch = (
            dist_initialized or env_world_size > 1 or env_local_world_size > 1
        )

        if is_multi_rank_launch:
            set_launch_env = getattr(tpu_dist, "set_tpu_launch_env", None)
            if callable(set_launch_env):
                set_launch_env()
            set_device_fn = getattr(tpu_mod, "set_device", None)
            if callable(set_device_fn):
                set_device_fn(env_local_rank)

        self._native_report = discover_tpu(
            torch, verify_allocation=True, include_properties=True
        )
        if self._native_report["status"] != "ready":
            detail = self._native_report.get("error_type") or self._native_report["status"]
            raise RuntimeError(
                "Native TorchTPU initialization failed before model transfer "
                f"({detail}). Check the installed torch/TorchTPU profile and "
                "attached TPU runtime."
            )

        visible = self._native_report.get("device_count_visible")
        if not is_multi_rank_launch and visible != 1:
            raise RuntimeError(
                "This experimental native TorchTPU runtime requires exactly "
                "one visible TPU device unless launched with WORLD_SIZE > 1 "
                "or LOCAL_WORLD_SIZE > 1."
            )

        if is_multi_rank_launch:
            target_world = (
                int(dist_mod.get_world_size())
                if dist_initialized and callable(getattr(dist_mod, "get_world_size", None))
                else max(env_world_size, env_local_world_size)
            )
            if not dist_initialized:
                if not dist_is_avail:
                    raise RuntimeError(
                        "Multi-chip TPU launch requires torch.distributed support."
                    )
                backend = (
                    getattr(tpu_dist, "BACKEND", None)
                    or getattr(tpu_dist, "backend", None)
                    or "tpu"
                )
                init_pg = getattr(dist_mod, "init_process_group", None)
                if not callable(init_pg):
                    raise RuntimeError("torch.distributed.init_process_group is unavailable.")
                init_pg(
                    backend=backend,
                    init_method="env://",
                    world_size=target_world,
                    rank=env_rank,
                )
                self._dist_backend = str(backend)
            else:
                get_backend = getattr(dist_mod, "get_backend", None)
                if callable(get_backend):
                    try:
                        self._dist_backend = str(get_backend())
                    except Exception:
                        self._dist_backend = None
            if callable(getattr(dist_mod, "get_world_size", None)):
                self._world_size = int(dist_mod.get_world_size())
            else:
                self._world_size = target_world
            if callable(getattr(dist_mod, "get_rank", None)):
                self._rank = int(dist_mod.get_rank())
            else:
                self._rank = env_rank
            self._local_rank = env_local_rank
            try:
                self._device = _resolve_tpu_device(self._local_rank)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Native TorchTPU allocation passed, but per-rank TPU device binding failed."
                ) from exc
        else:
            try:
                self._device = torch.device("tpu")
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Native TorchTPU allocation passed, but this PyTorch process "
                    "does not expose the registered 'tpu' device type."
                ) from exc

        mesh_probe = getattr(tpu_mod, "topology_aware_mesh", None)
        if callable(mesh_probe):
            try:
                self._topology_mesh = mesh_probe()
            except Exception:
                self._topology_mesh = "available"
        elif mesh_probe is not None:
            self._topology_mesh = mesh_probe

    @property
    def name(self) -> str:
        return "torchtpu"

    @property
    def world_size(self) -> int:
        return int(getattr(self, "_world_size", 1))

    @property
    def rank(self) -> int:
        return int(getattr(self, "_rank", 0))

    @property
    def local_rank(self) -> int:
        return int(getattr(self, "_local_rank", 0))

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary_process(self) -> bool:
        return self.rank == 0

    @property
    def mean_reduces_gradients_before_clipping(self) -> bool:
        return self.is_distributed

    def prepare_model(self, model: nn.Module) -> nn.Module:
        if self._precision == "bf16":
            model = model.to(device=self.device, dtype=torch.bfloat16)
        else:
            model = model.to(self.device)
        unwrapped = getattr(model, "module", model)
        if getattr(getattr(unwrapped, "config", None), "tie_word_embeddings", False):
            tie_fn = getattr(unwrapped, "tie_weights", None)
            if callable(tie_fn):
                tie_fn()
        return self.compile_model(model)

    def compile_model(self, model: nn.Module) -> nn.Module:
        if not self._compile_enabled or getattr(model, "_orig_mod", None) is not None:
            return model
        return _compile_with_tpu_options(model)

    @property
    def compile_optimizer_in_step(self) -> bool:
        """Avoid tracing Parameter-keyed Optimizer.state dicts in Dynamo on TPU."""
        return False

    @property
    def compile_stateless_optimizer_step(self) -> bool:
        """Compile clip_gradients and AdamW over flat tensor lists on TPU."""
        return bool(getattr(self, "_compile_enabled", False))

    def compile_training_step(self, step_fn):
        if not callable(step_fn):
            raise TypeError("step_fn must be callable.")
        if not self._compile_enabled:
            return step_fn
        return _compile_with_tpu_options(step_fn)

    def needs_explicit_grad_all_reduce(
        self, *, model: nn.Module, has_training_view: bool = False
    ) -> bool:
        """Return True when backward skipped DDP/FSDP2 reducer hooks on distributed TPU."""
        if not self.is_distributed:
            return False
        strategy = getattr(self, "_distributed_strategy", "auto")
        if strategy == "fsdp2":
            return False
        unwrapped = getattr(model, "_orig_mod", model)
        is_ddp = type(unwrapped).__name__ == "DistributedDataParallel"
        if is_ddp and not has_training_view:
            return False
        return True

    def all_reduce_gradients(self, grads: Iterable[torch.Tensor]) -> None:
        """Average gradient tensors across all TPU ranks via ICI/distributed collective."""
        if not self.is_distributed:
            return
        dist_mod = getattr(torch, "distributed", None)
        if (
            dist_mod is None
            or not callable(getattr(dist_mod, "is_initialized", None))
            or not dist_mod.is_initialized()
        ):
            return
        inv_world = 1.0 / float(self.world_size)
        op = getattr(getattr(dist_mod, "ReduceOp", None), "SUM", None)
        for grad in grads:
            if not isinstance(grad, torch.Tensor):
                continue
            if op is not None:
                dist_mod.all_reduce(grad, op=op)
            else:
                dist_mod.all_reduce(grad)
            grad.mul_(inv_world)

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be a torch.Tensor.")
        if not self.is_distributed:
            return value
        dist_mod = getattr(torch, "distributed", None)
        if (
            dist_mod is None
            or not callable(getattr(dist_mod, "is_initialized", None))
            or not dist_mod.is_initialized()
        ):
            return value
        out = value.clone()
        op = getattr(getattr(dist_mod, "ReduceOp", None), "SUM", None)
        if op is not None:
            dist_mod.all_reduce(out, op=op)
        else:
            dist_mod.all_reduce(out)
        return out

    def _configure_compile_cache(self) -> None:
        cache_dir = getattr(self, "_compile_cache_dir", None)
        if not cache_dir or not self._compile_enabled:
            return
        from pathlib import Path

        raw = str(cache_dir).strip()
        if raw.startswith("gs://"):
            resolved = raw
        else:
            path = Path(raw).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            resolved = str(path)
        os.environ.setdefault("TORCH_TPU_CACHE_DIR", resolved)
        os.environ.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", resolved)
        os.environ.setdefault("XLA_COMPILATION_CACHE_PATH", resolved)
        tpu_mod = getattr(torch, "tpu", None)
        for attr in ("set_compile_cache_dir", "set_compilation_cache_dir"):
            setter = getattr(tpu_mod, attr, None)
            if callable(setter):
                try:
                    setter(resolved)
                except Exception:
                    pass

    def on_train_begin(self) -> None:
        self._previous_eager_mode = None
        self._previous_matmul_precision = None
        self._previous_excess_precision = None
        self._configure_compile_cache()
        if getattr(self, "_precision", "fp32") == "bf16":
            getter_p = getattr(torch, "get_float32_matmul_precision", None)
            setter_p = getattr(torch, "set_float32_matmul_precision", None)
            if callable(setter_p):
                if callable(getter_p):
                    try:
                        self._previous_matmul_precision = getter_p()
                    except Exception:
                        self._previous_matmul_precision = None
                setter_p("medium")
        tpu_mod = getattr(torch, "tpu", None)
        allow_excess = getattr(self, "_allow_excess_precision", None)
        if allow_excess is not None and hasattr(
            tpu_mod, "allow_excess_precision"
        ):
            self._previous_excess_precision = getattr(
                tpu_mod, "allow_excess_precision"
            )
            setattr(tpu_mod, "allow_excess_precision", allow_excess)
        if self._compile_enabled:
            return
        eager_enum = getattr(tpu_mod, "EagerMode", None)
        setter = getattr(tpu_mod, "set_eager_mode", None)
        defer_and_fuse = getattr(eager_enum, "DEFER_AND_FUSE", None)
        if callable(setter) and defer_and_fuse is not None:
            getter = getattr(tpu_mod, "get_eager_mode", None)
            if callable(getter):
                try:
                    self._previous_eager_mode = getter()
                except Exception:
                    self._previous_eager_mode = None
            setter(defer_and_fuse)

    def on_train_end(self) -> None:
        tpu_mod = getattr(torch, "tpu", None)
        prev = getattr(self, "_previous_eager_mode", None)
        if prev is not None:
            setter = getattr(tpu_mod, "set_eager_mode", None)
            if callable(setter):
                try:
                    setter(prev)
                finally:
                    self._previous_eager_mode = None
        prev_excess = getattr(self, "_previous_excess_precision", None)
        if prev_excess is not None and hasattr(tpu_mod, "allow_excess_precision"):
            try:
                setattr(tpu_mod, "allow_excess_precision", prev_excess)
            finally:
                self._previous_excess_precision = None
        prev_matmul = getattr(self, "_previous_matmul_precision", None)
        if prev_matmul is not None:
            setter_p = getattr(torch, "set_float32_matmul_precision", None)
            if callable(setter_p):
                try:
                    setter_p(prev_matmul)
                finally:
                    self._previous_matmul_precision = None

    def initialize(self) -> None:
        torch.manual_seed(self._seed)

    def synchronize(self) -> None:
        accelerator = getattr(torch, "accelerator", None)
        synchronize = getattr(accelerator, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError(
                "This PyTorch version does not expose the supported generic "
                "accelerator synchronization API required by native TorchTPU."
            )
        synchronize()

    def barrier(self, name: str | None = None) -> None:
        del name
        self.synchronize()
        if self.is_distributed:
            dist_mod = getattr(torch, "distributed", None)
            if (
                dist_mod is not None
                and callable(getattr(dist_mod, "is_initialized", None))
                and dist_mod.is_initialized()
                and callable(getattr(dist_mod, "barrier", None))
            ):
                dist_mod.barrier()

    def state_dict(self) -> Mapping[str, Any]:
        """Capture the device RNG required by a resumable native checkpoint."""
        get_rng_state = _device_rng_getter()
        if not callable(get_rng_state):
            raise RuntimeError(
                "This PyTorch version does not expose supported accelerator "
                "RNG capture; "
                "a resumable native TPU checkpoint cannot be written."
            )
        return {
            "backend": self.name,
            "precision": self.precision,
            "world_size": self.world_size,
            "seed": self._seed,
            "compile_enabled": self._compile_enabled,
            "device_rng_state": get_rng_state(),
        }

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Check native checkpoint compatibility without changing device RNG."""
        if not isinstance(state_dict, Mapping):
            raise ValueError("Native TPU checkpoint state must be a mapping.")
        if state_dict.get("backend") != self.name:
            raise ValueError(
                "Native TPU resume requires a checkpoint created by the "
                "native TorchTPU backend."
            )
        if state_dict.get("world_size") != self.world_size:
            raise ValueError(
                f"Native TPU resume requires world_size={self.world_size} "
                f"(checkpoint has {state_dict.get('world_size')})."
            )
        if state_dict.get("precision") != self.precision:
            raise ValueError("Native TPU resume precision does not match the checkpoint.")
        if state_dict.get("compile_enabled", False) != self._compile_enabled:
            raise ValueError("Native TPU resume compile mode does not match the checkpoint.")
        if "device_rng_state" not in state_dict:
            raise ValueError("Native TPU checkpoint is missing device RNG state.")
        set_rng_state = _device_rng_setter()
        if not callable(set_rng_state):
            raise RuntimeError(
                "This PyTorch/TorchTPU profile does not expose supported TPU "
                "RNG restore; "
                "native TPU device RNG restoration is unavailable."
            )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        set_rng_state = _device_rng_setter()
        set_rng_state(state_dict["device_rng_state"])

    def diagnostics(self) -> BackendDiagnostics:
        base = super().diagnostics()
        values = dict(base.values)
        report = self._native_report or {}
        values.update(
            {
                "native_tpu_status": report.get("status", "not_initialized"),
                "device_count_visible": report.get("device_count_visible"),
                "device_name": report.get("device_name"),
                "torch_tpu_version": report.get("torch_tpu_version"),
                "libtpu_version": report.get("libtpu_version"),
                "dist_backend": getattr(self, "_dist_backend", None),
                "topology_mesh": (
                    str(getattr(self, "_topology_mesh", None))
                    if getattr(self, "_topology_mesh", None) is not None
                    else None
                ),
            }
        )
        return BackendDiagnostics(
            backend=base.backend,
            device_type=base.device_type,
            precision=base.precision,
            world_size=base.world_size,
            rank=base.rank,
            local_rank=base.local_rank,
            values=values,
        )


def _compile_with_tpu_options(target: Any) -> Any:
    from . import runtime as base_runtime

    compile_fn = base_runtime.torch.compile
    try:
        return compile_fn(target, backend="tpu", dynamic=False)
    except TypeError:
        return compile_fn(target)


__all__ = ["TorchTPURuntime"]
