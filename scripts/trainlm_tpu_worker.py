"""Single-VM v5e-8 entry point. Importing this module does not initialize XLA."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path


def event(stage: str, **values) -> None:
    print(json.dumps({"stage": stage, "pid": os.getpid(), **values}), flush=True)


def configure_environment() -> None:
    # Only the launcher parent calls this. Spawned workers inherit the topology
    # that PJRT assigns; they must never clear their rank/chip settings.
    os.environ.update(PJRT_DEVICE="TPU", USE_TF="0", USE_FLAX="0", USE_TORCH="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                      PYTHONFAULTHANDLER="1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for name in (
        "XRT_TPU_CONFIG", "TPU_PROCESS_ADDRESSES", "JAX_TPU_PROCESS_ADDRESSES",
        "CLOUD_TPU_TASK_ID", "TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_VISIBLE_CHIPS", "TPU_NUM_DEVICES", "PJRT_LOCAL_PROCESS_RANK",
        "PJRT_LOCAL_PROCESS_COUNT", "XLA_USE_SPMD", "XLA_USE_BF16",
        "XLA_DOWNCAST_BF16",
    ):
        os.environ.pop(name, None)


def run_worker(index, args, shards) -> None:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    torch.set_num_threads(1)
    rank, world = int(xr.global_ordinal()), int(xr.world_size())
    event("worker_entered", rank=rank, world_size=world)
    if world != 8:
        raise RuntimeError(f"Expected DP8, got world_size={world}; no fallback is enabled.")
    # Cache must be configured before the first tensor computation, including probe.
    xr.initialize_cache(str(Path(args.cache_dir) / f"rank-{rank}"))
    device = torch_xla.device()
    total = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(float(rank + 1), device=device))
    torch_xla.sync(wait=True)
    if total.item() != 36.0:
        raise RuntimeError("DP8 collective probe failed (expected rank sum 36).")
    event("probe_passed", rank=rank, world_size=world, device=str(device))
    if args.probe_only:
        xm.rendezvous("trainlm-probe-finished")
        return
    from trainlm_tpu_training import train_fn
    train_fn(index, args, shards)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--micro-batch-per-device", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--cache-dir", default="/tmp/trainlm_xla_cache")
    parser.add_argument("--output-dir", default="runs/trainlm_v5e8")
    parser.add_argument("--manifest-dir", default="data/packed/train")
    parser.add_argument("--data-mode", choices=("local", "hf", "raw"), default="local")
    parser.add_argument("--bin-path", action="append", default=[])
    parser.add_argument("--header-bytes", type=int)
    parser.add_argument("--token-dtype", choices=("uint16", "uint32", "int32", "int64"), default="uint16")
    parser.add_argument("--byte-order", choices=("little", "big"), default="little")
    parser.add_argument("--token-vocab-size", type=int, default=32011)
    parser.add_argument("--expected-token-count", type=int)
    parser.add_argument("--dataset-repo", default="LaughTaleAI/LaughLM-Tokenized-Fine")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--dataset-root", default="laughlm-v1")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--manifest-template", default="{root}/laughlm-v1_shard_{index:05d}.manifest.json")
    parser.add_argument("--hf-cache-dir", default="/tmp/laughlm_hf_cache")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--export-hf", action="store_true")
    args = parser.parse_args()
    for name in ("max_steps", "gradient_accumulation_steps", "micro_batch_per_device",
                 "sequence_length", "log_every_steps", "shard_count", "token_vocab_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sequence_length < 2 or args.warmup_steps < 0:
        parser.error("sequence length must be >=2 and warmup steps >=0")
    if args.data_mode == "raw" and not args.probe_only:
        if not args.bin_path or args.header_bytes is None or args.header_bytes < 0:
            parser.error("raw mode requires --bin-path and explicit nonnegative --header-bytes")
    if args.expected_token_count is not None and args.expected_token_count < 1:
        parser.error("--expected-token-count must be positive")
    return args


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    configure_environment()
    memory = Path("/proc/meminfo")
    if memory.exists():
        event("host_memory", meminfo=[
            line for line in memory.read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:"))
        ])
    event("import_xla")
    import torch_xla
    event("xla_imported")
    shards = None
    if not args.probe_only:
        event("data_preflight")
        # Validate once on the host, then pass small immutable descriptors to
        # workers. Never scan every multi-GB shard eight times simultaneously.
        from trainlm_tpu_data import _shards
        shards = _shards(args)
        event("data_validated", shards=len(shards),
              tokens=sum(s.manifest.token_count for s in shards))
    event("launch_dp8")
    torch_xla.launch(run_worker, args=(args, shards), start_method="spawn")
    event("all_workers_finished")


if __name__ == "__main__":
    main()
