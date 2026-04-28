import numpy as np
from typing import List, Union


# ------------------------------------------------------------
# Memmap Token Dataset
# ------------------------------------------------------------

class MemmapTokenDataset:
    """
    High-throughput dataset for tokenized `.bin` shards.

    Design:
    -------
    - memory-mapped (zero-copy)
    - vectorized sampling
    - multi-shard aware
    - infinite stream

    Output:
    -------
    np.ndarray of shape:
        [global_batch_size, seq_len]
    dtype:
        int32 (JAX-compatible)
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        seq_len: int,
        global_batch_size: int,
        seed: int = 42,
    ):
        if isinstance(paths, str):
            paths = [paths]

        self.shards = [
            np.memmap(p, dtype=np.uint16, mode="r")
            for p in paths
        ]

        self.shard_lengths = np.array([len(s) for s in self.shards])
        self.total_tokens = int(self.shard_lengths.sum())

        self.seq_len = seq_len
        self.batch_size = global_batch_size

        if self.batch_size <= 0:
            raise ValueError("global_batch_size must be > 0")

        self.rng = np.random.default_rng(seed)

        # Precompute offsets
        self._offsets = np.arange(self.seq_len)

        print(
            f"[dataset] loaded {self.total_tokens:,} tokens "
            f"across {len(self.shards)} shards"
        )
        print(f"[dataset] global_batch_size = {self.batch_size:,}")

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------

    def sample(self) -> np.ndarray:
        """
        Sample one batch:
            [global_batch_size, seq_len]
        """

        shard_ids = self.rng.integers(
            0,
            len(self.shards),
            size=self.batch_size,
        )

        lengths = self.shard_lengths[shard_ids]

        # safe max offsets
        max_offsets = np.maximum(lengths - self.seq_len - 1, 1)

        start_positions = (
            self.rng.random(self.batch_size) * max_offsets
        ).astype(np.int64)

        indices = start_positions[:, None] + self._offsets[None, :]

        # group by shard for efficient reads
        unique_shards, inverse = np.unique(shard_ids, return_inverse=True)

        batch = np.empty(
            (self.batch_size, self.seq_len),
            dtype=np.uint16,
        )

        for group_idx, shard_id in enumerate(unique_shards):
            mask = (inverse == group_idx)

            if not np.any(mask):
                continue

            batch[mask] = self.shards[shard_id][indices[mask]]

        # convert → int32 (required by JAX)
        return np.ascontiguousarray(batch, dtype=np.int32)

    # ------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------

    def __iter__(self):
        while True:
            yield self.sample()
