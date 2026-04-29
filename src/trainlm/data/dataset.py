import numpy as np
from typing import List, Union


class MemmapTokenDataset:
    """
    Safe memmap dataset:
    - NO full concatenation (avoids RAM explosion)
    - supports multiple shards
    - mostly vectorized sampling
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

        # ✅ keep memmaps separate (NO concatenate)
        self.arrays = [
            np.memmap(p, dtype=np.uint16, mode="r")
            for p in paths
        ]

        self.lengths = [len(a) for a in self.arrays]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_tokens = self.cum_lengths[-1]

        self.seq_len = seq_len
        self.batch_size = global_batch_size
        self.rng = np.random.default_rng(seed)

        self._offsets = np.arange(self.seq_len)

        print(f"[dataset] total tokens: {self.total_tokens:,}")
        print(f"[dataset] global_batch_size = {self.batch_size:,}")

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------

    def sample(self) -> np.ndarray:
        ix = self.rng.integers(
            0,
            self.total_tokens - self.seq_len - 1,
            size=self.batch_size,
        )

        batch = np.empty((self.batch_size, self.seq_len), dtype=np.uint16)

        for i, start in enumerate(ix):
            shard_idx = np.searchsorted(
                self.cum_lengths, start, side="right"
            )

            if shard_idx == 0:
                local_start = start
            else:
                local_start = start - self.cum_lengths[shard_idx - 1]

            arr = self.arrays[shard_idx]

            batch[i] = arr[local_start : local_start + self.seq_len]

        return batch.astype(np.int32, copy=False)

    # ------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------

    def __iter__(self):
        while True:
            yield self.sample()