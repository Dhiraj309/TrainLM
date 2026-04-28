import numpy as np
from typing import List, Union


class MemmapTokenDataset:
    """
    Frontier-grade memmap dataset:
    - single contiguous token buffer
    - fully vectorized sampling
    - no Python loops in hot path
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

        # 🔥 concatenate shards ONCE (huge speed win)
        arrays = [
            np.memmap(p, dtype=np.uint16, mode="r")
            for p in paths
        ]

        self.tokens = np.concatenate(arrays)

        self.seq_len = seq_len
        self.batch_size = global_batch_size
        self.rng = np.random.default_rng(seed)

        self._offsets = np.arange(self.seq_len)

        print(f"[dataset] total tokens: {len(self.tokens):,}")
        print(f"[dataset] global_batch_size = {self.batch_size:,}")

    # ------------------------------------------------------------
    # Sampling (fully vectorized)
    # ------------------------------------------------------------

    def sample(self) -> np.ndarray:
        ix = self.rng.integers(
            0,
            len(self.tokens) - self.seq_len - 1,
            size=self.batch_size,
        )

        indices = ix[:, None] + self._offsets[None, :]

        # 🔥 single vectorized gather
        batch = self.tokens[indices]

        # no extra copy if already int32-compatible
        return batch.astype(np.int32, copy=False)

    # ------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------

    def __iter__(self):
        while True:
            yield self.sample()