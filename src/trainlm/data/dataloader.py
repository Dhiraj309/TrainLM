from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from trainlm.data.dataset import MemmapTokenDataset


# ------------------------------------------------------------
# Dataloader wrapper
# ------------------------------------------------------------

class TokenDataLoader:
    """
    Thin iterable wrapper around MemmapTokenDataset.

    The dataset already yields full batches:
        [global_batch_size, seq_len]

    This wrapper exists to keep the public API clean and to make it easy
    to add prefetching, shuffling policies, or distributed sampling later.
    """

    def __init__(
        self,
        dataset: MemmapTokenDataset,
        steps_per_epoch: Optional[int] = None,
    ):
        self.dataset = dataset
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.steps_per_epoch is None:
            yield from self.dataset
            return

        for _ in range(self.steps_per_epoch):
            yield self.dataset.sample()


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_train_dataloader(
    paths,
    seq_len: int,
    global_batch_size: int,
    seed: int = 42,
    steps_per_epoch: Optional[int] = None,
) -> TokenDataLoader:
    """
    Build a training dataloader for token shards.
    """
    dataset = MemmapTokenDataset(
        paths=paths,
        seq_len=seq_len,
        global_batch_size=global_batch_size,
        seed=seed,
    )
    return TokenDataLoader(
        dataset=dataset,
        steps_per_epoch=steps_per_epoch,
    )
