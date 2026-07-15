"""
Thin adapter making a plain PyTorch `DataLoader` over `FindGoalOfflineDataset`
look like a `FlashbaxReplayBuffer` to `og_marl`'s `BaseOfflineSystem.train()`
-- the only method `train()` actually calls on its `replay_buffer` argument
is `.sample()`:

    # og_marl/baselines/base.py, BaseOfflineSystem.train()
    for _ in range(training_steps):
        ...
        experience = replay_buffer.sample()
        train_logs = self.train_step(experience)

`.sample()` is called once per training step, indefinitely -- there's no
epoch concept on the caller's side, matching Flashbax's infinite-stream,
sample-with-replacement semantics. A `DataLoader` doesn't support that
directly: it's an iterable that raises `StopIteration` once you've made
one pass over the dataset. `OfflineReplayBuffer` bridges that gap by
holding an internal iterator and transparently starting a new (freshly
shuffled) epoch whenever the current one runs out, so from
`BaseOfflineSystem.train()`'s point of view it's indistinguishable from
calling `FlashbaxReplayBuffer.sample()`.
"""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


class OfflineReplayBuffer:
    """
    Args:
        dataset: a `FindGoalOfflineDataset` (or anything with the same
            per-item shape -- this class doesn't depend on its internals,
            only on standard `Dataset` behavior).
        batch_size: matches Flashbax's `batch_size` param.
        num_workers: parallel HDF5-reading workers. `persistent_workers`
            defaults to True whenever `num_workers > 0`, so worker
            processes aren't respawned every epoch boundary -- offline RL
            runs typically call `.sample()` tens of thousands of times
            (e.g. `training_steps=100_000`), implying many dozens of
            epochs over the underlying windows; without this, every epoch
            transition would pay full worker-startup cost again.
        shuffle: standard DataLoader shuffling; a NEW random permutation
            is drawn every time a fresh epoch starts (i.e. every time this
            class re-creates its internal iterator), so windows aren't
            revisited in the same order epoch to epoch.
        drop_last: drops a final undersized batch so every `.sample()`
            call returns a fixed `batch_size` -- not strictly required
            downstream, but keeps batch shape constant, which is the
            simpler assumption to build on.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self._iterator = None
        self.epoch = 0            # completed epochs so far
        self.samples_this_epoch = 0

    def _start_new_epoch(self) -> None:
        self._iterator = iter(self.loader)
        self.epoch += 1
        self.samples_this_epoch = 0

    def sample(self) -> dict:
        """
        Returns one batch, shaped exactly like `FindGoalOfflineDataset`'s
        collated output (`observations (B,T,N,H,W,C)`, `actions (B,T,N)`,
        `rewards (B,T,N)`, `terminals (B,T)`, `truncations (B,T)`,
        `infos.legals (B,T,N,A)`) -- i.e. exactly what
        `IQLCQLSystem.train_step`'s `experience` argument expects.
        Transparently starts a new shuffled epoch on exhaustion.
        """
        if self._iterator is None:
            self._start_new_epoch()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._start_new_epoch()
            batch = next(self._iterator)
        self.samples_this_epoch += 1
        return batch

    def __len__(self) -> int:
        """Number of windows in the underlying dataset (NOT batches/steps)."""
        return len(self.dataset)

    def close(self) -> None:
        if hasattr(self.dataset, "close"):
            self.dataset.close()