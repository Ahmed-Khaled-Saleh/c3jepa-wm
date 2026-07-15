"""
PyTorch Dataset over the merged HDF5 rollout file (from `collect_dataset` /
`merge_npz_to_hdf5`), producing fixed-length sequence windows in exactly the
shape `og_marl`'s `IQLCQLSystem.train_step`'s "Experience" dict expects (see
`og_marl/baselines/torch_systems/offline/iql_cql.py::_train_step`):

    observations : (B, T, N, H, W, C) uint8   -- raw POV pixels; a later
                    ViT-fronted network (not this file) is responsible for
                    encoding these into the flat feature vectors the
                    original DeepRNN-based network assumed
    actions      : (B, T, N)         int64
    rewards      : (B, T, N)         float32
    terminals    : (B, T, N)         float32  -- PER-AGENT, not team-level;
                    see __getitem__ for why (og-marl's own inline comments
                    claim (B,T), which doesn't match what
                    merge_batch_and_agent_dim_of_time_major_sequence
                    actually requires -- traced through
                    FlashbaxReplayBuffer.add() to confirm)
    truncations  : (B, T, N)         float32  -- same as terminals, per-agent
    infos.legals : (B, T, N, A)      bool

A `DataLoader(dataset, batch_size=B)` batch from this class collates
directly into that shape (PyTorch's default_collate handles the nested
"infos" dict recursively), so it's a drop-in replacement for
`FlashbaxReplayBuffer.sample()` -- no `flashbax`/JAX dependency needed for
a from-scratch custom-env dataset. `IQLCQLSystem.train_step` still runs its
own `jax.tree.map(lambda x: np.array(x), experience)` conversion
internally, which accepts torch.Tensor input fine (CPU tensors satisfy
numpy's `__array__` protocol) -- swapping that outer wrapper for something
JAX-free entirely is left to the network-adaptation piece, not this file.

Deliberately does NOT gate window generation on episode success, unlike
`MultiAgentWorldModelDataset`'s `np.min(agents_success_at)` approach (which
implicitly trains only on already-successful episodes, since
`agents_success_at[ag] == -1` for any agent that never reaches the goal,
poisoning the min for any non-fully-successful episode -- see chat).
CQL's conservative penalty specifically needs to see the states/actions a
mixed-quality behavior policy (this dataset mixes random/epsilon-greedy-A*/
waypoint policies) visits, not just successful trajectories.
"""
from __future__ import annotations

import numpy as np
import h5py
from torch.utils.data import Dataset


class FindGoalOfflineDataset(Dataset):
    """
    Args:
        h5_path: path to the merged HDF5 file (one group per episode, as
            written by `collect_one_rollout` + `merge_npz_to_hdf5`).
        sequence_length: T, number of consecutive frames per window
            (matches Flashbax's `sequence_length` concept). Episodes
            shorter than this are skipped entirely (see `total_frames`
            below) -- worth checking how many episodes that excludes for
            your `max_steps=150` collection run.
        sample_period: stride between valid window start positions within
            an episode (matches Flashbax's `sample_period`/`period`; 1 =
            every possible start position, no windows skipped).
        split / val_ratio: train/val split by episode, using the SAME
            convention as `MultiAgentWorldModelDataset` (sorted episode
            keys, contiguous prefix/suffix split, no shuffling) so the two
            dataset classes stay consistent with each other if ever used
            side by side (e.g. jointly evaluating a world model and an
            offline policy) -- deliberately not an independently-shuffled
            split, which could otherwise leak differently between them.
        num_agents / num_actions: must match your env (FindGoalEnv's
            NavigationAction has 4 actions: left, right, forward, done).
        obs_key: "pov" (RGB, what a ViT-fronted network needs) or "img"
            (symbolic grid encoding, if you ever want a non-pixel variant).

    Scalar episode-level fields (`episode_len`, `success`, `success_at`,
    also `seed`/`policy`, unused here) are read from HDF5 group *attrs*,
    not datasets -- confirmed against `merge_npz_to_hdf5`'s actual source:
    it stores any 0-dim array via `grp.attrs[k] = arr.item()`, and only
    non-scalar arrays (everything else: `agents_success_at`, `layout`,
    `goal_obs`, `goal_pos`, and all `{agent}_*` per-timestep arrays) via
    `grp.create_dataset(...)`. Since `episode_len`/`success`/`success_at`
    started life as plain Python int/bool/int in `collect_one_rollout`'s
    `save_dict`, they're 0-dim -> attrs.
    """

    def __init__(
        self,
        h5_path: str,
        sequence_length: int = 20,
        sample_period: int = 1,
        split: str = "train",
        val_ratio: float = 0.1,
        num_agents: int = 2,
        num_actions: int = 4,
        obs_key: str = "pov",
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.obs_key = obs_key
        self.file: h5py.File | None = None
        self.index_map: list[tuple[str, int]] = []

        with h5py.File(h5_path, "r") as f:
            all_episodes = sorted(f.keys())
            val_count = int(len(all_episodes) * val_ratio)
            train_count = len(all_episodes) - val_count
            active_episodes = (
                all_episodes[:train_count] if split == "train" else all_episodes[train_count:]
            )

            n_skipped_short = 0
            n_success = 0
            for ep_key in active_episodes:
                grp = f[ep_key]
                episode_len = int(grp.attrs["episode_len"])
                success = bool(grp.attrs["success"])
                n_success += int(success)

                # collect_one_rollout appends episode_len regular transitions
                # (indices 0..episode_len-1) PLUS one extra padding frame at
                # the end (index episode_len) holding the post-episode
                # observation with a -1 action/reward sentinel -- see module
                # docstring and collect_one_rollout's "if done or t ==
                # max_steps - 1" block. So there are episode_len + 1 stored
                # frames total, regardless of whether the episode succeeded.
                total_frames = episode_len + 1
                if total_frames < sequence_length:
                    n_skipped_short += 1
                    continue

                for start in range(0, total_frames - sequence_length + 1, sample_period):
                    self.index_map.append((ep_key, start))

        print(
            f"[FindGoalOfflineDataset:{split}] {len(active_episodes)} episodes "
            f"({n_success} successful, {len(active_episodes) - n_success} not) -> "
            f"{len(self.index_map)} windows (sequence_length={sequence_length}, "
            f"sample_period={sample_period}); {n_skipped_short} episode(s) skipped "
            f"for being shorter than sequence_length."
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def _lazy_open(self) -> None:
        # h5py.File objects aren't fork-safe / picklable across DataLoader
        # worker processes -- open lazily per-worker on first access rather
        # than in __init__, matching MultiAgentWorldModelDataset's pattern.
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int) -> dict:
        self._lazy_open()
        ep_key, start = self.index_map[idx]
        grp = self.file[ep_key]
        end = start + self.sequence_length

        episode_len = int(grp.attrs["episode_len"])
        agents_success_at = grp["agents_success_at"][...]  # (N,) int, -1 if that agent never reached its goal this episode

        obs_stack, act_stack, rew_stack = [], [], []
        for ag in range(self.num_agents):
            obs_stack.append(grp[f"{ag}_{self.obs_key}"][start:end])           # (T, H, W, C) uint8
            act_stack.append(grp[f"{ag}_act"][start:end].astype(np.int64))     # (T,)
            rew_stack.append(grp[f"{ag}_rew"][start:end].astype(np.float32))   # (T,)

        observations = np.stack(obs_stack, axis=1)   # (T, N, H, W, C)
        actions = np.stack(act_stack, axis=1)         # (T, N)
        rewards = np.stack(rew_stack, axis=1)         # (T, N)

        # The -1 action/reward sentinel only ever appears at the single
        # padding frame appended at an episode's end (see module docstring)
        # and is never a real action. IQLCQLSystem._train_step's
        # `gather(qs_out, actions, dim=3)` runs over the FULL (unsliced)
        # sequence before its later `chosen_action_qs[:, :-1]` slicing
        # discards exactly that last timestep -- so an un-clamped -1 here
        # would crash gather() with an out-of-bounds index even though the
        # value it produces is guaranteed to never affect the loss. Safe to
        # clamp to any valid action index; 0 is arbitrary.
        actions = np.clip(actions, 0, self.num_actions - 1)

        # PER-AGENT terminals/truncations, shape (T, N) -- NOT team-level
        # (T,). `og_marl`'s `_train_step` reshapes these through
        # `merge_batch_and_agent_dim_of_time_major_sequence`, which requires
        # a 3-D+ (T,B,N,...) input; its own inline comments claim (B,T),
        # which would actually crash that reshape. Tracing
        # `FlashbaxReplayBuffer.add()` confirms the real shape is (B,T,N):
        # terminals/truncations arrive as per-agent dicts
        # (`Terminals = Dict[str, np.ndarray]`) and get `np.stack`ed over
        # the agent axis, matching standard Independent-Q-Learners
        # semantics -- each agent's own Q-target zeroes out at ITS OWN
        # termination, independent of its teammates. That means the
        # correct source signal here is `agents_success_at` (per-agent),
        # not the team-level `success`/`success_at` fields used in an
        # earlier version of this file -- team-level "all agents reached
        # the goal" is the right definition for evaluating overall task
        # success elsewhere, but the wrong signal for this per-agent loss.
        terminals = np.zeros((self.sequence_length, self.num_agents), dtype=np.float32)
        truncations = np.zeros((self.sequence_length, self.num_agents), dtype=np.float32)
        for ag in range(self.num_agents):
            sa = int(agents_success_at[ag])
            if sa >= 0 and start <= sa < end:
                terminals[sa - start, ag] = 1.0
            elif sa < 0 and start <= episode_len - 1 < end:
                # this agent never reached its own goal in the episode, and
                # this window reaches the episode's actual end -> truncated
                # (ran out of time) from this agent's perspective.
                truncations[episode_len - 1 - start, ag] = 1.0
            # else: interior window, or this agent's own outcome doesn't
            # land inside this window -> terminals/truncations stay 0 for
            # that agent at every position in this window, correctly.

        # FindGoalEnv/NavigationAction has no notion of an illegal action
        # (walls block movement silently rather than disallowing the
        # action) -- all actions legal, always.
        legals = np.ones((self.sequence_length, self.num_agents, self.num_actions), dtype=bool)

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
            "infos": {"legals": legals},
        }

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


# if __name__ == "__main__":
#     # Smoke test / usage example.
#     import sys
#     from torch.utils.data import DataLoader

#     h5_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/project_2009050/datasets/findgoal/dataset.h5"

#     train_ds = FindGoalOfflineDataset(h5_path, sequence_length=20, sample_period=1, split="train")
#     val_ds = FindGoalOfflineDataset(h5_path, sequence_length=20, sample_period=1, split="val")

#     loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
#     batch = next(iter(loader))

#     print("observations:", batch["observations"].shape, batch["observations"].dtype)
#     print("actions:     ", batch["actions"].shape, batch["actions"].dtype)
#     print("rewards:     ", batch["rewards"].shape, batch["rewards"].dtype)
#     print("terminals:   ", batch["terminals"].shape, batch["terminals"].dtype)
#     print("truncations: ", batch["truncations"].shape, batch["truncations"].dtype)
#     print("infos.legals:", batch["infos"]["legals"].shape, batch["infos"]["legals"].dtype)
#     print("any terminal in batch:", bool(batch["terminals"].any()))
#     print("any truncation in batch:", bool(batch["truncations"].any()))