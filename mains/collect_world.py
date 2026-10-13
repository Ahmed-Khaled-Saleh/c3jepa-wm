"""Collect the FindGoal dataset with stable-marl's `World.collect` and its `MixedPolicy`, over many layouts, in parallel.

Episode `k` plays in layout `layouts[k % L]` (the walls and clutter of that reset seed), from a goal and starts drawn
for the episode, so every layout is seen from many starts. The episodes are split into tasks of `chunk` episodes, run
by `workers` processes; each task writes one part. No channel is collected: the layouts are ray-traced afterwards
from the trajectories (`comm_power.py stage=trace`, configs/comm). Settings: configs/collect/config.yaml.

    python collect_world.py
    python collect_world.py stage=collect     # in each job of a SLURM array, then once: python collect_world.py stage=merge

Stages:

* collect: `<out_dir>/parts/part_<first episode>.h5`, one per task (stable-marl's HDF5 format);
* merge: `<out_dir>/dataset.h5`, the parts as one dataset (an HDF5 virtual dataset: nothing is copied, the parts stay);
* trajectories: `<out_dir>/trajectories/trajectories_<layout>.npz`, every agent's cell and heading at every step, as
  `c3jepa_wm.comm.channel_data` reads them.

(The old collector, collect.py, wrote one .npz per rollout in a single layout, with the channel of that layout.)
"""
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

BEHAVIOURS = ('random', 'epsilon_expert', 'waypoint')        # stable-marl's MixedPolicy.BEHAVIOURS: the `behaviour` column


class LayoutWrapper(gym.Wrapper):
    """Episode seed `s` plays in the layout of reset seed `layouts[s % len(layouts)]`, from a goal and starts drawn
    with `s`: agents on distinct free cells, at least the env's `min_goal_spawn_distance` from the goal, random headings.
    Every agent's info holds the layout's seed (`layout`) and the episode's behaviour under `policy` (`behaviour`, the
    index in `BEHAVIOURS`; -1 before the first action). `variations`: factors of variation every reset draws from the
    layout's seed (the env config's, e.g. obstacles.number and clutter.number), as `channel_data.make_env` sets them."""

    def __init__(self, env, layouts, policy=None, variations=()):
        super().__init__(env)
        self.layouts, self.policy, self.layout = list(layouts), policy, None
        if variations:
            env.unwrapped.DEFAULT_VARIATIONS = tuple(variations)

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(None if seed is None else [int(seed), 7])
        self.layout = int(self.layouts[int(seed) % len(self.layouts)] if seed is not None else rng.choice(self.layouts))
        self.env.reset(seed=self.layout, options=options)
        u = self.env.unwrapped
        self._place(u, rng)
        u._goal_infos = u.goal_infos()                              # the goal image and cell, from the new goal
        return u.gen_obs(), self._tag(u._infos())

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, self._tag(infos)

    def _tag(self, infos):
        u = self.env.unwrapped
        name = self.policy.behaviour(u) if self.policy is not None else None
        tags = {'layout': np.int64(self.layout), 'behaviour': np.int64(BEHAVIOURS.index(name) if name else -1)}
        return {a: {**infos.get(a, {}), **tags} for a in range(u.num_agents)}

    @staticmethod
    def _place(u, rng):
        from stable_marl.envs.multigrid.core.world_object import Goal
        u.grid.set(*(int(v) for v in u.goal_pos), None)
        free = [(x, y) for x in range(u.grid.width) for y in range(u.grid.height)
                if (o := u.grid.get(x, y)) is None or o.can_overlap()]
        goal = free[rng.integers(len(free))]
        u.grid.set(*goal, Goal())
        u.goal_pos = np.asarray(goal)
        far = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])
        cells = [c for c in free if c != goal]
        reach = min(int(u._variation('goal.min_spawn_distance')), max(far(c) for c in cells))
        starts = [c for c in cells if far(c) >= reach]               # as FindGoal's own spawn rule
        for agent, k in zip(u.agents, rng.choice(len(starts), len(u.agents), replace=False)):
            agent.state.pos = starts[k]
            agent.state.dir = int(rng.integers(4))


def _env_kwargs(cfg) -> dict:
    "The env's arguments, without Hydra's keys and the variations (`LayoutWrapper`'s)."
    kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    kwargs.pop('_target_'), kwargs.pop('_convert_', None), kwargs.pop('variations', None)
    return kwargs


def _variations(cfg) -> tuple:
    "The factors of variation every reset draws from the layout's seed (the env config's `variations`)."
    return tuple(cfg.env.get('variations') or ())


def collect_part(task):
    "One task: episodes [start, stop) into `path` (written aside, then renamed: a part on disk is complete)."
    cfg, start, stop, path = task
    import stable_marl as sm
    from stable_marl.envs.multigrid.expert import MixedPolicy
    layouts = list(range(*cfg.layouts))
    policy = MixedPolicy(OmegaConf.to_container(cfg.policy.weights), cfg.policy.epsilon, seed=[cfg.policy.seed, start])
    size = cfg.image_size
    world = sm.World(cfg.env_id, num_envs=cfg.num_envs, image_shape=None if size is None else (size, size),
                     pre_wrappers=[lambda env: LayoutWrapper(env, layouts, policy, _variations(cfg))], **_env_kwargs(cfg))
    world.set_policy(policy)
    partial = path.with_suffix('.partial.h5')
    partial.unlink(missing_ok=True)
    world.collect(partial, episodes=stop - start, seed=start, progress=False)   # episode k: seed k
    world.close()
    partial.rename(path)
    return start, stop


def collect(cfg, out):
    "This shard's tasks, `workers` at a time."
    total = (cfg.layouts[1] - cfg.layouts[0]) * cfg.episodes_per_layout
    first, last = total * cfg.shard.index // cfg.shard.count, total * (cfg.shard.index + 1) // cfg.shard.count
    (out / 'parts').mkdir(parents=True, exist_ok=True)
    tasks = [(cfg, a, min(a + cfg.chunk, last), out / 'parts' / f'part_{a:06d}.h5') for a in range(first, last, cfg.chunk)]
    todo = [t for t in tasks if cfg.overwrite or not t[3].exists()]
    logger.info(f"episodes {first}-{last - 1} of {total}: {len(todo)} of {len(tasks)} tasks to run, {cfg.workers} workers")
    done = 0
    with Pool(min(cfg.workers, max(len(todo), 1)), maxtasksperchild=8) as pool:
        for start, stop in pool.imap_unordered(collect_part, todo):
            done += stop - start
            logger.info(f"episodes {start}-{stop - 1} done ({done} / {sum(t[2] - t[1] for t in todo)})")


def merge(out):
    "`<out>/dataset.h5`: every part's columns as one virtual dataset (episode offsets recomputed), parts in episode order."
    import h5py
    parts = sorted((out / 'parts').glob('part_[0-9]*[0-9].h5'))
    if not parts:
        raise FileNotFoundError(f"no parts in {out / 'parts'}")
    lens, shapes = [], {}
    for p in parts:
        with h5py.File(p, 'r') as f:
            lens.append(f['ep_len'][:])
            for k, d in f.items():
                if k not in ('ep_len', 'ep_offset'):
                    shapes.setdefault(k, (d.dtype, d.shape[1:], []))[2].append(d.shape[0])
            attrs = dict(f.attrs) if p == parts[0] else attrs
    ep_len = np.concatenate(lens)
    with h5py.File(out / 'dataset.h5', 'w') as f:
        for k, (dtype, shape, rows) in shapes.items():
            layout = h5py.VirtualLayout(shape=(sum(rows), *shape), dtype=dtype)
            at = 0
            for p, n in zip(parts, rows):
                layout[at:at + n] = h5py.VirtualSource(os.path.relpath(p, out), k, shape=(n, *shape))
                at += n
            f.create_virtual_dataset(k, layout)
        f['ep_len'] = ep_len.astype(np.int32)
        f['ep_offset'] = np.r_[0, np.cumsum(ep_len)[:-1]].astype(np.int64)
        f.attrs.update(attrs)
    logger.info(f"{len(ep_len)} episodes, {ep_len.sum()} steps, from {len(parts)} parts in {out / 'dataset.h5'}")


def export_trajectories(out, max_steps):
    """`<out>/trajectories/trajectories_<layout>.npz` as `c3jepa_wm.comm.channel_data.collect_trajectories` writes them
    (cells -1 once an agent reached the goal: it no longer sends), for `trace_visited_pairs`, `build_dataset` and
    `compare_network`."""
    import h5py
    episodes = defaultdict(list)
    for path in sorted((out / 'parts').glob('part_[0-9]*[0-9].h5')):
        with h5py.File(path, 'r') as f:
            pos, dirs, done = f['position'][:], f['direction'][:], f['terminated'][:]
            layout, behaviour = f['layout'][:, 0], f['behaviour'][:, 0]
            for offset, length in zip(f['ep_offset'][:], f['ep_len'][:]):
                rows = slice(int(offset), int(offset) + min(int(length), max_steps))
                b = behaviour[int(offset):int(offset) + int(length)].max()
                episodes[int(layout[offset])].append((pos[rows], dirs[rows], done[rows], BEHAVIOURS[b] if b >= 0 else ''))
    to = out / 'trajectories'
    to.mkdir(parents=True, exist_ok=True)
    for layout, eps in sorted(episodes.items()):
        n = eps[0][0].shape[1]
        cells = np.full((len(eps), max_steps, n, 2), -1, np.int16)
        heads = np.full((len(eps), max_steps, n), -1, np.int8)
        for e, (p, d, t, _) in enumerate(eps):
            cells[e, :len(p)][~t] = p[~t]
            heads[e, :len(d)][~t] = d[~t]
        np.savez_compressed(to / f'trajectories_{layout}.npz', cells=cells, dirs=heads,
                            behaviour=np.array([b for *_, b in eps]), layout=layout)
    logger.info(f"{sum(map(len, episodes.values()))} episodes of {len(episodes)} layouts in {to}")


@hydra.main(version_base=None, config_path="../configs", config_name="collect/config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    out = Path(to_absolute_path(cfg.out_dir))
    stages = ('collect', 'merge', 'trajectories') if cfg.stage == 'all' else (cfg.stage,)
    if 'collect' in stages:
        collect(cfg, out)
    if 'merge' in stages:
        merge(out)
    if 'trajectories' in stages:
        export_trajectories(out, cfg.env.max_steps)


if __name__ == '__main__':
    main()
