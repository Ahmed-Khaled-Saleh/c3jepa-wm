"""Tests of mains/collect_world.py on a small collection (4 layouts x 3 episodes, 2 workers).

    python tests/test_collect_world.py              # or: pytest tests/test_collect_world.py
    python tests/test_collect_world.py --keep DIR   # keep the collection (tests/inspect_collect_world.ipynb plots it)

`small_collection` is also what the inspection notebook runs.
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mains'))
import collect_world as cw                                    # noqa: E402

OVERRIDES = ['layouts=[0,4]', 'episodes_per_layout=3', 'workers=2', 'chunk=2', 'shard.index=0', 'shard.count=1']


def config(out_dir, overrides=()):
    "The collector's Hydra config (configs/collect/config.yaml) with `overrides`, writing to `out_dir`."
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose('collect/config', overrides=[*OVERRIDES, f'out_dir={out_dir}', *overrides])


def small_collection(out_dir, overrides=()):
    "Collect, merge and export a small dataset into `out_dir`; its config."
    cfg = config(out_dir, overrides)
    out = Path(out_dir)
    cw.collect(cfg, out)
    cw.merge(out)
    cw.export_trajectories(out, cfg.env.max_steps)
    return cfg


def episodes(path):
    "Each episode's rows of every column of a dataset (a part or the merged one)."
    with h5py.File(path, 'r') as f:
        cols = [k for k in f if k not in ('ep_len', 'ep_offset')]
        return [{k: f[k][o:o + n] for k in cols} for o, n in zip(f['ep_offset'][:], f['ep_len'][:])]


def same(a, b) -> bool:
    "Equal columns (the NaN of the reset's action and reward included)."
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == 'f') for k in a)


def reference_walls(cfg, layout):
    "The wall cells of the layout of reset seed `layout`, as the ray tracing builds it (configs/comm's env)."
    from c3jepa_wm.comm.channel_data import make_env
    from stable_marl.envs.multigrid.core.constants import Type
    from stable_marl.envs.multigrid.core.world_object import WorldObj
    env = make_env(cfg.env)
    env.reset(seed=int(layout))
    return env.unwrapped.grid.state[..., WorldObj.TYPE] == Type.wall.to_index()


def walls(state):
    from stable_marl.envs.multigrid.core.constants import Type
    from stable_marl.envs.multigrid.core.world_object import WorldObj
    return state[..., WorldObj.TYPE] == Type.wall.to_index()


def check(out_dir, cfg):
    out = Path(out_dir)
    layouts = list(range(*cfg.layouts))
    eps = episodes(out / 'dataset.h5')
    total = len(layouts) * cfg.episodes_per_layout

    # every episode, once, in episode order; episode k in layout layouts[k % L]
    assert len(eps) == total, (len(eps), total)
    seeds = [int(e['seed'][0]) for e in eps]
    assert seeds == list(range(total)), seeds
    for k, e in enumerate(eps):
        assert (e['layout'] == layouts[k % len(layouts)]).all()
        assert (e['seed'] == k).all()

    # the merged dataset is the parts, in order
    parts = [e for p in sorted((out / 'parts').glob('part_[0-9]*[0-9].h5')) for e in episodes(p)]
    assert len(parts) == len(eps)
    for a, b in zip(parts, eps):
        assert same(a, b)

    # the walls are the layout's, as the tracing builds it; the goal and the starts change within a layout
    by_layout = {}
    for e in eps:
        layout = int(e['layout'][0, 0])
        assert np.array_equal(walls(e['state'][0]), reference_walls(cfg, layout)), f'layout {layout}: other walls'
        assert all(np.array_equal(walls(s), walls(e['state'][0])) for s in e['state']), 'walls changed in an episode'
        by_layout.setdefault(layout, []).append(e)
    for layout, es in by_layout.items():                        # no two episodes of a layout with the same goal and starts
        starts = {(tuple(e['goal_position'][0, 0]), tuple(map(tuple, e['position'][0]))) for e in es}
        assert len(starts) == len(es), f'layout {layout}: episodes with the same goal and starts'

    for e in eps:
        goal = e['goal_position'][0, 0]
        # the goal: one per episode, the same for both agents, on a free cell, its image constant
        assert (e['goal_position'] == goal).all()
        assert not walls(e['state'][0])[tuple(goal)]
        assert all(np.array_equal(g, e['goal'][0]) for g in e['goal'])
        # the starts: distinct free cells, at least min_goal_spawn_distance (5) from the goal
        start = e['position'][0]
        assert tuple(start[0]) != tuple(start[1])
        assert (np.abs(start - goal).sum(1) >= 5).all(), (start, goal)
        assert not any(walls(e['state'][0])[tuple(p)] for p in start)
        # the observations follow the agents: 'position' changes only by a step to a neighbouring cell
        moves = np.abs(np.diff(e['position'], axis=0)).sum(-1)
        assert (moves <= 1).all()
        # the behaviour: unknown at the reset, then one for the whole episode
        b = e['behaviour'][:, 0]
        assert b[0] == -1 and (b[1:] == b[1]).all() and b[1] in (0, 1, 2)
        # an agent that reached the goal is on it; the episode ends when both are, or at max_steps
        t = e['terminated']
        for i in range(2):
            if t[:, i].any():
                assert (e['position'][np.argmax(t[:, i]), i] == goal).all()
        steps = len(b) - 1
        assert t[-1].all() or steps == cfg.env.max_steps, (steps, t[-1])
        assert steps <= cfg.env.max_steps

    # the trajectories, as c3jepa_wm.comm.channel_data reads them
    from c3jepa_wm.comm.channel_data import visited_pairs
    for layout, es in by_layout.items():
        z = np.load(out / 'trajectories' / f'trajectories_{layout}.npz')
        assert z['cells'].shape == (len(es), cfg.env.max_steps, 2, 2) and int(z['layout']) == layout
        for e, cells, b in zip(es, z['cells'], z['behaviour']):
            n = min(len(e['position']), cfg.env.max_steps)
            on = ~e['terminated'][:n]
            assert np.array_equal(cells[:n][on], e['position'][:n][on])
            assert (cells[:n][~on] == -1).all() and (cells[n:] == -1).all()
            assert b == cw.BEHAVIOURS[e['behaviour'][1, 0]]
        assert len(visited_pairs(out / 'trajectories' / f'trajectories_{layout}.npz')) > 0

    # stable-marl reads the merged dataset
    import stable_marl as sm
    ds = sm.HDF5Dataset(path=str(out / 'dataset.h5'), num_steps=4)
    clip = ds[0]
    assert tuple(clip['pixels'].shape[:2]) == (4, 2)
    del ds


def check_reproducible(out_dir, cfg):
    "A task run again gives the same episodes."
    out = Path(out_dir)
    again = out / 'again.h5'
    cw.collect_part((cfg, 0, cfg.chunk, again))
    first = episodes(out / 'parts' / 'part_000000.h5')
    for a, b in zip(first, episodes(again)):
        assert same(a, b)
    again.unlink()


def check_resume(out_dir, cfg):
    "A second run skips the parts already written."
    out = Path(out_dir)
    part = out / 'parts' / 'part_000000.h5'
    before = part.stat().st_mtime_ns
    cw.collect(cfg, out)
    assert part.stat().st_mtime_ns == before


def check_variations(out_dir, cfg):
    "With the env config's variations: the numbers of walls differ between layouts, and each is the traced layout's."
    eps = episodes(Path(out_dir) / 'dataset.h5')
    first = {int(e['layout'][0, 0]): e for e in eps}
    counts = {k: int(walls(e['state'][0]).sum()) for k, e in first.items()}
    assert len(set(counts.values())) > 1, counts                     # fixed numbers would give similar counts
    for k, e in first.items():                                      # reference_walls builds the env with make_env
        assert np.array_equal(walls(e['state'][0]), reference_walls(cfg, k)), f'layout {k}: other walls'


def test_collect_world_variations(tmp_path=None):
    with tempfile.TemporaryDirectory() as d:
        out = Path(tmp_path or d) / 'collect_varied'
        cfg = small_collection(out, ['env.variations=[obstacles.number,clutter.number]', 'layouts=[0,6]',
                                     'episodes_per_layout=2'])
        check(out, cfg)
        check_variations(out, cfg)


def test_collect_world(tmp_path=None):
    with tempfile.TemporaryDirectory() as d:
        out = Path(tmp_path or d) / 'collect'
        cfg = small_collection(out)
        check(out, cfg)
        check_reproducible(out, cfg)
        check_resume(out, cfg)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--keep', default=None, help='collect into this directory and keep it')
    a = p.parse_args()
    if a.keep:
        cfg = small_collection(a.keep)
        check(a.keep, cfg)
        check_reproducible(a.keep, cfg)
        check_resume(a.keep, cfg)
    else:
        test_collect_world()
        test_collect_world_variations()
    print('ok')
