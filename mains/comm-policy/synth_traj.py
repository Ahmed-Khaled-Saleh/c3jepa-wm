"""Synthesize agent position trajectories over the free-cell graph, for use as
input to the CSI-history regression when real logged positions are unavailable.

`valid_2d` is an unordered set of free cells, so real trajectories cannot be
recovered from it. What we can do is sample walks whose mobility statistics
approximate the heuristic data-collection policy (20% random, 30% waypoint
exploration, 50% A* expert), so that the temporal correlation of the resulting
gain series is at least in the right regime.

This is an approximation. If you can re-run collection and log agent positions,
do that instead -- see `notes` at the bottom.
"""

import heapq
import random
from collections import deque

import numpy as np

__all__ = [
    "build_graph",
    "shortest_path",
    "random_walk",
    "waypoint_walk",
    "expert_walk",
    "sample_trajectory",
    "make_episode_position_pairs",
]

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ---------------------------------------------------------------------------
# Graph over free cells
# ---------------------------------------------------------------------------

def build_graph(valid_2d):
    """Adjacency dict over 4-connected free cells.

    Coordinates are normalised to plain-int tuples so they hash consistently
    with keys built elsewhere (np.int64 tuples hash the same as int tuples, but
    mixing them makes debugging miserable).
    """
    cells = {(int(x), int(y)) for x, y in valid_2d}
    return {
        c: [n for n in ((c[0] + dx, c[1] + dy) for dx, dy in MOVES) if n in cells]
        for c in cells
    }


def shortest_path(graph, start, goal):
    """BFS path including both endpoints. Returns None if unreachable."""
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v in prev:
                continue
            prev[v] = u
            if v == goal:
                path = [v]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(v)
    return None


# ---------------------------------------------------------------------------
# Walk generators -- each returns a list of cells of length exactly T
# ---------------------------------------------------------------------------

def random_walk(graph, T, start=None, rng=None, stay_prob=0.1):
    """Unbiased walk with a small self-loop probability (the 'do nothing' action)."""
    rng = rng or random.Random()
    cur = start or rng.choice(list(graph))
    path = [cur]
    for _ in range(T - 1):
        cur = cur if rng.random() < stay_prob else rng.choice(graph[cur])
        path.append(cur)
    return path


def waypoint_walk(graph, T, start=None, goal=None, rng=None, n_waypoints=2):
    """Exploration: route through random intermediate waypoints before the goal."""
    rng = rng or random.Random()
    cells = list(graph)
    cur = start or rng.choice(cells)
    goal = goal or rng.choice(cells)
    targets = [rng.choice(cells) for _ in range(n_waypoints)] + [goal]

    path = [cur]
    for tgt in targets:
        seg = shortest_path(graph, cur, tgt)
        if seg is None:
            continue
        path.extend(seg[1:])
        cur = tgt
        if len(path) >= T:
            break
    return _pad_with_graph(graph, path, T, rng)


def expert_walk(graph, T, start=None, goal=None, rng=None):
    """A*-equivalent: on an unweighted grid graph, BFS is already optimal."""
    rng = rng or random.Random()
    cells = list(graph)
    cur = start or rng.choice(cells)
    goal = goal or rng.choice(cells)
    path = shortest_path(graph, cur, goal) or [cur]
    return _pad_with_graph(graph, path, T, rng)


def _pad_with_graph(graph, path, T, rng):
    """Truncate to T, or extend by dithering around the last cell.

    Dithering rather than freezing matters: a frozen tail gives a constant gain
    run, which the ridge fit will happily exploit as trivially predictable and
    which inflates your held-out R^2.
    """
    if len(path) >= T:
        return path[:T]
    out = list(path)
    while len(out) < T:
        out.append(rng.choice(graph[out[-1]]))
    return out


def sample_trajectory(graph, T, rng=None, mix=(0.2, 0.3, 0.5)):
    """Sample one trajectory under the paper's policy mixture."""
    rng = rng or random.Random()
    u = rng.random()
    p_rand, p_explore, _ = mix
    if u < p_rand:
        return random_walk(graph, T, rng=rng)
    if u < p_rand + p_explore:
        return waypoint_walk(graph, T, rng=rng)
    return expert_walk(graph, T, rng=rng)


# ---------------------------------------------------------------------------
# Episode assembly
# ---------------------------------------------------------------------------

def make_episode_position_pairs(valid_2d, n_episodes=2000, T=150, seed=0,
                                shared_goal=True, mix=(0.2, 0.3, 0.5)):
    """Return [(tx_traj, rx_traj), ...], the input `episode_position_pairs`.

    shared_goal=True routes both agents toward the same goal cell, reproducing
    the FindGoal setup where agents converge -- which is what makes the link
    gain drift upward over an episode rather than wander stationarily. Setting
    it False gives independent walks and a noticeably different, flatter
    temporal correlation.

    Steps where the two agents collide are left in; `trajectory_gain_series`
    marks them NaN and `make_windows` drops any window that touches them.
    """
    graph = build_graph(valid_2d)
    cells = list(graph)
    rng = random.Random(seed)

    episodes = []
    for _ in range(n_episodes):
        if shared_goal:
            goal = rng.choice(cells)
            starts = rng.sample(cells, 2)
            trajs = []
            for s in starts:
                u = rng.random()
                if u < mix[0]:
                    trajs.append(random_walk(graph, T, start=s, rng=rng))
                elif u < mix[0] + mix[1]:
                    trajs.append(waypoint_walk(graph, T, start=s, goal=goal, rng=rng))
                else:
                    trajs.append(expert_walk(graph, T, start=s, goal=goal, rng=rng))
            tx, rx = trajs
        else:
            tx = sample_trajectory(graph, T, rng=rng, mix=mix)
            rx = sample_trajectory(graph, T, rng=rng, mix=mix)
        episodes.append((tx, rx))
    return episodes