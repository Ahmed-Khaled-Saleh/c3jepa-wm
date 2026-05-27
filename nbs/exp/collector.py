

import os
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import gymnasium as gym
from wandb import agent
import multigrid.envs
from multigrid.core.actions import NavigationAction
from a_star_policy import astar, path_to_actions
from c3jepa_wm.data.utils import get_h_full, get_valid_2d, np_get_csi


H_full   = get_h_full()
valid_2d = get_valid_2d()
N = len(valid_2d)

grid_to_idx  = {(gx, gy): i for i, (gx, gy) in enumerate(valid_2d)}
idx_to_grid  = {i: (gx, gy) for i, (gx, gy) in enumerate(valid_2d)}


def collect_channel(env, agent_idx, terminations, d= 1):
    """
    Querry precollected H_full for the channel between tx_position and each of the rx_positions.
    return array of shape (len(rx_positions), d) where each row is the channel vector for that rx_position.
    """
    if terminations[agent_idx]:
        all_csi = np.zeros((len(env.agents) - 1, d), dtype=np.complex128)  # no valid CSI if agent is terminated
        return all_csi
    
    tx_position = env.agents[agent_idx].state.pos
    rx_positions = [ag.state.pos for ag in env.agents if ag != env.agents[agent_idx] and not terminations[ag.index]]

    if len(rx_positions) == 0:
        all_csi = np.zeros((len(env.agents) - 1, d), dtype=np.complex128) 
        return all_csi
    
    csi_values = []
    for pos in rx_positions:
        h = [np_get_csi(H_full, grid_to_idx, tx_position, pos)]
        csi_values.append(h)

    all_csi = np.array(csi_values, dtype=np.complex128).reshape(len(env.agents) - 1, d)
    
    return all_csi


# ── Policy helpers (reuse your existing A* code) ──────────────────────────────

def get_policy_actions(
    policy: str,
    env,
    goal_pos: Tuple[int, int],
    action_queues: Dict,
    waypoints: Dict,
    epsilon: float = 0.3,
) -> Dict[int, int]:
    """
    Return actions for all agents under the given policy.
    """
    actions = {}
    for i in range(env.unwrapped.num_agents):
        if policy == 'random':
            actions[i] = env.action_space[i].sample()

        elif policy == 'epsilon_astar':
            if env.unwrapped.np_random.random() < epsilon:
                actions[i] = env.action_space[i].sample()
            else:
                actions[i] = _astar_action(i, env.unwrapped, goal_pos, action_queues)

        elif policy == 'waypoint':
            actions[i] = _waypoint_action(i, env.unwrapped, goal_pos, waypoints)

    return actions


def _astar_action(
    agent_idx: int,
    env,
    goal_pos: Tuple[int, int],
    action_queues: Dict,
) -> int:
    """Pop next A* action, replanning if queue is empty."""
    if not action_queues[agent_idx]:
        start = tuple(env.agents[agent_idx].state.pos)
        path  = astar(start, tuple(goal_pos), env)
        if path and len(path) > 1:
            action_queues[agent_idx] = path_to_actions(
                path, int(env.agents[agent_idx].state.dir)
            )
    if action_queues[agent_idx]:
        return action_queues[agent_idx].pop(0)
    return env.action_space[agent_idx].sample()  # fallback


def _waypoint_action(
    agent_idx: int,
    env,
    goal_pos: Tuple[int, int],
    waypoints: Dict,
) -> int:
    """Navigate through random intermediate waypoints before the goal."""
    # If no waypoints queued, sample a new random one
    if not waypoints[agent_idx]:
        for _ in range(1000):
            wx = env.np_random.integers(1, env.width - 1)
            wy = env.np_random.integers(1, env.height - 1)
            if env.grid.get(int(wx), int(wy)) is None:
                waypoints[agent_idx] = [
                    (int(wx), int(wy)),
                    tuple(goal_pos),   # always end at goal
                ]
                break

    if waypoints[agent_idx]:
        current_pos = tuple(env.agents[agent_idx].state.pos)
        next_wp     = waypoints[agent_idx][0]

        # Pop waypoint if reached
        if current_pos == next_wp:
            waypoints[agent_idx].pop(0)
            if not waypoints[agent_idx]:
                return env.action_space[agent_idx].sample()
            next_wp = waypoints[agent_idx][0]

        # Plan one step toward next waypoint
        path = astar(current_pos, next_wp, env)
        if path and len(path) > 1:
            acts = path_to_actions(
                path, int(env.agents[agent_idx].state.dir)
            )
            return acts[0] if acts else env.action_space[agent_idx].sample()

    return env.action_space[agent_idx].sample()


# ── Policy sampler ────────────────────────────────────────────────────────────

POLICY_MIX = {
    'random':        0.20,
    'epsilon_astar': 0.50,
    'waypoint':      0.30,
}

def sample_policy(rng: np.random.Generator) -> str:
    return rng.choice(
        list(POLICY_MIX.keys()),
        p=list(POLICY_MIX.values()),
    )


# ── Single rollout ────────────────────────────────────────────────────────────

def collect_one_rollout(args):
    """
    Collect one rollout using a sampled policy mixture.

    Args tuple: (rollout_idx, seed, seed_steps, data_dir, n_agents,
                 env_width, env_height, num_obstacles)
    """
    (rollout_idx, seed, max_steps,
     data_dir, n_agents) = args

    # ── Build env ─────────────────────────────────────────────────────────────
    env = gym.make(
        'MultiGrid-FindGoal-15x15-v0',
        agents=n_agents,
        render_mode='rgb_array',
        num_obstacles=6,
        width=15,
        height=15,
    )

    agents    = list(range(n_agents))
    obs, info = env.reset(seed=seed)

    # ── Episode metadata ──────────────────────────────────────────────────────
    layout   = env.unwrapped.get_layout(tile_size=32)
    goal_pos = env.unwrapped.goal_pos
    goal_obs = np.array([
        env.unwrapped.get_goal_state(agent=env.unwrapped.agents[ag], agent_view_size=7)
        for ag in agents
    ])

    # ── Sample ONE policy for the whole rollout ───────────────────────────────
    rng    = np.random.default_rng(seed + rollout_idx)   # reproducible policy choice per seed
    policy = sample_policy(rng)

    # Per-agent A* state (queues reset each rollout)
    action_queues: Dict[int, List] = defaultdict(list)
    waypoints:     Dict[int, List] = defaultdict(list)

    # ── Per-agent storage ─────────────────────────────────────────────────────
    agent_data = {
        ag: {k: [] for k in ["img", "pov", "pos", "dir", "act", "rew", "csi"]}
        for ag in agents
    }

    episode_len = 0
    success     = False
    success_at  = -1

    # ── Rollout loop ──────────────────────────────────────────────────────────
    for t in range(max_steps):
        current_obs = obs

        # Capture positions BEFORE step (while still valid)
        current_pos = {ag: env.unwrapped.agents[ag].state.pos for ag in agents}
        current_dir = {ag: int(env.unwrapped.agents[ag].state.dir) for ag in agents}

        actions = get_policy_actions(
            policy=policy,
            env=env,
            goal_pos=tuple(goal_pos),
            action_queues=action_queues,
            waypoints=waypoints,
            epsilon=0.3,
        )

        if t > 0 and any(terminations.values()):
            for ag in agents:
                if terminations[ag]:
                    actions[ag] = NavigationAction.done

        obs, rewards, terminations, truncations, info = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())

        # ── Regular transition (o_t, a_t, r_t) ───────────────────────────
        for ag in agents:
            agent_data[ag]["img"].append(current_obs[ag]["image"])
            agent_data[ag]["pov"].append(current_obs[ag]["pov"])
            agent_data[ag]["pos"].append(current_pos[ag])   # ← pre-step pos
            agent_data[ag]["dir"].append(current_dir[ag])   # ← pre-step dir
            agent_data[ag]["act"].append(actions[ag])
            agent_data[ag]["rew"].append(rewards[ag])
            agent_data[ag]["csi"].append(
                collect_channel(env.unwrapped, ag, terminations))

        episode_len += 1

        if done or t == max_steps - 1:
            for ag in agents:
                agent_data[ag]["img"].append(obs[ag]["image"])   # ✓ already correct
                agent_data[ag]["pov"].append(obs[ag]["pov"])     # ✓ already correct
                agent_data[ag]["dir"].append(int(obs[ag]["direction"]))  # ✓ from _last_obs

                # ── Only this needed fixing ───────────────────────────────────
                term_pos = np.asarray(goal_pos) if terminations[ag] \
                        else env.unwrapped.agents[ag].state.pos#.copy()
                agent_data[ag]["pos"].append(term_pos)

                agent_data[ag]["act"].append(-1)
                agent_data[ag]["rew"].append(-1)
                agent_data[ag]["csi"].append(
                    collect_channel(env.unwrapped, ag, terminations))

            if all(terminations.values()):
                success    = True
                success_at = t
            break

    env.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f"rollout_{rollout_idx}.npz")

    save_dict = {
        'episode_len': episode_len,
        'success':     success,
        'success_at':  success_at,
        'seed':        seed,
        'policy':      policy,           # which policy was used
        'layout':      layout,
        'goal_obs':    goal_obs,
        'goal_pos':    np.asarray(goal_pos),
    }

    for ag in agents:
        save_dict[f"{ag}_img"] = np.stack(agent_data[ag]["img"]).astype(np.uint8)
        save_dict[f"{ag}_pov"]   = np.stack(agent_data[ag]["pov"]).astype(np.uint8)
        save_dict[f"{ag}_pos"]   = np.stack(agent_data[ag]["pos"])
        save_dict[f"{ag}_dir"]   = np.asarray(agent_data[ag]["dir"])
        save_dict[f"{ag}_act"]   = np.asarray(agent_data[ag]["act"])
        save_dict[f"{ag}_rew"]   = np.asarray(agent_data[ag]["rew"])
        save_dict[f"{ag}_csi"]   = np.stack(agent_data[ag]["csi"])

    np.savez_compressed(save_path, **save_dict)
    print(f"> [{policy:14s}] Rollout {rollout_idx:04d} | "
          f"len={episode_len:3d} | success={success} | saved to {save_path}")

    return rollout_idx


# ── Parallel dataset collection ───────────────────────────────────────────────

def collect_dataset(
    n_rollouts:    int  = 10_000,
    max_steps:    int  = 150,
    data_dir:      str  = "./data/rollouts",
    n_agents:      int  = 2,
    n_workers:     int  = 8,
    base_seed:     int  = 0,
):
    
    from multiprocessing import Pool

    args_list = [
        (
            idx,
            # base_seed + idx,   # unique seed per rollout → unique layout + goal
            base_seed, # same seed for all rollouts → same layouts, but different goal position, policies + trajectories
            max_steps,
            data_dir,
            n_agents,
        )
        for idx in range(n_rollouts)
    ]

    collect_one_rollout(args_list[0])  # sanity check with one rollout before parallel execution
    with Pool(n_workers) as pool:
        results = pool.map(collect_one_rollout, args_list)

    print(f"\nDone. Collected {len(results)} rollouts into {data_dir}")
    return results


if __name__ == "__main__":
    collect_dataset(
        n_rollouts=10,#10_000,
        max_steps=150,
        # data_dir="/scratch/project_2009050/datasets/findgoal/rollouts",
        n_agents=2,
        n_workers=4,#int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        base_seed=0,
    )

    # from merge_h5 import merge_npz_to_hdf5
    # merge_npz_to_hdf5(data_dir="/scratch/project_2009050/datasets/findgoal/rollouts", out_path="/scratch/project_2009050/datasets/findgoal/dataset.h5")