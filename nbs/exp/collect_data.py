
import copy
import os
import numpy as np
import gymnasium as gym
import multigrid.envs




def collect_one_rollout_numpy(args):
    rollout_idx, seed, seed_steps, data_dir, n_agents = args

    env = gym.make('MultiGrid-FindGoal-15x15-v0',
                   agents= n_agents,
                   render_mode='rgb_array',
                   num_obstacles=6,
                   width=15,
                   height=15)

    agents = [i for i in range(n_agents)]

    obs, info = env.reset(seed= seed)

    layout = env.unwrapped.get_layout(tile_size= 32)
    goal_obs = np.array([env.unwrapped.get_goal_state(agent= ag, agent_view_size= 7) for ag in agents])
    goal_pos = env.unwrapped.goal_pos

    agent_data = {
        ag: {k: [] for k in ["image", "pov", "pos", "dir", "act", "rew"]}
        for ag in agents
    }

    episode_len = 0
    success = False
    success_at = -1

    for t in range(seed_steps):
        current_obs = obs
        actions = {i: env.action_space[i].sample() for i in range(env.unwrapped.num_agents)}
        obs, rewards, terminations, truncations, info = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())

        for ag in agents:
            agent_data[ag]["image"].append(current_obs[ag]["image"])# list of ndarrays
            agent_data[ag]["pov"].append(current_obs[ag]["pov"]) # list of ndarrays
            agent_data[ag]["pos"].append(env.unwrapped.agents[ag].state.pos) # list of tuples
            agent_data[ag]["dir"].append(obs[ag]["direction"])

            agent_data[ag]["act"].append(actions[ag]) # actions are list of int64

            # agent_data[ag]["sees_goal"].append(info[ag]["sees_goal"])# list of np.int64
            agent_data[ag]["rew"].append(rewards[ag])# list of float64

        episode_len += 1
        if done:
            if all(terminations.values()):
                success = True
                success_at = t
            break

    env.close()
    
    ########### Saving the rollout Meta data ###########
    save_path = os.path.join(data_dir, f"rollout_{rollout_idx}.npz")
    os.makedirs(data_dir, exist_ok=True)
    save_dict = {}

    for ag in agents:
        save_dict[f"{ag}_image"] = np.stack(agent_data[ag]["image"]).astype(np.uint8)
        save_dict[f"{ag}_pov"] = np.stack(agent_data[ag]["pov"]).astype(np.uint8)
        save_dict[f"{ag}_act"] = np.asarray(agent_data[ag]["act"])
        save_dict[f"{ag}_done"] = np.asarray(agent_data[ag]["done"])

        save_dict[f"{ag}_rew"] = np.asarray(agent_data[ag]["rew"])
        save_dict[f"{ag}_pos"] = np.stack(agent_data[ag]["pos"])
        save_dict[f"{ag}_dir"] = np.asarray(agent_data[ag]["dir"])
        # save_dict[f"{ag}_sees_goal"] = np.asarray(agent_data[ag]["sees_goal"])

    
    save_dict['episode_len'] = episode_len
    save_dict['success'] = success
    save_dict['success_at'] = success_at
    save_dict['seed'] = seed
    save_dict['layout'] = layout
    save_dict['goal_obs'] = goal_obs
    save_dict['goal_pos'] = goal_pos

    np.savez_compressed(save_path, **save_dict)
    print(f"> Saved rollout {rollout_idx} to {save_path}")
    
    return rollout_idx
    

from multiprocessing import Pool, cpu_count

def generate_parallel(
    rollouts=100,
    seed_steps=4000,
    data_dir="../datasets/marl_grid_data_v3",
    workers=None,
):
    os.makedirs(data_dir, exist_ok=True)

    print("Number of workers:", workers)

    n_agents = 2
    seed = 0
    
    args = [
        (i, seed, seed_steps, data_dir, n_agents)
        for i in range(rollouts)
    ]

    with Pool(processes=workers) as p:
        for idx in p.imap_unordered(collect_one_rollout_numpy, args):
            print(f"✓ rollout {idx} done")

if __name__ == "__main__":
    generate_parallel(
        rollouts=10000,
        seed_steps=2000,
        data_dir="/scratch/project_2009050/datasets/MarlGridV3",
        workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
    )