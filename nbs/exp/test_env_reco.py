import gymnasium as gym
import multigrid.envs
from multigrid.wrappers.base import GridRecorder
import numpy as np

env = gym.make('MultiGrid-FindGoal-15x15-v0', agents=2, render_mode='rgb_array', num_obstacles=6, width=15, height=15)

env = GridRecorder(env.unwrapped, save_root=".", render_kwargs={"tile_size": 11}, max_steps= 150)
env.recording = True

action_mapping = {
    0: "Left",
    1: "Right",
    2: "Forward",
    3: "Done",
    }



agents = [i for i in range(2)]

lst_actions = [] 
lst_dirs = []
done = False
step = 0


obs, info = env.reset(seed= 0)

lst_dirs.append({f"agent_{i}": (agent.state.dir, agent.state.pos) for i, agent in enumerate(env.unwrapped.agents)})

while not done and step < 150:
    actions = {i: env.action_space[i].sample() for i in range(2)}

    # import ipdb; ipdb.set_trace()  # Debug point to inspect variables before stepping through the environment
    print(f"Step {step}, Actions: { {f'agent_{i}': action_mapping[actions[agent]] for i, agent in enumerate(agents)} }")
    
    obs, rewards, terminations, truncations, info = env.step(actions)
    
    lst_actions.append({agent: action_mapping[action] for agent, action in actions.items()})
    lst_dirs.append({f"agent_{i}": (agent.state.dir, agent.state.pos) for i, agent in enumerate(env.unwrapped.agents)})  # Record AFTER
    
    done = all(terminations.values()) or all(truncations.values())
    step += 1
    
env.export_frames(save_root=".")
np.save("actions.npy", np.array(lst_actions))
np.save("directions.npy", np.array(lst_dirs))

env.close()










