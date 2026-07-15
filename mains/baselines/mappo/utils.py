
import time
from pathlib import Path

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from torchrl.envs import RewardSum, TransformedEnv
import gymnasium as gym
import multigrid.envs  # noqa: F401 -- registers 'MultiGrid-FindGoal-15x15-v0'
from multigrid.wrappers.external import TorchRLPettingZooWrapper
from torchrl.envs.libs import pettingzoo as torchrl_pettingzoo


# --------------------------------------------------------------------------
# TorchRL env construction on top of train.py's gym env.
# --------------------------------------------------------------------------
# Assumed defaults -- see debug_env_structure() below. Adjust if your
# actual multigrid PettingZoo agent naming differs from "agent_0", "agent_1"
# (see the "IMPORTANT -- key-path assumptions" note at the top of this file).
GROUP = "agent"


def _keys(group: str = GROUP):
    return {
        "obs": (group, "observation", "pov"),
        "action": (group, "action"),
        "reward": (group, "reward"),
        "done": (group, "done"),
        "terminated": (group, "terminated"),
        "truncated": (group, "truncated"),
        "hidden": (group, "hidden"),
        "logits": (group, "logits"),
        "value": (group, "state_value"),
        "episode_reward": (group, "episode_reward"),
    }


def validate_episode_budget(cfg) -> None:
    """
    cfg.episode_len (the outer training/eval loop's step budget) and
    cfg.env.max_steps (the env's own internal truncation limit, passed to
    gym.make in make_env_fn) must match. If the outer loop's budget is
    smaller, episodes get cut off before the env itself ever gets a
    chance to truncate or for agents to reach a goal that's genuinely far
    away -- this is exactly what silently happened before (episode_len=40
    vs. the env's default max_steps=250), and reward/success metrics from
    a mismatched run aren't a fair read on the policy.
    """
    if cfg.episode_len != cfg.env.max_steps:
        raise ValueError(
            f"cfg.episode_len ({cfg.episode_len}) != cfg.env.max_steps "
            f"({cfg.env.max_steps}) -- the outer loop's step budget must "
            f"match the env's own internal max_steps, or episodes get cut "
            f"off before the env (and _reward()'s time-decay, which is "
            f"computed against env.max_steps) ever sees the full horizon "
            f"it was configured for. Set them equal, e.g. via "
            f"`python train.py episode_len=150 env.max_steps=150`."
        )




def make_torchrl_env_fn(env_cfg, fixed_seed: int | None = None):
    """
    Builds the base gym `FindGoalEnv` with the same kwargs as train.py's
    `make_env_fn` (max_steps/joint_reward/num_obstacles/width/height, all
    from `env_cfg`), but wraps it with `TorchRLPettingZooWrapper` --
    NOT `train.py`'s `make_env_fn` itself, which wraps with the plain
    `PettingZooWrapper` (integer agent IDs, correct for the CommNet /
    `MultiAgentEnvPool` path but wrong here). TorchRL's own
    `PettingZooWrapper` requires string agent IDs (it calls
    `.split("_")` on each entry of `possible_agents` to infer the default
    group name), which only `TorchRLPettingZooWrapper` provides -- passing
    the int-keyed wrapper produces
    `AttributeError: 'int' object has no attribute 'split'`.

    Then adds `RewardSum` so episode-total reward is tracked automatically
    (mirrors the tutorial's transform).

    fixed_seed: if set, EVERY reset of this env -- including
    `SyncDataCollector`'s own internal auto-resets whenever an episode
    ends during training, which this module has no direct call-site
    access to -- is forced onto this exact seed, giving a fixed grid/
    obstacle/spawn layout across the whole run (only the goal position
    still varies episode to episode, via FindGoalEnv's separate
    `_goal_rng`; see chat). A gymnasium-style `env.reset(seed=X)` only
    fixes the layout for that ONE reset -- every later bare `env.reset()`
    (no seed re-passed) just advances the same underlying RNG to a NEW
    state, not back to X's state, which is what a naive `env.set_seed(X)`
    call (torchrl's usual seeding entrypoint) would give you: a
    reproducible SEQUENCE of different layouts, not one repeated fixed
    layout. `rollout()`'s own `auto_reset=True` path also has no
    parameter for injecting a seed into its internal reset call at all
    (checked against the actual torchrl source). Monkey-patching
    `_reset_parallel` on this specific env instance is the one place
    that's guaranteed to see every reset the whole pipeline ever
    triggers, regardless of caller, and override the seed unconditionally.
    """
    import gymnasium as gym
    import multigrid.envs  # noqa: F401 -- registers 'MultiGrid-FindGoal-15x15-v0'
    from multigrid.wrappers.external import TorchRLPettingZooWrapper
    from torchrl.envs.libs import pettingzoo as torchrl_pettingzoo

    def _make():
        base_gym_env = gym.make(
            'MultiGrid-FindGoal-15x15-v0',
            agents=env_cfg.num_agents,
            render_mode='rgb_array',
            num_obstacles=env_cfg.num_obstacles,
            width=env_cfg.width,
            height=env_cfg.height,
            max_steps=env_cfg.max_steps,
            joint_reward=env_cfg.joint_reward,
        )
        pz_env = TorchRLPettingZooWrapper(base_gym_env)  # string agent IDs: "agent_0", "agent_1", ...
        env = torchrl_pettingzoo.PettingZooWrapper(
            env=pz_env,
            return_state=False,
            group_map=None,
            use_mask=False,
        )

        if fixed_seed is not None:
            _original_reset_parallel = env._reset_parallel

            def _seeded_reset_parallel(**kwargs):
                kwargs["seed"] = fixed_seed  # override whatever was passed (or nothing)
                return _original_reset_parallel(**kwargs)

            env._reset_parallel = _seeded_reset_parallel

        env = TransformedEnv(
            env,
            RewardSum(in_keys=[env.reward_key], out_keys=[_keys()["episode_reward"]]),
        )
        return env
    return _make


def debug_env_structure(cfg) -> None:
    """
    Run this FIRST (e.g. `python -c "from train_mappo import *; ..."` or
    call from a notebook) before trusting GROUP/*_KEY above. Prints
    `env.group_map` and one reset+step tensordict so you can confirm the
    actual key paths for your installed multigrid/torchrl versions.
    """
    env = make_torchrl_env_fn(cfg.env)()
    print("group_map:", env.group_map)
    td = env.reset()
    print("reset() tensordict:\n", td)
    td = env.rand_step(td)
    print("step() tensordict:\n", td)
    env.close()



# --------------------------------------------------------------------------
# Checkpointing (mirrors train.py's save_checkpoint/load_checkpoint).
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, policy: nn.Module, value_module: nn.Module,
                     optimizer: torch.optim.Optimizer, cfg,
                     frames_seen: int, mean_return: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "frames_seen": frames_seen,
        "mean_return": mean_return,
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, path)