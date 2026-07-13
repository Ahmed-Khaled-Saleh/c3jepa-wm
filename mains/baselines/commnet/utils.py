

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
from omegaconf import OmegaConf


import gymnasium as gym
import multigrid.envs
from multigrid.wrappers.external import PettingZooWrapper

# --------------------------------------------------------------------------
# Env construction -- adapt to your actual MultiGrid + PettingZooWrapper.
# --------------------------------------------------------------------------
def make_env_fn(env_cfg):
    """
    Returns a zero-arg factory that builds one PettingZoo-wrapped
    `FindGoalEnv` instance, with `env_cfg.num_agents` agents, an
    `env_cfg.view_size` x `env_cfg.view_size` egocentric view rendered at
    `env_cfg.tile_size` px/cell (so `view_size * tile_size == 224`).

    `max_steps`/`joint_reward` are passed straight through from
    `env_cfg` rather than hardcoded, so they can't silently drift out of
    sync with `cfg.episode_len` the way they did before (the env's own
    truncation limit and `_reward()`'s time-decay both key off
    `max_steps`; see `EnvConfig.max_steps`'s docstring comment).
    """
    

    def _make():
        env = gym.make(
            'MultiGrid-FindGoal-15x15-v0',
            agents=env_cfg.num_agents,
            render_mode='rgb_array',
            num_obstacles=env_cfg.num_obstacles,
            width=env_cfg.width,
            height=env_cfg.height,
            max_steps=env_cfg.max_steps,
            joint_reward=env_cfg.joint_reward,
        )
        return PettingZooWrapper(env)
    return _make


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


# --------------------------------------------------------------------------
# Checkpointing.
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer: optim.Optimizer,
                     cfg, update: int, mean_return: float) -> None:
    """
    Saves everything needed to resume training or run evaluation:
    model + optimizer state, the resolved config (so eval doesn't need to
    guess hyperparameters like hidden_dim/num_comm_steps), and bookkeeping
    (update step, the mean return at save time).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": update,
        "mean_return": mean_return,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, path)


def load_checkpoint(path: Path, device: torch.device):
    """
    Loads a checkpoint saved by `save_checkpoint`. Rebuild the model with
    the saved cfg before calling this, e.g. (for an eval script):

        ckpt = load_checkpoint(path, device)
        cfg = OmegaConf.create(ckpt["cfg"])
        model = CommNetAgent(cfg.hidden_dim, cfg.env.num_actions,
                              cfg.num_comm_steps, cfg.tie_weights,
                              cfg.gradient_checkpoint_encoder).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    """
    return torch.load(path, map_location=device)
