"""
Wraps FindGoalEnv (via multigrid's plain, int-keyed `PettingZooWrapper`) to
satisfy `og_marl.wrapped_environments.base.BaseEnvironment`'s interface --
needed ONLY for the periodic evaluation rollouts `BaseOfflineSystem.evaluate()`
runs during `system.train(buffer, ...)`. Training itself is 100%
dataset-driven (FindGoalOfflineDataset + OfflineReplayBuffer); this wrapper
never touches the offline data, only live env interaction for measuring
the current policy's actual performance.

Key interface points, and why they're implemented the way they are:

- `BaseEnvironment.__getattr__` falls back to `getattr(self._environment, name)`
  for any attribute not explicitly defined on the subclass -- so the wrapped
  PettingZooWrapper instance MUST be stored as `self._environment` (that
  exact name is hardcoded in the base class), not `self._env` or `self.env`,
  for that fallback to work for anything not explicitly proxied below.

- Agent IDs stay INTEGER (0, 1, ...), matching FindGoalEnv's own native
  `AgentID = int` and the plain `PettingZooWrapper`'s convention (as
  opposed to `TorchRLPettingZooWrapper`'s "agent_0"/"agent_1" strings,
  used elsewhere for torchrl compatibility). `BaseEnvironment`'s type hints
  say `Dict[str, ...]` but nothing in `BaseOfflineSystem`/`ViTIQLCQLSystem`
  actually requires string keys -- they just iterate `environment.agents`
  and do dict lookups with whatever key type it returns, so int keys work
  fine and avoid introducing a third agent-ID convention into this project.

- `FindGoalEnv`'s own `terminations`/`truncations` dicts (from the base
  `MultiGridEnv.step()`: `dict(enumerate(self.agent_states.terminated))`
  etc.) are ALREADY per-agent, int-keyed, and already match
  `BaseEnvironment`'s `Terminals`/`Truncations` types exactly -- passed
  through unchanged, no reshaping needed (only the observations dict needs
  reshaping, to pull out just the `pov` field per agent).

- `infos["legals"]` is synthesized as all-True every step: FindGoalEnv/
  NavigationAction has no illegal-action concept (walls block movement
  silently rather than disallowing the action), same reasoning as
  `FindGoalOfflineDataset`'s legals field.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from og_marl.wrapped_environments.base import BaseEnvironment


class OGMARLFindGoalWrapper(BaseEnvironment):
    """
    Args:
        num_agents, num_obstacles, width, height, max_steps, joint_reward:
            passed straight through to `gym.make('MultiGrid-FindGoal-15x15-v0', ...)`,
            same parameter set as `train.py`/`train_mappo.py`'s `make_env_fn`
            -- keep these equal to whatever config produced the offline
            dataset/trained encoder, or evaluation isn't measuring the same
            task the policy was trained on.
        num_actions: NavigationAction has 4 actions (left, right, forward,
            done).
        obs_key: which field of FindGoalEnv.gen_obs()'s per-agent obs dict
            to expose as the observation -- "pov" (RGB) for the ViT-fronted
            system.
    """

    def __init__(
        self,
        num_agents: int = 2,
        num_actions: int = 4,
        num_obstacles: int = 6,
        width: int = 15,
        height: int = 15,
        max_steps: int = 150,
        joint_reward: bool = True,
        obs_key: str = "pov",
        render_mode: str = "rgb_array",
    ):
        import gymnasium as gym
        import multigrid.envs  # noqa: F401 -- registers 'MultiGrid-FindGoal-15x15-v0'
        from multigrid.wrappers.external import PettingZooWrapper

        base_gym_env = gym.make(
            'MultiGrid-FindGoal-15x15-v0',
            agents=num_agents,
            render_mode=render_mode,
            num_obstacles=num_obstacles,
            width=width,
            height=height,
            max_steps=max_steps,
            joint_reward=joint_reward,
        )
        # MUST be named exactly `_environment` -- see module docstring,
        # BaseEnvironment.__getattr__ hardcodes this name for its fallback
        # attribute delegation.
        self._environment = PettingZooWrapper(base_gym_env)

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.obs_key = obs_key
        self.agents = list(range(num_agents))  # int-keyed, see module docstring

    def _extract_observations(self, raw_obs: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
        return {ag: raw_obs[ag][self.obs_key] for ag in self.agents}

    def _legals_infos(self) -> Dict[str, Dict[int, np.ndarray]]:
        return {
            "legals": {ag: np.ones(self.num_actions, dtype=bool) for ag in self.agents}
        }

    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        raw_obs, _env_info = self._environment.reset(seed= 0)
        observations = self._extract_observations(raw_obs)
        infos = self._legals_infos()
        return observations, infos

    def step(
        self, actions: Dict[int, Any]
    ) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[str, Any],
    ]:
        raw_obs, rewards, terminations, truncations, _env_info = self._environment.step(actions)
        observations = self._extract_observations(raw_obs)
        infos = self._legals_infos()
        # rewards/terminations/truncations are passed through UNCHANGED --
        # FindGoalEnv's own step() already returns these per-agent,
        # int-keyed, matching BaseEnvironment's expected types exactly
        # (see module docstring).
        return observations, rewards, terminations, truncations, infos

    def get_stats(self) -> Dict:
        return {}

    def render(self):
        return self._environment.render()