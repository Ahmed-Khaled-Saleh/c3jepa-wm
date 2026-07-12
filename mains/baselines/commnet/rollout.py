"""
Fixed-length rollout buffer for synchronous multi-agent policy-gradient
training.

Shapes follow (T, B, N, ...):
    T = number of steps stored (set this to the environment's max episode
        length, e.g. 40 for the paper's traffic-junction/combat tasks, so
        that each buffer holds one full episode per env -- see
        `compute_mc_returns` below, which is the paper's actual training
        signal, Appendix A Eq. 7)
    B = number of parallel environments
    N = number of agents (assumed fixed per env; use `alive_mask` for
        agents that finish early, e.g. reach their goal cell and are
        removed from the episode)
"""
from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(self, T: int, B: int, N: int, obs_shape,
                 device: torch.device):
        self.T, self.B, self.N = T, B, N
        self.device = device

        self.obs = torch.zeros((T, B, N, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((T, B, N), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((T, B, N), dtype=torch.float32, device=device)
        self.values = torch.zeros((T, B, N), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((T, B, N), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, B), dtype=torch.float32, device=device)  # per-env episode termination
        self.alive_mask = torch.ones((T, B, N), dtype=torch.bool, device=device)

        self.ptr = 0

    def add(self, obs, actions, log_probs, values, rewards, done, alive_mask=None):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = done
        if alive_mask is not None:
            self.alive_mask[t] = alive_mask
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.T

    def reset(self):
        self.ptr = 0

    @torch.no_grad()
    def compute_mc_returns(self, gamma: float = 1.0) -> torch.Tensor:
        """
        This is what the CommNet paper actually uses (Appendix A, Eq. 7):
        for each time t in the episode, the target is the *undiscounted*
        sum of rewards from t to the end of the episode T,

            R_t = sum_{i=t}^{T} r(i)

        with NO bootstrapping from a learned value function -- the whole
        buffer must contain complete episode(s) for this to be correct
        (i.e. call this only after an episode has actually terminated /
        been truncated within the buffer, not on an arbitrary n-step
        window). The paper doesn't discount within an episode, so the
        default here is gamma=1.0 to match; pass gamma<1 only if you
        deliberately want to deviate from the paper.

        `dones[t]` marks the last step of an episode for a given env
        (b). Returns are computed backwards and reset (not accumulated
        across) episode boundaries, so a buffer holding several
        back-to-back episodes for the same env is handled correctly.

        Returns: R of shape (T, B, N), zeroed out for non-`alive_mask`
        (agent, step) entries.
        """
        T, B, N = self.T, self.B, self.N
        returns = torch.zeros((T, B, N), device=self.device)
        running = torch.zeros((B, N), device=self.device)

        for t in reversed(range(T)):
            not_done = (1.0 - self.dones[t]).unsqueeze(-1)  # (B, 1) -> broadcasts over N
            running = self.rewards[t] + gamma * running * not_done
            returns[t] = running * self.alive_mask[t]

        return returns

    @torch.no_grad()
    def compute_gae(self, last_values: torch.Tensor, gamma: float = 0.99,
                     lam: float = 0.95):
        """
        NOTE: this is NOT what the CommNet paper uses -- the paper trains
        with plain REINFORCE + baseline over full undiscounted episode
        returns (see `compute_mc_returns`). This bootstrapped n-step GAE
        variant is provided as an optional, more sample-efficient
        alternative in case you want to deviate from the paper; swap
        `compute_mc_returns` for this in `train.py` if so.

        last_values: (B, N) bootstrap value for the state after the last
                     stored step.
        Returns advantages (T, B, N) and returns (T, B, N), both detached.
        Dead agents (alive_mask == False) get advantage/return = 0 and are
        excluded from the policy loss via the mask returned by caller.
        """
        T, B, N = self.T, self.B, self.N
        advantages = torch.zeros((T, B, N), device=self.device)
        gae = torch.zeros((B, N), device=self.device)

        for t in reversed(range(T)):
            not_done = (1.0 - self.dones[t]).unsqueeze(-1)  # (B, 1) -> broadcasts over N
            next_values = last_values if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae * self.alive_mask[t]

        returns = advantages + self.values
        return advantages, returns
