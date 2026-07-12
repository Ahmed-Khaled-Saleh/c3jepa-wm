"""
CommNet (Sukhbaatar, Fergus, Szlam, Fergus - "Learning Multiagent
Communication with Backpropagation", NeurIPS 2016).

Paper recap, mapped onto the code below:

  h_i^0        = encoder(obs_i)                      -- ViTTinyEncoder, per agent
  c_i^j        = mean_{i' != i} h_{i'}^j              -- communication vector
  h_i^{j+1}    = sigma(H^j h_i^j + C^j c_i^j)         -- CommNetLayer.forward
  ... repeated for j = 0 .. K-1 communication steps   -- CommNetCore
  pi_i, V_i    = policy_head(h_i^K), value_head(h_i^K) -- CommNetActorCritic

Notes / design choices:
  * H^j, C^j are untied across communication steps by default (the paper
    reports this ("unshared") variant works at least as well as full
    weight tying; a `tie_weights=True` flag is provided for the tied
    variant, which reduces parameter count and lets K vary at eval time).
  * Agents that don't exist / are masked out (dead, out of comm range) can
    be excluded from both the mean in c_i^j and from receiving updates by
    passing a boolean `alive_mask` of shape (B, N). This also lets you
    restrict communication to a local neighborhood (e.g. agents within the
    7x7 vision window of each other) via a full (B, N, N) `comm_mask`
    instead of the default fully-connected mean-field mask -- see
    `CommNetLayer.forward`.
  * Everything operates on a fixed but arbitrary number of agents N per
    forward call, batched over B independent episodes/environments.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean_others(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    h:    (B, N, D) hidden states
    mask: (B, N) boolean "alive"/"present" mask, or None (all agents present).
          Optionally (B, N, N) boolean "can-communicate-with" mask, where
          mask[b, i, k] = True means agent k's message reaches agent i.

    Returns c: (B, N, D), c[b, i] = mean over eligible k != i of h[b, k].
    Agents with zero eligible neighbours get c = 0 (matches the paper's
    convention that an isolated agent just uses its own hidden state).
    """
    B, N, D = h.shape
    device = h.device

    if mask is None:
        pair_mask = torch.ones(B, N, N, dtype=torch.bool, device=device)
    elif mask.dim() == 2:
        # (B, N) alive mask -> broadcast to pairwise "both alive" mask
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
    else:
        pair_mask = mask  # already (B, N, N)

    # exclude self-loops (i communicates with others, not itself, in c_i)
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    pair_mask = pair_mask & (~eye)

    counts = pair_mask.sum(dim=2, keepdim=True).clamp(min=1).float()  # (B, N, 1)
    summed = torch.bmm(pair_mask.float(), h)                          # (B, N, D)
    c = summed / counts
    # zero out agents that truly have no neighbours (avoid spurious signal)
    has_any = (pair_mask.sum(dim=2, keepdim=True) > 0).float()
    return c * has_any


class CommNetLayer(nn.Module):
    """
    One communication step: h^{j+1}_i = sigma(H h_i^j + C c_i^j).

    Optionally adds a skip connection from the original encoder output
    (h^0), which the paper found helps for larger K.
    """

    def __init__(self, hidden_dim: int, activation: str = "tanh",
                 use_skip: bool = True):
        super().__init__()
        self.H = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.use_skip = use_skip
        if use_skip:
            self.S = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = torch.tanh if activation == "tanh" else F.relu

    def forward(self, h: torch.Tensor, h0: Optional[torch.Tensor],
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        c = _masked_mean_others(h, mask)
        out = self.H(h) + self.C(c)
        if self.use_skip and h0 is not None:
            out = out + self.S(h0)
        return self.act(out)


class CommNetCore(nn.Module):
    """
    Stacks K communication steps on top of per-agent encoder features.
    Weight-shared across agents by construction (the same Linear layers
    are applied to every agent's slice of the (B, N, D) tensor).
    """

    def __init__(self, hidden_dim: int, num_comm_steps: int = 2,
                 activation: str = "tanh", use_skip: bool = True,
                 tie_weights: bool = False):
        super().__init__()
        self.num_comm_steps = num_comm_steps
        self.tie_weights = tie_weights
        if tie_weights:
            self.layer = CommNetLayer(hidden_dim, activation, use_skip)
            self.layers = None
        else:
            self.layers = nn.ModuleList([
                CommNetLayer(hidden_dim, activation, use_skip)
                for _ in range(num_comm_steps)
            ])
            self.layer = None

    def forward(self, h0: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h0:   (B, N, D) per-agent encoder features.
        mask: (B, N) alive mask or (B, N, N) pairwise comm mask, or None.
        Returns h_K: (B, N, D) communicated hidden states.
        """
        h = h0
        for j in range(self.num_comm_steps):
            layer = self.layer if self.tie_weights else self.layers[j]
            h = layer(h, h0, mask)
        return h


class CommNetActorCritic(nn.Module):
    """
    Full agent: shared encoder -> CommNet core -> per-agent policy & value
    heads. All weights are shared across agents (parameter sharing is what
    lets CommNet scale to a variable number of homogeneous agents).

    forward() expects already-encoded features (B, N, hidden_dim) so this
    module is agnostic to what the encoder is; wire it up with
    ViTTinyEncoder in the training loop (see train.py).
    """

    def __init__(self, hidden_dim: int, num_actions: int,
                 num_comm_steps: int = 2, activation: str = "tanh",
                 use_skip: bool = True, tie_weights: bool = False):
        super().__init__()
        self.core = CommNetCore(hidden_dim, num_comm_steps, activation,
                                 use_skip, tie_weights)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, h0: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        h0:   (B, N, hidden_dim) per-agent encoder output.
        mask: (B, N) alive mask, (B, N, N) comm mask, or None.

        Returns:
            logits: (B, N, num_actions) per-agent action logits
            value:  (B, N) per-agent value estimate V_i(s)
        """
        h_K = self.core(h0, mask)
        logits = self.policy_head(h_K)
        value = self.value_head(h_K).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, h0: torch.Tensor, mask: Optional[torch.Tensor] = None,
            deterministic: bool = False):
        """
        Convenience method for rollout collection.
        Returns actions (B, N) long, log_probs (B, N), value (B, N).
        """
        logits, value = self.forward(h0, mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, value
