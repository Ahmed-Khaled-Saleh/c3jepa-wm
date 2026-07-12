# CommNet for pixel-based MultiGrid (goal-configuration task)

Implementation of **CommNet** (Sukhbaatar, Fergus, Szlam, Fergus, *"Learning
Multiagent Communication with Backpropagation"*, NeurIPS 2016) for a
MultiGrid environment where each agent observes its local 7x7 grid
neighborhood as a **224x224 RGB image**, and the team's objective is to
reach a target joint configuration (each agent at its assigned goal cell).

## Files

```
commnet/
  encoder.py   ViTTinyEncoder: image -> feature vector (per-agent, weight-shared)
  model.py     CommNetLayer / CommNetCore / CommNetActorCritic (the paper's core algorithm)
  rollout.py   RolloutBuffer: n-step storage + GAE advantage computation
train.py       EnvAdapter + A2C training loop wiring everything together
```

## Paper -> code mapping

| Paper concept | Code |
|---|---|
| Per-agent view / state | `obs[b, i]`: `(224, 224, 3)` uint8 image, the agent's 7x7-cell egocentric render |
| `h_i^0 = encoder(state_i)` | `CommNetAgent.encode()` -> `ViTTinyEncoder.forward` |
| Communication step `c_i^j = mean_{i'≠i} h_{i'}^j` | `_masked_mean_others` in `model.py` |
| `h_i^{j+1} = sigma(H^j h_i^j + C^j c_i^j)` | `CommNetLayer.forward` |
| `K` communication steps | `CommNetCore(num_comm_steps=K)` |
| Weight sharing across agents | Same `nn.Linear` applied to every agent slice — no per-agent parameters anywhere |
| Policy `pi_i`, baseline `V_i` | `CommNetActorCritic.policy_head` / `.value_head` |
| Variable / local connectivity | `alive_mask` (B,N) or full `comm_mask` (B,N,N) passed through every layer |
| REINFORCE + baseline, `R_t = sum_{i=t}^{T} r(i)` (undiscounted, Appendix A Eq. 7) | `RolloutBuffer.compute_mc_returns` + `train.py::compute_reinforce_loss` |

## Design choices worth knowing about

- **Encoder is ViT-Tiny** (`timm`'s `vit_tiny_patch16_224` if `timm` is
  installed, else a small built-in fallback with the same shape) run
  **independently per agent, with shared weights** — exactly what
  "assume you have a ViT-Tiny encoder" means here: it's a drop-in
  `f_enc: image -> R^d`, swap it for anything else (a CNN, a bigger ViT,
  a frozen pretrained backbone) without touching `model.py`.
- **Untied communication weights by default** (`tie_weights=False`): each
  of the `K` communication steps gets its own `H^j, C^j`, which is the
  variant the paper reports as at least as good as full tying. Pass
  `tie_weights=True` if you want a single shared step applied `K` times
  (fewer parameters, and lets you change `K` at inference time).
- **Skip connections** (`use_skip=True`) add the original encoder output
  `h^0` back in at every communication step, which the paper notes helps
  stability for larger `K`.
- **Masking for locality**: since agents only see a 7x7 window, you may
  want communication itself to be local (only agents within range hear
  each other) rather than the paper's default fully-connected mean-field
  broadcast. Pass a `(B, N, N)` boolean `comm_mask` into
  `CommNetActorCritic.forward` / `.act` to do this — `mask[b, i, k]=True`
  means agent `k`'s hidden state contributes to agent `i`'s `c_i`. Leave
  it `None` for the paper's original fully-connected broadcast.
- **Training algorithm matches the paper exactly** (Appendix A, Eq. 7):
  REINFORCE with a learned state-specific baseline, over the
  **undiscounted** sum of rewards from t to the end of the episode,
  `R_t = sum_{i=t}^{T} r(i)` — no TD bootstrapping, no GAE. The baseline
  is trained by gradient descent on `(R_t - b(s_t))^2`, weighted by
  `alpha = 0.03` (the paper's value in all its experiments), while the
  policy term uses `(R_t - b(s_t))` as the (un-normalized) advantage.
  Concretely: `train()` resets all envs and collects one full episode of
  length `episode_len` per update, then does a single REINFORCE+baseline
  gradient step — see `commnet/rollout.py::compute_mc_returns` and
  `train.py::compute_reinforce_loss`. A bootstrapped n-step GAE variant
  (`RolloutBuffer.compute_gae`) is also included as an optional,
  non-paper, more sample-efficient alternative if you want to deviate
  from the paper later; it is *not* wired into `train()` by default.

## Wiring up your MultiGrid env

`train.py`'s `EnvAdapter` expects, per env instance:

```python
obs, info = env.reset()
# obs: dict {agent_id: (224, 224, 3) uint8} or array (N, 224, 224, 3)

obs, rewards, terminated, truncated, info = env.step(actions)
# actions: dict {agent_id: int} or array (N,)
```

Edit `make_env_fn()` at the bottom of `train.py` to construct your actual
MultiGrid goal-configuration env (7x7 agent view, rendered at 32px/tile so
`7*32=224`), e.g.:

```python
from multigrid.envs import GoalConfigEnv

def make_env_fn():
    return GoalConfigEnv(
        num_agents=cfg.num_agents,
        agent_view_size=7,
        tile_size=32,
        render_mode="rgb_array",
    )
```

Reward design for the goal-configuration task is up to you/your env, but a
common choice that works well with actor-critic: `+1` (or a small dense
shaping term, e.g. `-distance_to_goal`) per agent per step it is at its
assigned cell, `0` otherwise, with the episode ending once all agents are
simultaneously at their goals (or on timeout).

## Running

```bash
pip install torch timm   # timm optional but recommended for real ViT-Tiny weights
python train.py --num-envs 8 --num-agents 4 --num-comm-steps 2 --episode-len 40
```

## Practical notes

- **Sample efficiency**: training a ViT-Tiny encoder from scratch purely
  from RL reward is slow. Options, roughly in order of effort:
  1. `--freeze-encoder` with `pretrained=True` (needs `timm`) and only
     train the projection head + CommNet — cheapest, works if the visual
     task is simple (colored grid cells, goal markers).
  2. Warm up the encoder with a short auxiliary self-supervised or
     supervised pass (e.g. predict agent/goal position from the image)
     before RL.
  3. Full end-to-end fine-tuning once the policy has learned something
     with (1) or (2).
- **Batching**: the encoder is called once per (env, agent) per step —
  `B*N` images through ViT-Tiny per rollout step. For `B=8, N=4` that's
  32 forward passes per step; batch them (already done, via the
  `reshape(B*N, ...)` in `CommNetAgent.encode`) rather than looping in
  Python.
- **Variable / dying agents**: if agents that reach their goal are removed
  from the episode rather than just frozen in place, pass the appropriate
  `alive_mask` into `model.act(...)` / `model(...)` and into
  `RolloutBuffer.add(..., alive_mask=...)` so dead agents don't pollute
  the communication average or the loss.
