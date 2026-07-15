"""
Multi-Agent PPO (MAPPO), adapted from the TorchRL tutorial
(https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html) for our
FindGoalEnv (via `multigrid.wrappers.external.TorchRLPettingZooWrapper` ->
`torchrl.envs.libs.pettingzoo.PettingZooWrapper`), reusing the same
ViT-Tiny pixel encoder as `train.py`'s CommNet setup.

What's different from the tutorial, and why:
  - The tutorial's env (VMAS) gives low-dimensional vector observations fed
    directly into `MultiAgentMLP`. Ours gives 224x224x3 RGB images per
    agent, so a `VitEncoderModule` (architecturally identical to
    `CommNetAgent.encode` in train.py -- same `vit_hf` backbone, same
    `img_transform`) sits in front of both the actor and critic
    `MultiAgentMLP`s, mapping image -> `hidden_dim` feature per agent.
    Actor and critic each get their OWN encoder instance (not shared) --
    see the note above `build_networks` for why sharing one instance
    between them is a real foot-gun with `ClipPPOLoss`'s default
    `functional=True` behavior.
  - MAPPO (vs IPPO) = centralized critic (`centralized=True` in the
    critic's `MultiAgentMLP` -- it sees every agent's encoded feature
    concatenated, not just its own) + decentralized actor
    (`centralized=False`, each agent only conditions on its own encoded
    feature; weights are still shared across agents via
    `share_params=True`, matching CommNet's own weight-sharing
    assumption and the paper's homogeneous-policy setup).

IMPORTANT -- key-path assumptions, verified against torchrl==0.13.2's
`PettingZooWrapper` source, but STILL DEPEND on your specific
`multigrid.wrappers.external.TorchRLPettingZooWrapper` naming agents like
"agent_0", "agent_1", ... (the convention PettingZoo's default grouping
splits on to build the group name "agent" -- see `_get_default_group_map`
in torchrl/envs/libs/pettingzoo.py). Run `debug_env_structure()` at the
bottom of this file FIRST and confirm/adjust the `GROUP`/`*_KEY` constants
below before trusting anything else in this file:

  GROUP        = "agent"                      -- group_map key; verify via env.group_map
  OBS_KEY      = (GROUP, "observation", "pov") -- gen_obs()'s RGB key, nested one level
                                                   under "observation" because FindGoalEnv's
                                                   per-agent obs space is a gym.spaces.Dict
  ACTION_KEY   = (GROUP, "action")
  REWARD_KEY   = (GROUP, "reward")
  DONE_KEY / TERMINATED_KEY / TRUNCATED_KEY = (GROUP, "done"/"terminated"/"truncated")

  categorical_actions defaults to True in PettingZooWrapper, so actions
  are encoded as plain categorical integers (not one-hot) -- matching
  `distribution_class=torch.distributions.Categorical` below. If you
  passed `categorical_actions=False`, switch to
  `torchrl.modules.OneHotCategorical` instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor  # noqa: F401 (not used, discrete actions)

from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv, ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators

import torchvision.transforms.v2 as v2
from stable_pretraining.backbone.utils import vit_hf

# Reuse the env-construction/config plumbing already validated in train.py
# (EnvConfig, make_env_fn -- gym.make(...) with max_steps/joint_reward
# wired through, and the episode_len == env.max_steps guard).
from train import EnvConfig, _validate_episode_budget


# --------------------------------------------------------------------------
# Same image encoder as CommNet's CommNetAgent.encode -- reused verbatim.
# --------------------------------------------------------------------------
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VitEncoderModule(nn.Module):
    """
    (*batch, n_agents, 224, 224, 3) uint8 -> (*batch, n_agents, hidden_dim)
    float feature per agent. Same backbone/preprocessing as
    `CommNetAgent.encode` in train.py, factored out standalone so it can
    sit inside a `TensorDictModule` for both the MAPPO actor and critic.

    `noise_std`: standard deviation of simple additive white Gaussian
    noise applied to the normalized image tensor before the backbone.
    0.0 (default) = no noise. This is used to simulate a noisy
    communication channel corrupting what the *centralized critic*
    receives (see `evaluate()`'s channel_noise_std) -- it's a plain
    instance attribute rather than a constructor-only setting so
    `evaluate()` can toggle it on/off around specific forward passes
    without rebuilding the module.

    `gradient_checkpoint`: trades compute for memory by recomputing
    backbone activations during backward instead of storing them, while
    still allowing full gradient flow -- important here since both the
    actor's and critic's encoders train from scratch (`pretrained=False`
    by default) and so can never be frozen; freezing would mean they never
    learn anything. With two independent ViT instances (actor + critic,
    intentionally not weight-shared -- see the note above `build_networks`),
    this matters at least as much here as in CommNet's single-encoder setup.
    """

    def __init__(self, hidden_dim: int, pretrained: bool = False, noise_std: float = 0.0,
                 gradient_checkpoint: bool = False):
        super().__init__()
        self.encoder = vit_hf(
            size="tiny", patch_size=14, image_size=224,
            pretrained=pretrained, use_mask_token=True,
        )
        self.processor = img_transform
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        self._manual_checkpoint = False
        if gradient_checkpoint:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            else:
                self._manual_checkpoint = True

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        # Defensive device move: SyncDataCollector's device-casting splits
        # into policy_device/env_device/storing_device, and whether the
        # observation actually arrives on this module's device by the time
        # it reaches forward() depends on internal torchrl casting behavior
        # for a CPU-native gym env that isn't fully pinned down here (can't
        # verify against a live run in this sandbox). Moving explicitly,
        # unconditionally, sidesteps needing to get that exactly right --
        # this is a no-op (near-zero cost) if it's already on the right
        # device, and fixes a real
        # "Input type torch.FloatTensor and weight type torch.cuda.FloatTensor"
        # crash if it wasn't.
        obs_uint8 = obs_uint8.to(next(self.parameters()).device)

        *batch, n_agents, H, W, C = obs_uint8.shape
        x = obs_uint8.reshape(-1, H, W, C).movedim(-1, -3)  # (*batch*n_agents, 3, 224, 224)
        x = self.processor(x)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std  # simple AWGN "channel effect"

        if self._manual_checkpoint and self.training:
            def _encoder_fwd(inp):
                return self.encoder(inp)['last_hidden_state']
            hidden = torch.utils.checkpoint.checkpoint(_encoder_fwd, x, use_reentrant=False)
        else:
            hidden = self.encoder(x)['last_hidden_state']

        feats = hidden[:, 0]  # CLS token, (*batch*n_agents, hidden)
        return feats.reshape(*batch, n_agents, self.hidden_dim)


# --------------------------------------------------------------------------
# Config (Hydra structured config), extending train.py's EnvConfig.
# --------------------------------------------------------------------------
@dataclass
class MAPPOConfig:
    env: EnvConfig = field(default_factory=EnvConfig)

    num_envs: int = 8   # NOTE: currently UNUSED -- SyncDataCollector below is given a single-env
                         # factory (env_fn), not wrapped in ParallelEnv/SerialEnv, so collection
                         # runs from exactly one env sequentially regardless of this value. Not a
                         # memory risk as-is (it isn't silently multiplying GPU usage), but don't
                         # expect changing this to do anything until real vectorization is added.
    hidden_dim: int = 192
    pretrained_encoder: bool = False
    gradient_checkpoint_encoder: bool = True  # see VitEncoderModule's docstring -- applies to
                                               # BOTH the actor's and critic's (separate) encoders

    actor_depth: int = 2
    actor_num_cells: int = 256
    critic_depth: int = 2
    critic_num_cells: int = 256

    # PPO hyperparameters (tutorial defaults, adjust as needed)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 1.0
    normalize_advantage: bool = True
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # frames_per_batch/minibatch_size are the real memory levers here (unlike CommNet's
    # single-shot buffer recompute, PPO already minibatches -- the risk is GAE's one-shot
    # no_grad forward over the whole frames_per_batch batch, and each minibatch_size *
    # n_agents * 2 (separate actor + critic ViT encoders) images per gradient step).
    frames_per_batch: int = 1_200
    total_frames: int = 3_000_000
    num_epochs: int = 4              # PPO passes over each collected batch (compute, not memory)
    minibatch_size: int = 150

    episode_len: int = 150           # must equal env.max_steps -- see train.py's _validate_episode_budget
    device: str = "auto"
    seed: int | None = None          # also fixes the env's grid/obstacle/spawn layout across the
                                       # whole run (goal position still varies -- see
                                       # make_torchrl_env_fn's docstring), not just torch's RNG
    log_every: int = 1               # in outer collector iterations, not frames

    eval_every: int = 20             # outer iterations; 0 disables periodic eval
    eval_episodes: int = 20
    eval_deterministic: bool = True  # ExplorationType.MODE (argmax) vs sampling
    eval_channel_noise_std: float = 0.0  # AWGN std added to the CENTRALIZED CRITIC's
                                          # observations only during evaluation, simulating a
                                          # noisy channel; does not affect action selection
                                          # (the decentralized actor always sees clean obs).
                                          # 0.0 = no noise, matches training behavior.

    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 20       # outer iterations
    project_name: str = "commnet-mappo"


cs = ConfigStore.instance()
cs.store(name="mappo_schema", node=MAPPOConfig)


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


def make_torchrl_env_fn(env_cfg: EnvConfig, fixed_seed: int | None = None):
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


def debug_env_structure(cfg: MAPPOConfig) -> None:
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
# Networks: shared VitEncoderModule design note.
# --------------------------------------------------------------------------
# It's tempting to save compute by giving the actor and critic the SAME
# VitEncoderModule instance (one ViT forward pass feeding both heads,
# exactly how CommNetAgent shares its encoder between policy and value
# heads). Don't do this with ClipPPOLoss's default `functional=True`:
# torchrl "functionalizes" actor_network and critic_network SEPARATELY
# internally, which silently breaks weight sharing between them even if
# you passed the same nn.Module object into both TensorDictSequentials --
# you'd end up training two independently-diverging copies of what you
# thought was one shared encoder. If you want a shared trunk, you must
# pass `ClipPPOLoss(..., functional=False)` and verify parameter identity
# yourself; simpler and safer default here is two separate encoder
# instances (2x ViT compute, but correct and matches the tutorial's
# fully-independent actor/critic networks).
def build_networks(cfg: MAPPOConfig, n_agents: int, n_actions: int,
                    action_spec, device: torch.device):
    K = _keys()

    actor_encoder = TensorDictModule(
        VitEncoderModule(cfg.hidden_dim, cfg.pretrained_encoder,
                         gradient_checkpoint=cfg.gradient_checkpoint_encoder).to(device),
        in_keys=[K["obs"]], out_keys=[K["hidden"]],
    )
    actor_head = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=cfg.hidden_dim, n_agent_outputs=n_actions, n_agents=n_agents,
            centralized=False, share_params=True, device=device,
            depth=cfg.actor_depth, num_cells=cfg.actor_num_cells, activation_class=nn.Tanh,
        ),
        in_keys=[K["hidden"]], out_keys=[K["logits"]],
    )
    policy_module = TensorDictSequential(actor_encoder, actor_head)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=[K["logits"]],
        out_keys=[K["action"]],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    critic_encoder = TensorDictModule(
        VitEncoderModule(cfg.hidden_dim, cfg.pretrained_encoder,  # separate instance -- see note above
                         gradient_checkpoint=cfg.gradient_checkpoint_encoder).to(device),
        in_keys=[K["obs"]], out_keys=[K["hidden"]],
    )
    critic_head = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=cfg.hidden_dim, n_agent_outputs=1, n_agents=n_agents,
            centralized=True,  # MAPPO: critic conditions on every agent's encoded feature
            share_params=True, device=device,
            depth=cfg.critic_depth, num_cells=cfg.critic_num_cells, activation_class=nn.Tanh,
        ),
        in_keys=[K["hidden"]], out_keys=[K["value"]],
    )
    value_module = TensorDictSequential(critic_encoder, critic_head)

    return policy, value_module


# --------------------------------------------------------------------------
# Checkpointing (mirrors train.py's save_checkpoint/load_checkpoint).
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, policy: nn.Module, value_module: nn.Module,
                     optimizer: torch.optim.Optimizer, cfg: MAPPOConfig,
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


# --------------------------------------------------------------------------
# Evaluation: same quantities as CommNet's evaluate() in train.py, plus a
# centralized-critic-under-noisy-channel diagnostic.
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(policy, value_module, cfg: MAPPOConfig, num_episodes: int,
             deterministic: bool | None = None, channel_noise_std: float | None = None,
             device: torch.device | None = None) -> dict:
    """
    Runs `num_episodes` single-episode rollouts with `policy` and computes
    the same quantities as CommNet's `evaluate()` in train.py:
      - success_rate: every agent's own episode-end was `terminated`
        (reached the goal), none were `truncated` (timed out).
      - any_goal_reached_rate: at least one agent's `terminated` fired at
        some point (diagnostic -- lets you tell "nobody ever reaches the
        goal" apart from "the strict success criterion is over-filtering").
      - ended_early_rate: episode ended before `cfg.episode_len` for any
        reason.
      - mean_return, mean_episode_length, num_episodes.

    Action selection always uses CLEAN observations (`ExplorationType.MODE`
    for argmax when deterministic=True, matching what would actually be
    deployed -- the decentralized actor never sees noise).

    Additionally, and separately from action selection: after each
    rollout, the CENTRALIZED CRITIC is run twice, offline, over the
    recorded observations from that episode -- once clean, once with
    simple AWGN (`channel_noise_std`) added to the normalized image before
    the backbone (see `VitEncoderModule.noise_std`) -- to gauge how
    sensitive the trained value function is to a noisy channel corrupting
    what gets shared for centralized training/critique. This has NO effect
    on which actions were taken or on success_rate/mean_return above; it's
    a pure diagnostic on the critic. Returned as:
      - mean_critic_value_clean / mean_critic_value_noisy
      - mean_abs_critic_value_delta: mean(|V_clean - V_noisy|) -- larger
        means the critic's value estimate is more sensitive to channel
        noise.

    Args:
        deterministic: defaults to cfg.eval_deterministic if None.
        channel_noise_std: defaults to cfg.eval_channel_noise_std if None.
    """
    _validate_episode_budget(cfg)
    device = device or torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    deterministic = cfg.eval_deterministic if deterministic is None else deterministic
    channel_noise_std = cfg.eval_channel_noise_std if channel_noise_std is None else channel_noise_std

    K = _keys()
    was_policy_training = policy.training
    was_value_training = value_module.training
    policy.eval()
    value_module.eval()

    # value_module = TensorDictSequential(critic_encoder, critic_head);
    # critic_encoder is a TensorDictModule whose .module is our
    # VitEncoderModule -- reach in to toggle its AWGN noise_std around
    # specific forward passes only (see class docstring).
    critic_vit: VitEncoderModule = value_module[0].module
    original_noise_std = critic_vit.noise_std

    env = make_torchrl_env_fn(cfg.env, fixed_seed=cfg.seed)()

    successes, any_goal, ended_early = [], [], []
    returns, lengths = [], []
    value_deltas, clean_values, noisy_values = [], [], []

    exploration_type = ExplorationType.MODE if deterministic else ExplorationType.RANDOM
    try:
        with set_exploration_type(exploration_type):
            for _ in range(num_episodes):
                critic_vit.noise_std = 0.0  # rollout/action-selection path: always clean
                td = env.rollout(max_steps=cfg.episode_len, policy=policy, break_when_any_done=True)

                next_terminated = td.get(("next", *K["terminated"])).squeeze(-1)  # (T, n_agents)
                next_truncated = td.get(("next", *K["truncated"])).squeeze(-1)    # (T, n_agents)
                next_reward = td.get(("next", *K["reward"])).squeeze(-1)          # (T, n_agents)

                ever_terminated = next_terminated.any(dim=0)  # (n_agents,)
                ever_truncated = next_truncated.any(dim=0)
                episode_length = td.batch_size[0]

                successes.append(bool(ever_terminated.all() and not ever_truncated.any()))
                any_goal.append(bool(ever_terminated.any()))
                ended_early.append(episode_length < cfg.episode_len)
                # per-agent return summed over the episode, then averaged
                # over agents into one scalar (under joint_reward=True all
                # agents get identical reward, so this just recovers the
                # team reward; under per-agent reward it's the mean
                # individual return).
                returns.append(next_reward.sum(dim=0).mean().item())
                lengths.append(episode_length)

                # --- centralized critic under a noisy observation channel ---
                obs_td = td.select(K["obs"])
                critic_vit.noise_std = 0.0
                clean_val = value_module(obs_td.clone()).get(K["value"])
                critic_vit.noise_std = channel_noise_std
                noisy_val = value_module(obs_td.clone()).get(K["value"])
                critic_vit.noise_std = 0.0

                value_deltas.append((clean_val - noisy_val).abs().mean().item())
                clean_values.append(clean_val.mean().item())
                noisy_values.append(noisy_val.mean().item())
    finally:
        critic_vit.noise_std = original_noise_std
        policy.train(was_policy_training)
        value_module.train(was_value_training)
        env.close()

    successes_arr = np.array(successes, dtype=bool)
    any_goal_arr = np.array(any_goal, dtype=bool)
    ended_early_arr = np.array(ended_early, dtype=bool)
    returns_arr = np.array(returns, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.float32)
    return {
        "success_rate": float(successes_arr.mean()),
        "any_goal_reached_rate": float(any_goal_arr.mean()),
        "ended_early_rate": float(ended_early_arr.mean()),
        "mean_return": float(returns_arr.mean()),
        "mean_episode_length": float(lengths_arr.mean()),
        "num_episodes": int(len(successes_arr)),
        "mean_critic_value_clean": float(np.mean(clean_values)),
        "mean_critic_value_noisy": float(np.mean(noisy_values)),
        "mean_abs_critic_value_delta": float(np.mean(value_deltas)),
        "channel_noise_std": channel_noise_std,
        "successes": successes_arr,
        "returns": returns_arr,
        "lengths": lengths_arr,
    }


# --------------------------------------------------------------------------
# Training loop.
# --------------------------------------------------------------------------
def train(cfg: MAPPOConfig):
    _validate_episode_budget(cfg)  # reuses train.py's episode_len == env.max_steps guard
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    K = _keys()
    env_fn = make_torchrl_env_fn(cfg.env, fixed_seed=cfg.seed)
    env = env_fn()  # single instance to read specs from; the collector builds its own internally

    n_agents = cfg.env.num_agents
    n_actions = cfg.env.num_actions

    policy, value_module = build_networks(cfg, n_agents, n_actions, env.action_spec, device)
    env.close()  # only needed it for specs; the collector constructs its own instances via env_fn
    policy = policy.to(device)
    value_module = value_module.to(device)

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=cfg.entropy_coeff,
        critic_coeff=cfg.critic_coeff,
        normalize_advantage=cfg.normalize_advantage,
    )
    loss_module.set_keys(
        reward=K["reward"], action=K["action"],
        done=K["done"], terminated=K["terminated"], value=K["value"],
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.gae_lambda)
    GAE = loss_module.value_estimator

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    collector = SyncDataCollector(
        env_fn,
        policy,
        device=device,          # device the policy runs on during collection (rollout forward passes)
        storing_device="cpu",   # collected batch (raw uint8 images) stays on CPU, not GPU --
                                 # frames_per_batch=6000 * n_agents * 224*224*3 bytes is ~1.8GB of
                                 # image data that has no reason to sit on the GPU for the whole
                                 # collected batch when only one minibatch at a time is actually used
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.frames_per_batch, device="cpu"),  # same reasoning as above
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,
    )

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = run_dir / cfg.checkpoint_dir
    best_mean_return = float("-inf")
    best_eval_score = float("-inf")

    start = time.time()
    for it, tensordict_data in enumerate(collector):
        # tensordict_data arrives on CPU (storing_device="cpu" above). GAE's
        # critic network lives on GPU (device), so move the batch there just
        # for this one forward pass -- it's under torch.no_grad(), so no
        # backward-activation memory is retained (much cheaper than the
        # training forward+backward passes below), which is why this is
        # safe even though it briefly puts the whole frames_per_batch batch
        # on GPU. If frames_per_batch is still too large for even this
        # no_grad forward pass, shrink frames_per_batch further.
        tensordict_data = tensordict_data.to(device)
        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        data_view = tensordict_data.reshape(-1).cpu()  # back to CPU for buffer storage
        replay_buffer.extend(data_view)

        epoch_losses = []
        for _ in range(cfg.num_epochs):
            for _ in range(cfg.frames_per_batch // cfg.minibatch_size):
                subdata = replay_buffer.sample().to(device)  # only this minibatch touches GPU
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optimizer.zero_grad()
                loss_value.backward()
                grad_norm = nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optimizer.step()
                epoch_losses.append({
                    "loss_objective": loss_vals["loss_objective"].item(),
                    "loss_critic": loss_vals["loss_critic"].item(),
                    "loss_entropy": loss_vals["loss_entropy"].item(),
                    "grad_norm": grad_norm.item(),
                })

        collector.update_policy_weights_()

        # episode_reward is only meaningful on steps where an episode
        # actually finished (RewardSum resets it on done) -- mask to those.
        done = tensordict_data.get(("next", *K["done"]))
        ep_reward = tensordict_data.get(("next", *K["episode_reward"]))
        finished_returns = ep_reward[done] if done.any() else torch.zeros(0)
        mean_return = finished_returns.mean().item() if finished_returns.numel() > 0 else float("nan")

        mean_loss = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        frames_seen = (it + 1) * cfg.frames_per_batch
        wandb.log({
            "loss/objective": mean_loss["loss_objective"],
            "loss/critic": mean_loss["loss_critic"],
            "loss/entropy": mean_loss["loss_entropy"],
            "optim/grad_norm": mean_loss["grad_norm"],
            "reward/mean_return": mean_return,
            "frames": frames_seen,
        }, step=it)

        if mean_return == mean_return and mean_return > best_mean_return:  # NaN-safe check
            best_mean_return = mean_return
            save_checkpoint(ckpt_dir / "best.pt", policy, value_module, optimizer, cfg,
                             frames_seen, best_mean_return)

        if cfg.eval_every > 0 and it % cfg.eval_every == 0 and it > 0:
            eval_stats = evaluate(policy, value_module, cfg, num_episodes=cfg.eval_episodes, device=device)
            wandb.log({
                "eval/success_rate": eval_stats["success_rate"],
                "eval/any_goal_reached_rate": eval_stats["any_goal_reached_rate"],
                "eval/ended_early_rate": eval_stats["ended_early_rate"],
                "eval/mean_return": eval_stats["mean_return"],
                "eval/mean_episode_length": eval_stats["mean_episode_length"],
                "eval/mean_critic_value_clean": eval_stats["mean_critic_value_clean"],
                "eval/mean_critic_value_noisy": eval_stats["mean_critic_value_noisy"],
                "eval/mean_abs_critic_value_delta": eval_stats["mean_abs_critic_value_delta"],
            }, step=it)
            print(f"  [eval @ {it}] success_rate {eval_stats['success_rate']:.3f} | "
                  f"any_goal {eval_stats['any_goal_reached_rate']:.3f} | "
                  f"ended_early {eval_stats['ended_early_rate']:.3f} | "
                  f"mean_return {eval_stats['mean_return']:.3f} | "
                  f"critic_delta(noise={cfg.eval_channel_noise_std}) "
                  f"{eval_stats['mean_abs_critic_value_delta']:.4f} "
                  f"({eval_stats['num_episodes']} episodes)")

            eval_score = eval_stats["success_rate"] + 0.01 * eval_stats["any_goal_reached_rate"]
            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                save_checkpoint(ckpt_dir / "best_eval.pt", policy, value_module, optimizer, cfg,
                                 frames_seen, eval_stats["success_rate"])

        if cfg.checkpoint_every > 0 and it % cfg.checkpoint_every == 0 and it > 0:
            save_checkpoint(ckpt_dir / "last.pt", policy, value_module, optimizer, cfg,
                             frames_seen, mean_return)

        if it % cfg.log_every == 0:
            elapsed = time.time() - start
            print(f"iter {it:5d} | frames {frames_seen:8d} | "
                  f"objective {mean_loss['loss_objective']:.4f} | "
                  f"critic {mean_loss['loss_critic']:.4f} | "
                  f"entropy {mean_loss['loss_entropy']:.4f} | "
                  f"mean_return {mean_return:.3f} | {elapsed:.1f}s")

    # final evaluation pass, regardless of eval_every, so a completed run
    # always has a reported success rate (mirrors CommNet's train.py)
    final_eval = evaluate(policy, value_module, cfg, num_episodes=cfg.eval_episodes, device=device)
    wandb.log({
        "eval/success_rate": final_eval["success_rate"],
        "eval/any_goal_reached_rate": final_eval["any_goal_reached_rate"],
        "eval/ended_early_rate": final_eval["ended_early_rate"],
        "eval/mean_return": final_eval["mean_return"],
        "eval/mean_episode_length": final_eval["mean_episode_length"],
        "eval/mean_critic_value_clean": final_eval["mean_critic_value_clean"],
        "eval/mean_critic_value_noisy": final_eval["mean_critic_value_noisy"],
        "eval/mean_abs_critic_value_delta": final_eval["mean_abs_critic_value_delta"],
    })
    print(f"[final eval] success_rate {final_eval['success_rate']:.3f} | "
          f"any_goal {final_eval['any_goal_reached_rate']:.3f} | "
          f"ended_early {final_eval['ended_early_rate']:.3f} | "
          f"mean_return {final_eval['mean_return']:.3f} | "
          f"mean_len {final_eval['mean_episode_length']:.1f} | "
          f"critic_value(clean={final_eval['mean_critic_value_clean']:.3f}, "
          f"noisy[std={cfg.eval_channel_noise_std}]={final_eval['mean_critic_value_noisy']:.3f}) "
          f"({final_eval['num_episodes']} episodes)")

    save_checkpoint(ckpt_dir / "final.pt", policy, value_module, optimizer, cfg,
                     cfg.total_frames, mean_return)
    collector.shutdown()
    return policy, value_module


@hydra.main(version_base=None, config_path="conf", config_name="mappo_config")
def main(cfg: MAPPOConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    wandb.init(name="mappo", project=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    try:
        train(cfg)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()