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
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from stable_pretraining.backbone.utils import vit_hf

from utils import (
    validate_episode_budget,
    _keys,
    debug_env_structure,
    make_torchrl_env_fn,
    save_checkpoint,
)
from modules import VitEncoderModule, build_networks
from cfg import MAPPOConfig

cs = ConfigStore.instance()
cs.store(name="mappo_schema", node=MAPPOConfig)

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
    validate_episode_budget(cfg)
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

    env = make_torchrl_env_fn(cfg.env)()

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
    validate_episode_budget(cfg)  # reuses train.py's episode_len == env.max_steps guard
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    K = _keys()
    env_fn = make_torchrl_env_fn(cfg.env)
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