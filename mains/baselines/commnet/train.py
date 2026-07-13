"""
Train CommNet (ViT-Tiny encoder -> CommNet communication core -> shared
actor-critic heads) on a pixel-observation MultiGrid environment where
agents must reach a target configuration (a goal cell each).

Env pooling is delegated entirely to `MultiAgentEnvPool` (your existing
utility over N independent PettingZoo-wrapped MultiGrid envs) -- see
`MultiGridPoolAdapter` below for the thin dict<->tensor translation layer
that lets the rest of the training code work with plain (B, N, ...)
tensors.

Training algorithm matches the paper's Appendix A exactly: REINFORCE with
a learned state-specific baseline, over the *undiscounted* sum of rewards
from t to the end of the episode (no TD bootstrapping, no GAE):

    R_t = sum_{i=t}^{T} r(i)
    dtheta = sum_t [ dlog(pi(a_t|s_t)) * (R_t - b(s_t)) - alpha * d(R_t - b(s_t))^2 ]

with alpha = 0.03 (the paper's value, in all their experiments). Every
training update consumes one full episode per env (buffer length T =
episode_len), matching "after finishing an episode, we update the model
parameters theta". A GAE-based n-step actor-critic variant is also
available in `commnet/rollout.py::RolloutBuffer.compute_gae` if you want a
more sample-efficient (but non-paper) alternative.

Configuration is managed with Hydra: see `conf/config.yaml`, override
anything from the CLI (`python train.py num_envs=16 episode_len=60`), or
sweep (`python train.py -m lr=1e-4,3e-4,1e-3`).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import wandb
from rollout import RolloutBuffer

from c3jepa_wm.utils.env_utils import MultiAgentEnvPool

from cfg import TrainConfig
from module import CommNetAgent
from utils import *

cs = ConfigStore.instance()
cs.store(name="base_schema", node=TrainConfig)

# --------------------------------------------------------------------------
# Tensor-shaped adapter around MultiAgentEnvPool.
# --------------------------------------------------------------------------
class MultiGridPoolAdapter:
    """
    Thin dict<->tensor translation layer around `MultiAgentEnvPool` so the
    training loop can work with plain `(B, N, ...)` arrays/tensors, the way
    `CommNetAgent` / `RolloutBuffer` expect, while all the actual env-pool
    machinery (per-env masking, dynamic per-agent termination, noop
    substitution for finished agents) stays in `MultiAgentEnvPool` itself
    -- nothing about pooling is reimplemented here.

    Args:
        pool: a constructed `MultiAgentEnvPool`.
        obs_key: key under which each agent's stacked info dict holds its
            RGB image observation (`(B, 1, 224, 224, 3)` per the pool's
            "(batch, time, ...)" convention). Adjust if your wrapper stores
            it under a different key.
        noop_action: forwarded to `pool.step` for agents no longer live in
            a given env this step (see `MultiAgentEnvPool.step` docstring).
    """

    def __init__(self, pool: MultiAgentEnvPool, obs_key: str = "pov",
                 noop_action: int = 3):
        self.pool = pool
        self.agents = pool.agents
        self.obs_key = obs_key
        self.noop_action = noop_action

    @property
    def num_envs(self) -> int:
        return self.pool.num_envs

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def _stack_obs(self, stacked_infos: dict) -> np.ndarray:
        """{agent: {obs_key: (B, 1, 224, 224, 3)}} -> (B, N, 224, 224, 3)"""
        per_agent = [stacked_infos[a][self.obs_key][:, 0] for a in self.agents]
        return np.stack(per_agent, axis=1)

    def reset(self, seed=None) -> np.ndarray:
        _, stacked_infos = self.pool.reset(seed=seed)
        return self._stack_obs(stacked_infos)

    def step(self, actions: np.ndarray, mask: np.ndarray | None = None):
        """
        actions: (B, N) int array.
        mask: (B,) bool array; envs with mask[i] == False are skipped
              entirely by the pool (no-op for this call, previous state
              retained) -- pass "any agent still alive" here so envs where
              every agent has already finished aren't wastefully re-stepped.

        Returns:
            obs:         (B, N, 224, 224, 3) uint8
            rewards:     (B, N) float32
            done_env:    (B,) bool -- True once every agent in that env has
                         terminated or been truncated.
            agent_alive: (B, N) bool -- per-agent "still active" mask for
                         *this* step.
            term_arr:    (B, N) bool -- per-agent `terminated` this step
                         (goal reached, for FindGoalEnv).
            trunc_arr:   (B, N) bool -- per-agent `truncated` this step
                         (episode time limit hit without reaching goal).

            FindGoalEnv.gen_obs keeps every agent in the returned obs dict
            forever (terminated agents just get their last cached frame
            replayed), so `terminated`/`truncated` are freshly written by
            the pool every step for every agent (its `if a not in obs:
            continue` guard never skips them here) -- aggregating those
            dicts directly is reliable, unlike checking
            `len(env.agents) == 0` (that PettingZoo convention doesn't
            hold for an env that never actually drops agents from `obs`).
            Returning `term_arr`/`trunc_arr` separately (rather than just
            OR-ing them into `agent_alive` as before) is what lets a
            caller tell *why* an episode ended -- success (terminated) vs.
            timeout (truncated) -- which `evaluate()` needs.
        """
        action_dict = {a: actions[:, i] for i, a in enumerate(self.agents)}
        _, rewards, terminateds, truncateds, stacked_infos = self.pool.step(
            action_dict, mask=mask, noop_action=self.noop_action
        )
        rewards_arr = np.stack([rewards[a] for a in self.agents], axis=1)      # (B, N)
        term_arr = np.stack([terminateds[a] for a in self.agents], axis=1)     # (B, N)
        trunc_arr = np.stack([truncateds[a] for a in self.agents], axis=1)     # (B, N)
        agent_done = term_arr | trunc_arr
        agent_alive = ~agent_done
        done_env = agent_done.all(axis=1)  # whole episode over once every agent is done
        obs = self._stack_obs(stacked_infos)
        return obs, rewards_arr, done_env, agent_alive, term_arr, trunc_arr


# --------------------------------------------------------------------------
# Loss: paper-exact REINFORCE + baseline (Appendix A, Eq. 7).
# --------------------------------------------------------------------------
def compute_reinforce_loss(model: CommNetAgent, buf: RolloutBuffer, cfg: TrainConfig):
    """
    `buf` must hold one full (terminated or max-length-truncated) episode
    per env -- no bootstrapping is used, unlike A2C/GAE.
    """
    returns = buf.compute_mc_returns(gamma=cfg.gamma)  # R_t, (T, B, N), no grad

    T, B, N = buf.T, buf.B, buf.N
    obs_flat = buf.obs.reshape(T * B, N, *buf.obs.shape[3:])
    mask_flat = buf.alive_mask.reshape(T * B, N)  # must match the mask used during rollout collection
    logits, baseline = model(obs_flat, mask=mask_flat)  # recompute with grad, (T*B, N, A), (T*B, N)
    logits = logits.reshape(T, B, N, -1)
    baseline = baseline.reshape(T, B, N)

    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(buf.actions)  # (T, B, N)
    entropy = dist.entropy()                # (T, B, N), not in the paper -- see entropy_coef

    mask = buf.alive_mask.float()
    denom = mask.sum().clamp(min=1.0)

    advantage = returns - baseline  # (R_t - b(s_t)), paper does not normalize this

    policy_loss = -(log_probs * advantage.detach() * mask).sum() / denom
    baseline_loss = (advantage ** 2 * mask).sum() / denom
    entropy_loss = -(entropy * mask).sum() / denom  # optional, off by default (entropy_coef=0)

    total_loss = policy_loss + cfg.baseline_coef * baseline_loss + cfg.entropy_coef * entropy_loss
    stats = {
        "policy_loss": policy_loss.item(),
        "baseline_loss": baseline_loss.item(),
        "entropy": -entropy_loss.item(),
        "mean_return": returns[buf.alive_mask].mean().item() if denom > 0 else 0.0,
    }
    return total_loss, stats


# --------------------------------------------------------------------------
# Evaluation: success rate.
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: CommNetAgent, cfg: TrainConfig, num_episodes: int,
             deterministic: bool = True, device: torch.device | None = None,
             seed: int | None = None, pool: "MultiAgentEnvPool | None" = None) -> dict:
    """
    Runs episodes and computes the success rate: the fraction of episodes
    where every agent's episode ended via `terminated` (reached its goal)
    rather than `truncated` (hit `episode_len` without reaching it) --
    i.e. "success happens when agents reach the goal, and the env is done
    after that."

    Individual agents can reach their goal (terminate) at different steps
    within a shared episode -- this is tracked with a per-agent "ever
    terminated" / "ever truncated" accumulator over the whole episode
    (rather than only checking the final step), so a run is only counted
    as a success if *every* agent's own episode-end was a termination and
    *none* were truncations.

    Runs `cfg.num_envs` episodes in parallel per batch and repeats until
    `num_episodes` have been collected (the last batch is trimmed if
    `num_episodes` isn't a multiple of `cfg.num_envs`).

    Args:
        pool: reuse an existing `MultiAgentEnvPool` instead of building a
            fresh one (e.g. for periodic in-training eval where you don't
            want to keep spinning up new env processes). If None, a
            temporary pool of `cfg.num_envs` envs is built and closed
            afterward.

    Returns a dict: success_rate, mean_return, mean_episode_length,
    num_episodes, and the raw per-episode arrays (successes, returns,
    lengths) for further analysis (e.g. a histogram in a notebook).
    """
    validate_episode_budget(cfg)
    device = device or torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    was_training = model.training
    model.eval()

    owns_pool = pool is None
    if owns_pool:
        env_fns = [lambda: make_env_fn(cfg.env)() for _ in range(cfg.num_envs)]
        pool = MultiAgentEnvPool(env_fns)
    adapter = MultiGridPoolAdapter(pool, obs_key=cfg.env.obs_key, noop_action=cfg.env.noop_action)

    successes, any_goal, ended_early, returns, lengths = [], [], [], [], []
    batch_idx = 0
    try:
        while len(successes) < num_episodes:
            batch_seed = (seed + batch_idx) if seed is not None else None
            obs_np = adapter.reset(seed=batch_seed)
            obs = torch.from_numpy(obs_np).to(device)

            agent_alive = np.ones((cfg.num_envs, cfg.env.num_agents), dtype=bool)
            ever_terminated = np.zeros_like(agent_alive)
            ever_truncated = np.zeros_like(agent_alive)
            episode_return = np.zeros(cfg.num_envs, dtype=np.float32)
            episode_length = np.zeros(cfg.num_envs, dtype=np.int64)
            env_finished = np.zeros(cfg.num_envs, dtype=bool)

            for _ in range(cfg.episode_len):
                alive_mask = torch.from_numpy(agent_alive).to(device)
                actions, _, _ = model.act(obs, mask=alive_mask, deterministic=deterministic)
                actions_np = actions.cpu().numpy()

                env_mask = agent_alive.any(axis=1) & (~env_finished)
                next_obs_np, rewards_np, done_env, next_agent_alive, term_arr, trunc_arr = adapter.step(
                    actions_np, mask=env_mask
                )

                episode_return += rewards_np.sum(axis=1) * env_mask
                episode_length += env_mask.astype(np.int64)
                ever_terminated |= term_arr & env_mask[:, None]
                ever_truncated |= trunc_arr & env_mask[:, None]

                env_finished = env_finished | (done_env & env_mask)
                agent_alive = agent_alive & next_agent_alive
                obs = torch.from_numpy(next_obs_np).to(device)

                if env_finished.all():
                    break

            # Strict success: every agent's own episode-end was a
            # termination (reached goal), none were truncations (timed
            # out). This is what "success" should mean, but relies on the
            # env/wrapper never setting truncated=True alongside
            # terminated=True at genuine episode end -- some TimeLimit-
            # style wrappers do exactly that to signal "definitely over"
            # regardless of cause, which would make this always False even
            # on real successes. The two looser diagnostics below let you
            # tell that apart from "the agent genuinely never reaches the
            # goal" without touching the env:
            env_success = ever_terminated.all(axis=1) & ~ever_truncated.any(axis=1) & env_finished
            # at least one agent's terminated fired at some point, ignoring
            # truncated entirely -- if this is >0 while env_success stays
            # exactly 0, the strict criterion above is almost certainly
            # over-filtering (truncated/terminated co-occurring), not a
            # "never reaches the goal" problem.
            env_any_goal = ever_terminated.any(axis=1)
            # episode ended strictly before the outer step budget, for any
            # reason -- if this is also ~0, the agent isn't finishing
            # early at all (consistent with episode_len being too short
            # for the grid, or the agent just wandering for the full
            # budget every episode).
            env_ended_early = episode_length < cfg.episode_len

            n_new = min(cfg.num_envs, num_episodes - len(successes))
            successes.extend(env_success[:n_new].tolist())
            any_goal.extend(env_any_goal[:n_new].tolist())
            ended_early.extend(env_ended_early[:n_new].tolist())
            returns.extend(episode_return[:n_new].tolist())
            lengths.extend(episode_length[:n_new].tolist())
            batch_idx += 1
    finally:
        if owns_pool:
            pool.close()
        model.train(was_training)

    successes_arr = np.array(successes, dtype=bool)
    any_goal_arr = np.array(any_goal, dtype=bool)
    ended_early_arr = np.array(ended_early, dtype=bool)
    returns_arr = np.array(returns, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.float32)
    return {
        "success_rate": float(successes_arr.mean()),
        "any_goal_reached_rate": float(any_goal_arr.mean()),   # diagnostic, see comment above
        "ended_early_rate": float(ended_early_arr.mean()),     # diagnostic, see comment above
        "mean_return": float(returns_arr.mean()),
        "mean_episode_length": float(lengths_arr.mean()),
        "num_episodes": int(len(successes_arr)),
        "successes": successes_arr,
        "returns": returns_arr,
        "lengths": lengths_arr,
    }


# --------------------------------------------------------------------------
# Training loop.
# --------------------------------------------------------------------------
def train(cfg: TrainConfig):
    """
    Each update = one full episode (length cfg.episode_len) collected
    across cfg.num_envs environments in lockstep, then a single REINFORCE
    + baseline update over that batch of episodes (paper Appendix A).
    Environments are reset at the *start* of every update -- there is no
    cross-episode bootstrapping, and `MultiAgentEnvPool` doesn't
    auto-reset on done, so explicit reset-per-update is the correct (and
    only) way to get clean episode boundaries here.
    """
    validate_episode_budget(cfg)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    # Save under this run's Hydra output dir (conf/config.yaml sets
    # hydra.run.dir to outputs/<date>/<time>/) so checkpoints, logs, and
    # the resolved config for a given run all live together, whether or
    # not hydra.job.chdir is enabled.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = run_dir / cfg.checkpoint_dir
    best_mean_return = float("-inf")
    best_eval_score = float("-inf")

    env_fns = [lambda: make_env_fn(cfg.env)() for _ in range(cfg.num_envs)]
    pool = MultiAgentEnvPool(env_fns)
    adapter = MultiGridPoolAdapter(pool, obs_key=cfg.env.obs_key, noop_action=cfg.env.noop_action)

    model = CommNetAgent(cfg.hidden_dim, cfg.env.num_actions, cfg.num_comm_steps,
                          cfg.tie_weights, cfg.gradient_checkpoint_encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    buf = RolloutBuffer(cfg.episode_len, cfg.num_envs, cfg.env.num_agents,
                         (224, 224, 3), device)

    start = time.time()
    for update in range(cfg.total_updates):
        buf.reset()
        obs_np = adapter.reset(seed=cfg.seed + update if cfg.seed is not None else None)
        obs = torch.from_numpy(obs_np).to(device)
        # per-agent alive tracking: an individual agent can terminate (reach
        # its own goal) before the rest of its env's episode ends, so this
        # is (num_envs, num_agents), not just (num_envs,).
        agent_alive = np.ones((cfg.num_envs, cfg.env.num_agents), dtype=bool)

        for _ in range(cfg.episode_len):
            alive_mask = torch.from_numpy(agent_alive).to(device)
            # mask excludes already-terminated agents from the communication
            # average (their frozen/cached observation shouldn't influence
            # still-active teammates); noop_action is substituted for their
            # action inside adapter.step regardless.
            actions, log_probs, values = model.act(obs, mask=alive_mask)
            actions_np = actions.cpu().numpy()

            env_mask = agent_alive.any(axis=1)  # skip envs where every agent is already done
            next_obs_np, rewards_np, done_env, next_agent_alive, _term, _trunc = adapter.step(
                actions_np, mask=env_mask
            )

            buf.add(
                obs=obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                # zero out reward for agents that were already terminated
                # *before* this step (a step where an env is masked out
                # returns unchanged/zero reward already, this additionally
                # covers agents that finished earlier than their env)
                rewards=torch.from_numpy(rewards_np).to(device) * alive_mask.float(),
                done=torch.from_numpy(done_env).to(device).float(),
                alive_mask=alive_mask,
            )
            # once an agent is done it stays done for the rest of this episode
            agent_alive = agent_alive & next_agent_alive
            obs = torch.from_numpy(next_obs_np).to(device)

            if not agent_alive.any():
                break  # every agent in every env has finished; remaining buffer rows stay zero/masked-out

        loss, stats = compute_reinforce_loss(model, buf, cfg)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        wandb.log({
            "loss/total": loss.item(),
            "loss/policy": stats["policy_loss"],
            "loss/baseline": stats["baseline_loss"],
            "policy/entropy": stats["entropy"],
            "reward/mean_return": stats["mean_return"],
            "optim/grad_norm": grad_norm.item(),
            "optim/lr": optimizer.param_groups[0]["lr"],
        }, step=update)

        if cfg.save_best and stats["mean_return"] > best_mean_return:
            best_mean_return = stats["mean_return"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, cfg, update, best_mean_return)

        if cfg.eval_every > 0 and update % cfg.eval_every == 0 and update > 0:
            eval_stats = evaluate(model, cfg, num_episodes=cfg.eval_episodes,
                                   deterministic=cfg.eval_deterministic, device=device)
            wandb.log({
                "eval/success_rate": eval_stats["success_rate"],
                "eval/any_goal_reached_rate": eval_stats["any_goal_reached_rate"],
                "eval/ended_early_rate": eval_stats["ended_early_rate"],
                "eval/mean_return": eval_stats["mean_return"],
                "eval/mean_episode_length": eval_stats["mean_episode_length"],
            }, step=update)
            print(f"  [eval @ {update}] success_rate {eval_stats['success_rate']:.3f} | "
                  f"any_goal {eval_stats['any_goal_reached_rate']:.3f} | "
                  f"ended_early {eval_stats['ended_early_rate']:.3f} | "
                  f"mean_return {eval_stats['mean_return']:.3f} | "
                  f"mean_len {eval_stats['mean_episode_length']:.1f} "
                  f"({eval_stats['num_episodes']} episodes)")

            # Composite score, not raw success_rate: while success_rate is
            # genuinely 0 everywhere (as it can be for a long time on a
            # hard coordination task), comparing "0.0 > 0.0" never fires
            # again after the very first eval call (which trivially beats
            # the initial -inf) -- silently freezing best_eval.pt on a
            # near-random early snapshot forever. Blending in
            # any_goal_reached_rate as a tie-breaker means a later
            # checkpoint that's clearly more competent (even if still not
            # fully succeeding) can still displace an earlier weaker one;
            # success_rate still dominates the score once it's nonzero.
            eval_score = eval_stats["success_rate"] + 0.01 * eval_stats["any_goal_reached_rate"]
            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                save_checkpoint(ckpt_dir / "best_eval.pt", model, optimizer, cfg,
                                 update, eval_stats["success_rate"])

        if cfg.checkpoint_every > 0 and update % cfg.checkpoint_every == 0 and update > 0:
            save_checkpoint(ckpt_dir / f"update_{update}.pt", model, optimizer, cfg,
                             update, stats["mean_return"])
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, cfg,
                             update, stats["mean_return"])

        if update % cfg.log_every == 0:
            elapsed = time.time() - start
            print(f"update {update:5d} | loss {loss.item():.4f} | "
                  f"policy {stats['policy_loss']:.4f} | baseline {stats['baseline_loss']:.4f} | "
                  f"entropy {stats['entropy']:.4f} | mean_return {stats['mean_return']:.3f} | "
                  f"{elapsed:.1f}s")

    # final evaluation pass, regardless of eval_every, so a completed run
    # always has a reported success rate
    final_eval = evaluate(model, cfg, num_episodes=cfg.eval_episodes,
                           deterministic=cfg.eval_deterministic, device=device)
    wandb.log({
        "eval/success_rate": final_eval["success_rate"],
        "eval/any_goal_reached_rate": final_eval["any_goal_reached_rate"],
        "eval/ended_early_rate": final_eval["ended_early_rate"],
        "eval/mean_return": final_eval["mean_return"],
        "eval/mean_episode_length": final_eval["mean_episode_length"],
    }, step=cfg.total_updates - 1)
    print(f"[final eval] success_rate {final_eval['success_rate']:.3f} | "
          f"any_goal {final_eval['any_goal_reached_rate']:.3f} | "
          f"ended_early {final_eval['ended_early_rate']:.3f} | "
          f"mean_return {final_eval['mean_return']:.3f} | "
          f"mean_len {final_eval['mean_episode_length']:.1f} "
          f"({final_eval['num_episodes']} episodes)")

    # always leave a final checkpoint on disk regardless of checkpoint_every
    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, cfg, cfg.total_updates - 1,
                     stats["mean_return"])
    wandb.save(str(ckpt_dir / "final.pt"))  # sync the final checkpoint as a wandb artifact/file

    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        name="wm",
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    try:
        train(cfg)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()