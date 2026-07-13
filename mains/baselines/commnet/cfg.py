from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Config (Hydra structured config).
# --------------------------------------------------------------------------
@dataclass
class EnvConfig:
    num_agents: int = 2
    num_actions: int = 4      # NavigationAction: {left, right, forward, done}
    view_size: int = 7        # agent's egocentric grid window (7x7 cells)
    tile_size: int = 32       # pixels/cell -> 7*32 = 224, matching ViT-Tiny's input res
    obs_key: str = "pov"      # key holding the RGB frame in each agent's obs dict (FindGoalEnv.gen_obs)
    noop_action: int = 3      # action substituted for agents no longer live in a given env (NavigationAction.done, a genuine no-op)
    num_obstacles: int = 6
    width: int = 15
    height: int = 15
    max_steps: int = 150      # passed straight through to gym.make(...) -- the env's OWN truncation
                               # limit and _reward()'s time-decay denominator both key off this.
                               # cfg.episode_len (below) must match it, or the outer training/eval
                               # loop cuts episodes off before the env itself ever gets a chance to
                               # truncate -- see the max_steps/episode_len mismatch discussed in chat.
    joint_reward: bool = True  # on_success() gives _reward() to ALL agents once ANY agent reaches
                                # the goal, instead of only the agent that reached it. Termination
                                # stays per-agent (success_termination_mode='all' in FindGoalEnv, i.e.
                                # unaffected by this flag) -- so the coordination requirement (every
                                # agent must still individually walk onto the goal) is unchanged; only
                                # reward density changes, converting "one agent already reaches the
                                # goal in most episodes" into training signal for BOTH agents.


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)

    num_envs: int = 2         # keep low: compute_reinforce_loss recomputes the ENTIRE buffer
                               # (episode_len * num_envs * num_agents images) through ViT in ONE
                               # forward+backward pass -- at num_envs=8/episode_len=150/2 agents
                               # that's 2,400 images with full gradient tracking, which risks OOM
                               # on a 32GB GPU. num_envs=2 brings that down to 600 images/update.
    hidden_dim: int = 192
    num_comm_steps: int = 2
    tie_weights: bool = False
    gradient_checkpoint_encoder: bool = True  # trades compute for memory: recomputes backbone
                                   # activations during backward instead of storing them, while
                                   # still allowing full gradient flow -- the right lever here,
                                   # since the encoder trains from scratch (pretrained=False) and
                                   # so can never be frozen (freezing would stop it from learning
                                   # anything at all; there used to be a freeze_encoder flag here
                                   # but it was actually a dead parameter -- never wired to the
                                   # vit_hf encoder after the switch away from ViTTinyEncoder --
                                   # removed rather than left around as a misleading no-op).

    episode_len: int = 150    # T: max episode length (buffer holds one full episode/env).
                               # Must match cfg.env.max_steps -- see EnvConfig.max_steps comment.
    total_updates: int = 10_000
    gamma: float = 1.0        # paper does NOT discount within an episode
    lr: float = 3e-4
    baseline_coef: float = 0.03   # paper's alpha (Appendix A, Eq. 7)
    entropy_coef: float = 0.0     # paper does not use an entropy bonus; opt in if you want one
    max_grad_norm: float = 0.5

    device: str = "auto"      # "auto" | "cuda" | "cpu"
    seed: int | None = None
    log_every: int = 10
    project_name: str = "commnet"  # for wandb logging

    checkpoint_every: int = 200   # save a periodic checkpoint every N updates (0 disables periodic saves)
    checkpoint_dir: str = "checkpoints"  # relative to this run's Hydra output dir
    save_best: bool = True        # additionally track+overwrite a best.pt by mean_return

    eval_every: int = 200         # run evaluate() every N updates during training (0 disables)
    eval_episodes: int = 20       # episodes per evaluate() call
    eval_deterministic: bool = True  # argmax actions during eval instead of sampling
