from dataclasses import dataclass, field

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
    seed: int = 0 # | None = None          # also fixes the env's grid/obstacle/spawn layout across the
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
    project_name: str = "mappo"
