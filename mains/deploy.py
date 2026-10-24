"""Deployment: the agents plan with the world model while their messages go through the channel
(c3jepa_wm.evaluators.deploy; configs/deploy/config.yaml).

For each method (no communication, a perfect channel, the link with perfect-CSI channel inversion, the link with the
learned power control), the same episodes of the test layouts: stable-marl's World with C3WMCommunication (each
agent's view's VQ tokens and pose, over comm-core's link for the episode's layout) and WorldModelPolicy (the world
model as each agent's planning cost). Writes <out_dir>/results.json and logs to wandb: per method, the success rate,
the episodes' length and return and, over a link, its reliability and power.

    python deploy.py checkpoints.jepa=... checkpoints.vqvae=...

With `record.dir`, one method's episodes are also kept as comm_power's trajectories (every agent's cell and heading at
every step, `trajectories_<layout>.npz`), for choosing κ, τ and λ on the planner's own movement (comm_power's `planner`):

    python deploy.py methods=['perfect channel'] layouts=[200,260] episodes=600 record.dir=.../trajectories_planner ...
"""
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import stable_marl as sm
from stable_marl.planning.solver import CategoricalCEMSolver
from stable_marl.policy import PlanConfig, WorldModelPolicy

from c3jepa_wm.comm.channel_data import PairCache, fading_correlation
from c3jepa_wm.comm.evaluation import load_lambda_policies
from c3jepa_wm.comm.power_control import load_power_control
from c3jepa_wm.evaluators.deploy import (ActionSubset, C3WMCommunication, C3WMCost, TrajectoryRecorder, c3wm_link, deploy,
                                         jepa_transform, save_trajectories, vq_tokens)
from c3jepa_wm.planners.mpc import DiscreteCEMSolver
from c3jepa_wm.utils.env_utils import LayoutWrapper

HISTORY_KEYS = ('pixels', 'msg_tokens', 'msg_received', 'msg_age', 'msg_pose')


def load_models(cfg, device):
    "The world model and the VQ-VAE, from their configs and checkpoints."
    jepa = hydra.utils.instantiate({k: v for k, v in OmegaConf.to_container(cfg.jepa, resolve=True).items()
                                    if k != 'checkpoint_path'})
    jepa.load_state_dict(torch.load(to_absolute_path(cfg.checkpoints.jepa), map_location='cpu')['wm_model_state_dict'])
    vqvae = hydra.utils.instantiate({**OmegaConf.to_container(cfg.vqvae, resolve=True)})
    vqvae.load_state_dict(torch.load(to_absolute_path(cfg.checkpoints.vqvae), map_location='cpu')['model_state_dict'])
    return jepa.to(device).eval(), vqvae.to(device).eval()


def rho(cfg) -> float:
    "The fading's ρ between steps while both agents stand still (as comm_power's)."
    link = cfg.link
    return float(link.fading_correlation) if link.fading_correlation is not None else fading_correlation(link.step_seconds,
                                                                                                         link.doppler_hz)


@hydra.main(version_base=None, config_path="../configs", config_name="deploy/config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    jepa, vqvae = load_models(cfg, device)
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    world_kwargs = {k: v for k, v in env_cfg.items() if k not in ('_target_', 'variations')}
    world_kwargs['max_steps'] = cfg.max_steps
    layouts = list(range(*cfg.layouts))
    over_link = {'learned', 'perfect CSI', 'lambda-policy'} & set(cfg.methods)
    saved = load_power_control(to_absolute_path(cfg.checkpoints.power_control)) if over_link else None
    policies = None
    if 'lambda-policy' in cfg.methods:                              # comm_power's, next to its power_control.pkl by default
        path = cfg.checkpoints.lambda_policy or Path(to_absolute_path(cfg.checkpoints.power_control)).parent / 'lambda_policy.pkl'
        policies = load_lambda_policies(to_absolute_path(str(path)))
    trace = OmegaConf.to_container(cfg.tracing, resolve=True)
    caches = {}

    def cache_for(layout):
        if layout not in caches:
            caches[layout] = PairCache(Path(cfg.data_dir) / 'layouts' / f'pairs_{layout}.npz', env_cfg, layout,
                                       trace['frequency'], trace['max_depth'], trace['scene'], **trace['solver'])
        return caches[layout]

    tokens = vq_tokens(vqvae, device)

    def make_world(method):
        silent = method == 'no communication'
        link = None if method in ('no communication', 'perfect channel') else \
            c3wm_link(method, cache_for, saved, crc=cfg.link.crc, rho=rho(cfg), rician=cfg.link.rician, policies=policies,
                      policy=cfg.link.lambda_policy.name, target=cfg.link.lambda_policy.target)
        record = [TrajectoryRecorder] if cfg.record.dir and method == cfg.record.method else []
        return sm.World(cfg.env_id, num_envs=cfg.num_envs, goal_conditioned=True, **world_kwargs,
                        pre_wrappers=[lambda e: LayoutWrapper(e, layouts, variations=tuple(env_cfg.get('variations') or ())),
                                      *record,
                                      lambda e: C3WMCommunication(e, None if silent else tokens, link,
                                                                  max_age=jepa.predictor.max_age),
                                      lambda e: ActionSubset(e, cfg.actions)])

    def make_policy(method):
        p, cost = cfg.planner, C3WMCost(jepa, cfg.pipeline.history_size)
        if p.solver == 'stable_marl':
            solver = CategoricalCEMSolver(cost, batch_size=None, num_samples=p.num_samples, n_steps=p.n_steps, topk=p.topk,
                                          mode='per_agent', device=device, seed=cfg.seed)
        elif p.solver == 'c3jepa':
            solver = DiscreteCEMSolver(cost, pop_size=p.num_samples, topk=p.topk, opt_steps=p.n_steps, mode='per_agent',
                                       device=device, seed=cfg.seed)
        else:
            raise ValueError(f"planner.solver: stable_marl or c3jepa, not {p.solver!r}")
        return WorldModelPolicy(solver, PlanConfig(horizon=p.horizon, receding_horizon=p.receding_horizon,
                                                   history_len=cfg.pipeline.history_size),
                                transform={'pixels': jepa_transform, 'goal': jepa_transform}, history_keys=HISTORY_KEYS)

    if cfg.record.dir and cfg.record.method not in cfg.methods:
        raise ValueError(f"record.method ({cfg.record.method!r}) must be one of methods {list(cfg.methods)}")
    results = deploy(make_world, make_policy, list(cfg.methods), cfg.episodes, cfg.seed,
                     saved['budget'] if saved is not None else None)
    for method, r in results.items():                               # the recorded movement, as comm_power's trajectories
        if 'trajectories' in r:
            episodes = r.pop('trajectories')
            paths = save_trajectories(episodes, to_absolute_path(cfg.record.dir))
            logger.info(f"{method}: {len(episodes)} episodes' trajectories in {len(paths)} layouts, {to_absolute_path(cfg.record.dir)}")
    for cache in caches.values():                                   # the pairs traced on the way
        if cache.unsaved:
            cache.save()

    rows = []
    for method, r in results.items():
        link = r['link'] or {}
        rows.append(dict(method=method, success_rate=r['success_rate'], mean_length=r['mean_length'],
                         mean_return=r['mean_return'], **{f'link_{k}': v for k, v in link.items()}))
        logger.info(f"{method:18s} success {r['success_rate']:5.1f}%  length {r['mean_length']:6.1f}"
                    + (f"  reached {100 * link['reached']:.1f}%  power {link['power']:.2f}×  withheld {100 * link['withheld']:.1f}%"
                       if link else ''))
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    plain = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    (out / 'results.json').write_text(json.dumps({m: {k: plain(v) for k, v in r.items()} for m, r in results.items()},
                                                 indent=1, default=float))
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"deploy-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, config=OmegaConf.to_container(cfg, resolve=True))
        columns = sorted({k for row in rows for k in row}, key=lambda k: (k != 'method', k))
        run.log({'deploy/methods': wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in rows])})
        run.finish()
    logger.info(f"results in {out / 'results.json'}")


if __name__ == "__main__":
    main()
