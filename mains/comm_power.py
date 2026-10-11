"""The learned power control: collect trajectories, trace their channels, train and calibrate, evaluate.

    python comm_power.py                         # every stage, configs/comm/config.yaml
    python comm_power.py stage=trace shard.index=$SLURM_ARRAY_TASK_ID shard.count=$SLURM_ARRAY_TASK_COUNT
"""
import json
import math
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from c3jepa_wm.comm.channel_data import build_dataset, collect_trajectories, trace_layouts, trace_visited_pairs
from c3jepa_wm.comm.evaluation import evaluate_power_control, plot_frontiers, plot_snr, simulate_network, snr_summary, summary
from c3jepa_wm.comm.power_control import (Budget, calibrate, evaluate, fit_predictor, load_power_control,
                                          predict_gain, save_power_control)


def _dataset(cfg, data, seeds):
    link = cfg.link
    return build_dataset(data / 'layouts', data / 'trajectories', seeds, history=link.history, fading=link.fading,
                         fading_correlation=link.fading_correlation, symmetric=link.symmetric, seed=link.seed)


@hydra.main(version_base=None, config_path="../configs/comm/", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    data = Path(to_absolute_path(cfg.data_dir))
    seeds = {k: list(range(*cfg.layouts[k])) for k in ('train', 'calibration', 'test')}
    every = seeds['train'] + seeds['calibration'] + seeds['test']
    mine = every[cfg.shard.index::cfg.shard.count]
    stages = ('collect', 'trace', 'train', 'evaluate') if cfg.stage == 'all' else (cfg.stage,)
    budget = Budget(cfg.link.snr_target_db, cfg.link.noise_dbm, cfg.link.max_power_dbm)

    if 'collect' in stages:
        r = cfg.rollouts
        collect_trajectories(cfg.env, mine, data / 'trajectories', episodes=r.episodes, max_steps=r.max_steps,
                             policy=OmegaConf.to_container(r.policy), epsilon=r.epsilon, seed=r.seed, overwrite=r.overwrite)
        logger.info(f"{len(mine)} layouts' trajectories in {data / 'trajectories'}")

    if 'trace' in stages:
        t = cfg.tracing
        options = dict(frequency=t.frequency, max_depth=t.max_depth, scene=OmegaConf.to_container(t.scene),
                       **OmegaConf.to_container(t.solver))
        if t.mode == 'pairs':
            trace_visited_pairs(cfg.env, data / 'trajectories', mine, data / 'layouts', **options)
        elif t.mode == 'full':
            trace_layouts(cfg.env, mine, data / 'layouts', overwrite=t.overwrite, **options)
        else:
            raise ValueError(f"tracing.mode: pairs or full, not {t.mode!r}")
        logger.info(f"{len(mine)} layouts' channels in {data / 'layouts'}")

    if 'train' in stages:
        train, cal = _dataset(cfg, data, seeds['train']), _dataset(cfg, data, seeds['calibration'])
        logger.info(f"{len(train['gain'])} training and {len(cal['gain'])} calibration transmissions")
        p = cfg.predictor
        model = fit_predictor(train['X'], train['gain'], p.quantile, p.max_iter, p.learning_rate)
        estimate = predict_gain(model, cal['X_possible'], cal['possible_mask'])
        taus = [math.inf if math.isinf(x) else float(x) for x in cfg.calibration.taus_db]
        chosen = calibrate(estimate, cal['gain'], budget.g_min, cfg.calibration.target + cfg.calibration.slack, taus_db=taus)
        if not chosen['met']:
            logger.warning(f"no margin and threshold reach {cfg.calibration.target} on the calibration layouts: "
                           f"the most reliable is kept ({chosen['reliability']:.3f})")
        save_power_control(data / 'power_control.pkl', model, cfg.link.history, chosen, budget,
                           config=OmegaConf.to_container(cfg, resolve=True))
        logger.info(f"κ = {chosen['margin_db']:+.2f} dB, τ = {chosen['tau_db']} dB on the calibration layouts: "
                    f"{chosen['reliability']:.3f} reached, {chosen['power']:.2f}× perfect CSI's power")

    if 'evaluate' in stages:
        saved = load_power_control(data / 'power_control.pkl')
        e = cfg.evaluation
        test = _dataset(cfg, data, seeds['test'])
        c = saved['calibration']
        result = evaluate(predict_gain(saved['model'], test['X_possible'], test['possible_mask']), test['gain'],
                          saved['budget'].g_min, c['margin_db'], c['tau_db'])
        logger.info(f"on the {len(seeds['test'])} test layouts: {result['reliability']:.3f} of the servable transmissions "
                    f"reach Γ (target {cfg.calibration.target}), {result['power']:.2f}× perfect CSI's power, "
                    f"{result['withheld']:.3f} withheld; {result['servable']:.3f} servable")
        out = dict(calibration=c, test=result)
        snrs = [float(x) for x in (e.snr_targets_db or [])]
        if e.baselines or snrs:
            sets = dict(train=_dataset(cfg, data, seeds['train']), calibration=_dataset(cfg, data, seeds['calibration']), test=test)
            taus = [math.inf if math.isinf(x) else float(x) for x in cfg.calibration.taus_db]
            comparison = evaluate_power_control(saved, sets, targets=tuple(e.targets), taus_db=taus,
                                                slack=cfg.calibration.slack, quantile=cfg.predictor.quantile or 0.05,
                                                max_iter=cfg.predictor.max_iter, message_bits=e.message_bits,
                                                bits_per_token=e.bits_per_token, compare=e.baselines, snr_targets_db=snrs)
            sweep = comparison.pop('snr', None)
            out['comparison'] = comparison
            logger.info('\n' + summary(comparison))
            plot_frontiers(comparison, data / 'evaluation.png')
            if sweep:
                out['snr'] = sweep
                logger.info('\n' + snr_summary(sweep))
                plot_snr(sweep, data / 'evaluation_snr.png')
        if e.network:
            layouts = seeds['test'][:e.network_layouts] if e.network_layouts else seeds['test']
            out['network'] = simulate_network(saved, data / 'layouts', data / 'trajectories', layouts, fading=cfg.link.fading,
                                              fading_correlation=cfg.link.fading_correlation, symmetric=cfg.link.symmetric,
                                              message_bits=e.message_bits, bits_per_token=e.bits_per_token, seed=cfg.link.seed)
            n = out['network']
            logger.info(f"through a network ({len(layouts)} test layouts, {n['transmissions']} transmissions): "
                        f"{n['reliability']:.3f} reach Γ, {n['power']:.2f}× perfect CSI's power, {n['withheld']:.3f} withheld, "
                        f"{n['error_free']:.3f} of the servable messages error-free, {n['wrong_tokens']:.4f} of their tokens wrong")
        (data / 'results.json').write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
