"""The learned power control: collect trajectories, trace their channels, train and calibrate, evaluate.

    python comm_power.py                         # every stage, configs/comm/config.yaml (jobs/comm/comm_power.sh)
    python comm_power.py stage=trace shard.index=$SLURM_ARRAY_TASK_ID shard.count=$SLURM_ARRAY_TASK_COUNT

`evaluate` judges the saved power control on the test layouts (run 0), then fits, calibrates and judges again on new
splits of the same traced layouts (`monte_carlo`, runs 1..), and reports the spread over the runs. With `deployment`,
it also judges every method on deployment layouts grouped by their distance to the training layouts (in-distribution,
slightly OOD, OOD: chosen in `collect`, `<data_dir>/deployment.json`), with the margin also chosen per layout from a
pilot and adapted online (c3jepa_wm.comm.deployment). Everything is logged to wandb (`wandb`), and written to
`<data_dir>/results.json`.
"""
import json
import math
import os
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from c3jepa_wm.comm.channel_data import build_dataset, collect_trajectories, trace_layouts, trace_visited_pairs
from c3jepa_wm.comm.deployment import (deployment_layouts, deployment_summary, deployment_table, evaluate_deployment,
                                       layout_edits_of, load_deployment, plot_deployment, save_deployment)
from c3jepa_wm.comm.evaluation import (LEARNED, compare_network, evaluate_power_control, fit_baselines, layout_summary, network_summary,
                                       plot_frontiers, plot_layout_spread, plot_snr, snr_summary, summary)
from c3jepa_wm.comm.power_control import (Budget, calibrate, evaluate, fit_predictor, load_power_control,
                                          predict_gain, save_power_control)

SPLITS = ('train', 'calibration', 'test')


# --------------------------------------------------------------------------------------------- wandb
class Log:
    "wandb, if enabled: the config, scalars, tables, images and files of the run (all no-ops otherwise)."

    def __init__(self, cfg):
        self.run = None
        w = cfg.wandb
        if not w.enabled or w.mode == 'disabled':
            return
        import wandb
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / '.env')      # WANDB_API_KEY, as train.py
        except ImportError:
            pass
        self.wandb = wandb
        directory = Path(to_absolute_path(w.dir))
        directory.mkdir(parents=True, exist_ok=True)
        name = w.name or f"{cfg.stage}-{os.getenv('SLURM_JOB_ID', 'local')}"
        self.run = wandb.init(project=w.project, name=name, mode=w.mode, dir=str(directory), job_type=cfg.stage,
                              tags=list(w.tags), config=OmegaConf.to_container(cfg, resolve=True))
        self.run.define_metric('mc/run')
        self.run.define_metric('mc/*', step_metric='mc/run')          # the Monte Carlo runs: their own x axis

    def summary(self, values: dict):
        if self.run:
            self.run.summary.update(values)

    def log(self, values: dict):
        if self.run:
            self.run.log(values)

    def table(self, key: str, columns: list, rows: list):
        if self.run and rows:
            self.run.log({key: self.wandb.Table(columns=columns, data=_columns_typed(rows))})

    def images(self, paths: dict):
        if self.run:
            self.run.log({k: self.wandb.Image(str(p)) for k, p in paths.items() if Path(p).exists()})

    def files(self, name: str, paths):
        if self.run:
            artifact = self.wandb.Artifact(name, type='results')
            for p in paths:
                if Path(p).exists():
                    artifact.add_file(str(p))
            self.run.log_artifact(artifact)

    def finish(self):
        if self.run:
            self.run.finish()


def _plain(v):
    "A value wandb tables take (numpy scalars as Python ones, inf as a string)."
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    return str(v) if isinstance(v, float) and not math.isfinite(v) else v


def _columns_typed(rows: list) -> list:
    "The rows as plain values, a column of numbers and strings (e.g. τ, with inf) as strings: wandb's one type per column."
    rows = [[_plain(v) for v in r] for r in rows]
    for j in range(len(rows[0])):
        kinds = {isinstance(r[j], str) for r in rows if r[j] is not None}
        if len(kinds) > 1:
            for r in rows:
                r[j] = None if r[j] is None else str(r[j])
    return rows


# --------------------------------------------------------------------------------------------- the device
def select_device(device: str) -> str:
    "Mitsuba's variant for Sionna RT, before Sionna is imported: CUDA, or fail (no silent CPU fallback); or the CPU (LLVM)."
    import mitsuba as mi
    import drjit as dr
    variant = {'cuda': 'cuda_ad_mono_polarized', 'cpu': 'llvm_ad_mono_polarized'}[device]
    if device == 'cuda' and not dr.has_backend(dr.JitBackend.CUDA):
        raise RuntimeError("tracing.device=cuda, but Dr.Jit sees no CUDA device (a GPU node? CUDA driver loaded?); "
                           "tracing.device=cpu to trace on the CPU")
    mi.set_variant(variant)
    return mi.variant()


# --------------------------------------------------------------------------------------------- the data
def _per_layout(cfg, data, layouts, seed) -> dict:
    "Each layout's samples (`build_dataset` one layout at a time: the same as all at once, with the fading seeded per layout)."
    link = cfg.link
    return {k: build_dataset(data / 'layouts', data / 'trajectories', [k], history=link.history, fading=link.fading,
                             fading_correlation=link.fading_correlation, symmetric=link.symmetric, seed=seed)
            for k in layouts}


def _join(per_layout: dict, layouts) -> dict:
    parts = [per_layout[k] for k in layouts]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _split(cfg, run: int, pinned=()) -> dict:
    """Run 0: the split in `layouts`; run r: a new draw of the same sizes from all of them (`monte_carlo.resplit`),
    with the `pinned` layouts (the edited deployment layouts' bases) always in training."""
    seeds = {k: list(range(*cfg.layouts[k])) for k in SPLITS}
    if run == 0 or not cfg.monte_carlo.resplit:
        return seeds
    pinned = sorted(set(int(s) for s in pinned))
    pool = np.array(sorted(set().union(*seeds.values()) - set(pinned)))
    order = np.random.default_rng([cfg.monte_carlo.seed, run]).permutation(pool)
    sizes = np.cumsum([len(seeds['train']) - len(pinned), len(seeds['calibration'])])
    parts = dict(zip(SPLITS, np.split(order, sizes)))
    parts['train'] = np.r_[parts['train'], pinned]
    return {k: sorted(int(s) for s in parts[k]) for k in SPLITS}


def _deployment(cfg, data: Path, train) -> dict | None:
    "The deployment layouts (`deployment.json`): chosen against the training layouts of `layouts` the first time, then read."
    d = cfg.deployment
    if not d.enabled:
        return None
    path = data / 'deployment.json'
    if path.exists() and not d.overwrite:
        return load_deployment(path)
    groups = {name: tuple(float(x) for x in band) for name, band in d.groups.items()}
    deployment = deployment_layouts(cfg.env, train, range(*d.pool), groups, tuple(d.edited), d.per_group, d.seed, d.first_id)
    save_deployment(path, deployment)
    return load_deployment(path)


def _taus(cfg) -> list:
    return [math.inf if math.isinf(x) else float(x) for x in cfg.calibration.taus_db]


def _budget(cfg) -> Budget:
    return Budget(cfg.link.snr_target_db, cfg.link.noise_dbm, cfg.link.max_power_dbm)


# --------------------------------------------------------------------------------------------- fitting and judging
def fit_and_calibrate(cfg, sets: dict) -> tuple:
    "The predictor fit on the training layouts, and its κ and τ chosen on the calibration layouts for `calibration.target`."
    p, budget = cfg.predictor, _budget(cfg)
    model = fit_predictor(sets['train']['X'], sets['train']['gain'], p.quantile, p.max_iter, p.learning_rate)
    estimate = predict_gain(model, sets['calibration']['X_possible'], sets['calibration']['possible_mask'])
    chosen = calibrate(estimate, sets['calibration']['gain'], budget.g_min, cfg.calibration.target + cfg.calibration.slack,
                       taus_db=_taus(cfg))
    return model, chosen


def judge(cfg, saved: dict, sets: dict, snr_sweep: bool) -> tuple:
    "`evaluate_power_control` on the test layouts (every predictor, every target, per layout), and the baselines' models."
    e = cfg.evaluation
    fitted = fit_baselines(sets['train'], saved['history'], cfg.predictor.quantile or 0.05, cfg.predictor.max_iter)
    snrs = [float(x) for x in (e.snr_targets_db or [])] if snr_sweep else []
    comparison = evaluate_power_control(saved, sets, targets=tuple(e.targets), taus_db=_taus(cfg), slack=cfg.calibration.slack,
                                        quantile=cfg.predictor.quantile or 0.05, max_iter=cfg.predictor.max_iter,
                                        message_bits=e.message_bits, bits_per_token=e.bits_per_token, compare=e.baselines,
                                        snr_targets_db=snrs, fitted=fitted)
    return comparison, fitted


def network(cfg, data, saved, sets, comparison, fitted, layouts, seed) -> dict:
    "`compare_network` for each of `evaluation.network_targets`, on `layouts`."
    e = cfg.evaluation
    return {float(t): compare_network(saved, sets, data / 'layouts', data / 'trajectories', layouts, comparison=comparison,
                                      target=float(t), fitted=fitted, fading=cfg.link.fading,
                                      fading_correlation=cfg.link.fading_correlation, symmetric=cfg.link.symmetric,
                                      message_bits=e.message_bits, bits_per_token=e.bits_per_token, seed=seed)
            for t in e.network_targets}


def deploy(cfg, saved: dict, sets: dict, deployment: dict, fitted: dict) -> dict:
    "`evaluate_deployment` with the config's targets, calibration and adaptation."
    a, e = cfg.adaptation, cfg.evaluation
    return evaluate_deployment(saved, sets, deployment, fitted, targets=tuple(float(t) for t in e.targets), taus_db=_taus(cfg),
                               slack=cfg.calibration.slack, pilot=cfg.deployment.pilot, confidence=a.confidence,
                               adapt=tuple(a.predictors), min_servable=a.min_servable, steps_db=tuple(a.online.steps_db),
                               failure_scales=tuple(a.online.failure_scales), bounds_db=tuple(a.online.bounds_db),
                               quantile=cfg.predictor.quantile or 0.05, max_iter=cfg.predictor.max_iter)


DEPLOY_METRICS = ('reliability', 'power', 'withheld', 'layout_p10', 'layout_min', 'layout_below')

def deploy_metrics(results: dict) -> list:
    "Rows (method, target, '<group>/<metric>', value) of a run's deployment results."
    return [(n, t, f'{g}/{k}', o['groups'][g][k]) for n, r in results['methods'].items() for t, o in r['targets'].items()
            for g in o['groups'] for k in DEPLOY_METRICS if k in o['groups'][g]]


def deploy_mc_summary(agg: list, targets, groups) -> str:
    "A text table: per method and target, per group the mean ± std over the runs of the reliability and the power."
    get = {(n, t, k): (m, s) for n, t, k, m, s, *_ in agg}
    names = list(dict.fromkeys(n for n, *_ in agg))
    w = max(map(len, names)) + 2
    lines = []
    for t in targets:
        lines += [f"target {100 * t:g}%", f"{'':{w}s}" + ''.join(f"{g + ': reached / power':>33s}" for g in groups)]
        for n in names:
            row = f"{n:{w}s}"
            for g in groups:
                (r, rs), (p, ps) = (get.get((n, t, f'{g}/{k}'), (math.nan, math.nan)) for k in ('reliability', 'power'))
                row += f"{100 * r:14.1f}±{100 * rs:3.1f}%  {p:6.2f}±{ps:4.2f}×"
            lines.append(row)
    return '\n'.join(lines)


METRICS = ('reliability', 'power', 'withheld', 'error_free')
LAYOUT_METRICS = ('min', 'p10', 'std', 'below')

def metrics(comparison: dict) -> list:
    "Rows (predictor, target, metric, value) of a run: per target its reliability, power, withheld, error-free, κ, τ and per-layout spread; the prediction's errors."
    rows = []
    for name, r in comparison['predictors'].items():
        for k, v in r['prediction'].items():
            rows.append((name, None, k, v))
        for t, o in r['targets'].items():
            rows += [(name, t, k, o[k]) for k in (*METRICS, 'margin_db', 'tau_db')]
            rows += [(name, t, f'layout_{k}', r['layouts'][t][k]) for k in LAYOUT_METRICS if k in r['layouts'][t]]
    return rows


def aggregate(runs: list) -> list:
    "Per (predictor, target, metric): the mean, std, 95% interval half-width, min and max over the runs."
    values = {}
    for rows in runs:
        for name, t, k, v in rows:
            values.setdefault((name, t, k), []).append(float(v))
    out = []
    for (name, t, k), v in values.items():
        v = np.array(v)
        finite = v[np.isfinite(v)]
        if not len(finite):
            out.append((name, t, k, math.inf, 0.0, 0.0, math.inf, math.inf, len(v)))
            continue
        std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        out.append((name, t, k, float(finite.mean()), std, 1.96 * std / math.sqrt(len(finite)), float(finite.min()),
                    float(finite.max()), len(v)))
    return out


def mc_summary(agg: list, targets) -> str:
    "A text table: per predictor and target, the mean ± std over the runs of the reliability, the power and the worst layout."
    get = {(n, t, k): (m, s) for n, t, k, m, s, *_ in agg}
    names = list(dict.fromkeys(n for n, *_ in agg))
    w = max(map(len, names)) + 2
    head = f"{'':{w}s}" + ''.join(f"{f'{100 * t:g}%: reached / power / worst layout / layouts below':>58s}" for t in targets)
    lines = [head]
    for n in names:
        row = f"{n:{w}s}"
        for t in targets:
            (r, rs), (p, ps), (m, ms), (b, bs) = (get.get((n, t, k), (math.nan, math.nan))
                                                  for k in ('reliability', 'power', 'layout_min', 'layout_below'))
            row += f"   {100 * r:5.1f}±{100 * rs:3.1f}%  {p:5.2f}±{ps:4.2f}×  {100 * m:5.1f}±{100 * ms:3.1f}%  {100 * b:4.0f}±{100 * bs:2.0f}%"
        lines.append(row)
    return '\n'.join(lines)


# --------------------------------------------------------------------------------------------- the stages
@hydra.main(version_base=None, config_path="../configs/comm/", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    data = Path(to_absolute_path(cfg.data_dir))
    data.mkdir(parents=True, exist_ok=True)
    seeds = _split(cfg, 0)
    every = seeds['train'] + seeds['calibration'] + seeds['test']
    mine = every[cfg.shard.index::cfg.shard.count]
    stages = ('collect', 'trace', 'train', 'evaluate') if cfg.stage == 'all' else (cfg.stage,)
    deployment = _deployment(cfg, data, seeds['train']) if {'collect', 'trace', 'evaluate'} & set(stages) else None
    edits = layout_edits_of(deployment) if deployment else {}
    deployed = [k for ids in deployment['groups'].values() for k in ids] if deployment else []
    mine_deployed = deployed[cfg.shard.index::cfg.shard.count]
    if cfg.shard.count > 1 and {'train', 'evaluate'} & set(stages):
        raise ValueError("train and evaluate read every layout: run them once (shard.count=1), after the shards' collect and trace")
    budget = _budget(cfg)
    log = Log(cfg)
    log.summary({'layouts/variations': ','.join(cfg.env.get('variations') or []) or 'none',
                 **{f'layouts/{k}': len(v) for k, v in seeds.items()}})

    if 'collect' in stages:
        start, r = time.time(), cfg.rollouts
        collect_trajectories(cfg.env, mine, data / 'trajectories', episodes=r.episodes, max_steps=r.max_steps,
                             policy=OmegaConf.to_container(r.policy), epsilon=r.epsilon, seed=r.seed, overwrite=r.overwrite)
        logger.info(f"{len(mine)} layouts' trajectories in {data / 'trajectories'} ({time.time() - start:.0f} s)")
        if deployment:
            logger.info('deployment layouts (' + str(data / 'deployment.json') + '):\n' + deployment_summary(deployment))
            collect_trajectories(cfg.env, mine_deployed, data / 'trajectories', episodes=cfg.deployment.episodes,
                                 max_steps=r.max_steps, policy=OmegaConf.to_container(r.policy), epsilon=r.epsilon,
                                 seed=r.seed, overwrite=r.overwrite, edits=edits)
            logger.info(f"{len(mine_deployed)} deployment layouts' trajectories ({cfg.deployment.episodes} episodes each)")
            log.table('deploy/layouts', ['layout', 'group', 'base', 'distance', 'edits', 'walls', 'nearest'],
                      [[k, v['group'], v['base'], v['distance'], v['edits'], v['walls'], v['nearest']]
                       for k, v in deployment['layouts'].items()])
        log.summary({'collect/layouts': len(mine), 'collect/seconds': time.time() - start})

    if 'trace' in stages:
        start, t = time.time(), cfg.tracing
        variant = select_device(t.device)
        logger.info(f"Sionna RT on Mitsuba's {variant}")
        options = dict(frequency=t.frequency, max_depth=t.max_depth, scene=OmegaConf.to_container(t.scene),
                       **OmegaConf.to_container(t.solver))
        mine = mine + mine_deployed
        if t.mode == 'pairs':
            trace_visited_pairs(cfg.env, data / 'trajectories', mine, data / 'layouts', edits=edits, **options)
        elif t.mode == 'full':
            trace_layouts(cfg.env, mine, data / 'layouts', overwrite=t.overwrite, edits=edits, **options)
        else:
            raise ValueError(f"tracing.mode: pairs or full, not {t.mode!r}")
        pairs = sum(len(np.load(p)['H']) for k in mine
                    if (p := data / 'layouts' / (f'pairs_{k}.npz' if t.mode == 'pairs' else f'layout_{k}.npz')).exists())
        logger.info(f"{len(mine)} layouts' channels in {data / 'layouts'} ({pairs} traced, {time.time() - start:.0f} s)")
        log.summary({'trace/variant': variant, 'trace/layouts': len(mine), 'trace/traced': pairs,
                     'trace/seconds': time.time() - start})

    cache = {}                                                       # the samples per layout, per fading seed
    def sets_of(split: dict, seed: int) -> dict:
        if cfg.link.fading is None:
            seed = cfg.link.seed                                     # no fading: the samples don't depend on the seed
        if seed not in cache:
            cache.clear()
            cache[seed] = {}
        need = [k for v in split.values() for k in v if k not in cache[seed]]
        cache[seed].update(_per_layout(cfg, data, need, seed))
        return {k: _join(cache[seed], v) for k, v in split.items()}

    if 'train' in stages:
        start = time.time()
        sets = sets_of({k: seeds[k] for k in ('train', 'calibration')}, cfg.link.seed)
        logger.info(f"{len(sets['train']['gain'])} training and {len(sets['calibration']['gain'])} calibration transmissions")
        model, chosen = fit_and_calibrate(cfg, sets)
        if not chosen['met']:
            logger.warning(f"no margin and threshold reach {cfg.calibration.target} on the calibration layouts: "
                           f"the most reliable is kept ({chosen['reliability']:.3f})")
        save_power_control(data / 'power_control.pkl', model, cfg.link.history, chosen, budget,
                           config=OmegaConf.to_container(cfg, resolve=True))
        logger.info(f"κ = {chosen['margin_db']:+.2f} dB, τ = {chosen['tau_db']} dB on the calibration layouts: "
                    f"{chosen['reliability']:.3f} reached, {chosen['power']:.2f}× perfect CSI's power")
        log.summary({'train/transmissions': len(sets['train']['gain']), 'train/seconds': time.time() - start,
                     **{f'train/calibration_{k}': _plain(v) for k, v in chosen.items()}})

    if 'evaluate' in stages:
        e, mc = cfg.evaluation, cfg.monte_carlo
        saved = load_power_control(data / 'power_control.pkl')

        # run 0: the saved power control, on the split in `layouts`
        start = time.time()
        sets = sets_of(seeds, cfg.link.seed)
        c = saved['calibration']
        test_estimate = predict_gain(saved['model'], sets['test']['X_possible'], sets['test']['possible_mask'])
        result = evaluate(test_estimate, sets['test']['gain'], budget.g_min, c['margin_db'], c['tau_db'])
        logger.info(f"on the {len(seeds['test'])} test layouts: {result['reliability']:.3f} of the servable transmissions "
                    f"reach Γ (target {cfg.calibration.target}), {result['power']:.2f}× perfect CSI's power, "
                    f"{result['withheld']:.3f} withheld; {result['servable']:.3f} servable")
        comparison, fitted = judge(cfg, saved, sets, snr_sweep=True)
        sweep = comparison.pop('snr', None)
        out = dict(calibration=c, test=result, comparison=comparison, layouts=seeds)
        logger.info('\n' + summary(comparison))
        logger.info('\n' + layout_summary(comparison))
        plot_frontiers(comparison, data / 'evaluation.png')
        images = {'evaluate/frontiers': data / 'evaluation.png'}
        for t in e.targets:
            path = data / f'evaluation_layouts_{100 * float(t):g}.png'
            plot_layout_spread(comparison, float(t), path)
            images[f'evaluate/layouts_{100 * float(t):g}'] = path
        if sweep:
            out['snr'] = sweep
            logger.info('\n' + snr_summary(sweep))
            plot_snr(sweep, data / 'evaluation_snr.png')
            images['evaluate/snr'] = data / 'evaluation_snr.png'
        log.summary({f'evaluate/test_{k}': v for k, v in result.items()})
        log.table('evaluate/predictors', ['predictor', 'target', 'reliability', 'power', 'withheld', 'error_free',
                                          'margin_db', 'tau_db', 'met_on_calibration', 'layout_min', 'layout_p10', 'layout_below'],
                  [[n, t, o['reliability'], o['power'], o['withheld'], o['error_free'], o['margin_db'], o['tau_db'],
                    o['met_on_calibration'], *(r['layouts'][t].get(k) for k in ('min', 'p10', 'below'))]
                   for n, r in comparison['predictors'].items() for t, o in r['targets'].items()])
        log.table('evaluate/prediction', ['predictor', 'bias_db', 'spread_db', 'coverage'],
                  [[n, *(r['prediction'][k] for k in ('bias_db', 'spread_db', 'coverage'))] for n, r in comparison['predictors'].items()])
        log.table('evaluate/per_layout', ['predictor', 'target', 'layout', 'reliability', 'servable'],
                  [[n, t, k, s['per_layout'][k], s['servable'][k]] for n, r in comparison['predictors'].items()
                   for t, s in r['layouts'].items() for k in s['per_layout']])
        log.images(images)
        if e.network:                              # every method as one block of C3-WM's link (c3wm_link), next to the analysis
            layouts = seeds['test'][:e.network_layouts] if e.network_layouts else seeds['test']
            out['network'] = network(cfg, data, saved, sets, comparison, fitted, layouts, cfg.link.seed)
            for t, r in out['network'].items():
                logger.info(f"through a network, {len(layouts)} test layouts, κ and τ chosen for {t}:\n" + network_summary(r))
            log.table('evaluate/network', ['target', 'method', 'margin_db', 'tau_db', 'crc', 'source', 'reached', 'power',
                                           'withheld', 'intact', 'wrong', 'garbled', 'cost_kb_per_episode'],
                      [[t, n, m['margin_db'], m['tau_db'], m['crc'], src, m[src]['reached'], m[src]['power'], m[src]['withheld'],
                        m[src]['intact'], m[src].get('wrong'), m[src]['garbled'], m[src].get('cost')]
                       for t, r in out['network'].items() for n, m in r.items() for src in ('network', 'analysis')])
        log.summary({'evaluate/seconds': time.time() - start})

        deployed_runs = []
        if deployment:                              # the deployment layouts, per group; the margin also per layout and online
            start = time.time()
            d_sets = dict(sets_of({k: seeds[k] for k in ('train', 'calibration')}, cfg.link.seed),
                          deployment=sets_of({'deployment': deployed}, cfg.link.seed)['deployment'])
            out['deployment'] = deploy(cfg, saved, d_sets, deployment, fitted)
            deployed_runs.append(deploy_metrics(out['deployment']))
            for t in e.targets:
                logger.info('\n' + deployment_table(out['deployment'], float(t)))
                path = data / f'deployment_{100 * float(t):g}.png'
                plot_deployment(out['deployment'], float(t), path)
                images[f'deploy/groups_{100 * float(t):g}'] = path
            log.table('deploy/methods', ['method', 'target', 'group', 'reliability', 'power', 'withheld', 'layout_p10',
                                         'layout_min', 'layout_below', 'margin_db', 'tau_db', 'step_db', 'success'],
                      [[n, t, g, *(o['groups'][g].get(k) for k in DEPLOY_METRICS), o['margin_db'], o['tau_db'],
                        o.get('step_db'), o.get('success')]
                       for n, r in out['deployment']['methods'].items() for t, o in r['targets'].items() for g in o['groups']])
            log.table('deploy/prediction', ['method', 'group', 'bias_db', 'spread_db', 'coverage'],
                      [[n, g, *(p[k] for k in ('bias_db', 'spread_db', 'coverage'))]
                       for n, r in out['deployment']['methods'].items() for g, p in r['prediction'].items()])
            log.images({k: v for k, v in images.items() if k.startswith('deploy/')})
            log.summary({'deploy/seconds': time.time() - start, 'deploy/transmissions': out['deployment']['transmissions']})

        # runs 1..: fitted, calibrated and judged again on new splits of the same layouts (nothing traced again)
        runs, per_run = [metrics(comparison)], [dict(run=0, layouts=seeds, calibration=c)]
        for run in range(1, mc.runs):
            start = time.time()
            split = _split(cfg, run, [v['base'] for v in edits.values()])
            sets = sets_of(split, cfg.link.seed + run)
            model, chosen = fit_and_calibrate(cfg, sets)
            saved_r = dict(model=model, history=cfg.link.history, calibration=chosen, budget=budget)
            comparison_r, fitted_r = judge(cfg, saved_r, sets, snr_sweep=mc.snr_sweep)
            runs.append(metrics(comparison_r))
            entry = dict(run=run, layouts=split, calibration=chosen, seconds=time.time() - start)
            if mc.snr_sweep:
                entry['snr'] = comparison_r.pop('snr', None)
            if mc.network and e.network:
                layouts = split['test'][:e.network_layouts] if e.network_layouts else split['test']
                entry['network'] = network(cfg, data, saved_r, sets, comparison_r, fitted_r, layouts, cfg.link.seed + run)
            if deployment and mc.deployment:
                d_sets = sets_of({'train': split['train'], 'calibration': split['calibration'], 'deployment': deployed},
                                 cfg.link.seed + run)
                entry['deployment'] = deploy(cfg, saved_r, d_sets, deployment, fitted_r)
                deployed_runs.append(deploy_metrics(entry['deployment']))
                entry['deployment'] = {n: {t: {k: o[k] for k in ('margin_db', 'tau_db', 'step_db', 'success') if k in o}
                                           for t, o in r['targets'].items()} for n, r in entry['deployment']['methods'].items()}
            per_run.append(entry)
            learned = comparison_r['predictors'][LEARNED]['targets']
            logger.info(f"run {run} / {mc.runs - 1}: " + ', '.join(f"{100 * t:g}%: {100 * o['reliability']:.1f}% at "
                                                               f"{o['power']:.2f}×" for t, o in learned.items())
                        + f" (learned; {time.time() - start:.0f} s)")
            log.log({'mc/run': run, **{f'mc/{n}/{t}/{k}': v for n, t, k, v in runs[-1]
                                       if t is not None and k in ('reliability', 'power', 'layout_min')}})
        agg = aggregate(runs)
        out['monte_carlo'] = dict(runs=per_run, metrics=[dict(predictor=n, target=t, metric=k, mean=m, std=s, ci95=ci, min=lo,
                                                             max=hi, n=cnt) for n, t, k, m, s, ci, lo, hi, cnt in agg],
                                  per_run=[[list(r) for r in rows] for rows in runs])
        logger.info(f"over {mc.runs} runs (fit, calibration and test layouts drawn again: {mc.resplit}), mean ± std:\n"
                    + mc_summary(agg, [float(t) for t in e.targets]))
        log.table('monte_carlo/summary', ['predictor', 'target', 'metric', 'mean', 'std', 'ci95', 'min', 'max', 'runs'],
                  [list(r) for r in agg])
        log.table('monte_carlo/runs', ['run', 'predictor', 'target', 'metric', 'value'],
                  [[i, *r] for i, rows in enumerate(runs) for r in rows])
        log.summary({f'mc/{n}/{t}/{k}_{stat}': v for n, t, k, m, s, ci, *_ in agg if t is not None
                     for stat, v in (('mean', m), ('std', s), ('ci95', ci))})
        if deployed_runs:
            d_agg = aggregate(deployed_runs)
            groups = ['all', *deployment['groups']]
            out['deployment_monte_carlo'] = dict(metrics=[dict(method=n, target=t, metric=k, mean=m, std=s, ci95=ci, min=lo,
                                                               max=hi, n=cnt) for n, t, k, m, s, ci, lo, hi, cnt in d_agg])
            logger.info(f"deployment over {len(deployed_runs)} runs, mean ± std:\n"
                        + deploy_mc_summary(d_agg, [float(t) for t in e.targets], groups))
            log.table('deploy_mc/summary', ['method', 'target', 'metric', 'mean', 'std', 'ci95', 'min', 'max', 'runs'],
                      [list(r) for r in d_agg])
            log.summary({f'deploy_mc/{n}/{t}/{k}_{stat}': v for n, t, k, m, s, ci, *_ in d_agg
                         if k.split('/')[1] in ('reliability', 'power', 'layout_p10')
                         for stat, v in (('mean', m), ('std', s))})
        (data / 'results.json').write_text(json.dumps(out, indent=1, default=str))
        log.files('comm-power-results', [data / 'results.json', data / 'power_control.pkl', *images.values()])
    log.finish()


if __name__ == "__main__":
    main()
