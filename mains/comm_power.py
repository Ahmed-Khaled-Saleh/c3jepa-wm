"""The learned power control: collect trajectories, trace their channels, train and calibrate, evaluate.

    python comm_power.py                         # every stage, configs/comm/config.yaml (jobs/comm/comm_power.sh)
    python comm_power.py stage=trace shard.index=$SLURM_ARRAY_TASK_ID shard.count=$SLURM_ARRAY_TASK_COUNT

`evaluate` judges the saved power control on the test layouts (run 0), then fits, calibrates and judges again on new
splits of the same traced layouts (`monte_carlo`, runs 1..), and reports the spread over the runs. With `deployment`,
it also judges every method on deployment layouts grouped by their distance to the training layouts (in-distribution,
slightly OOD, OOD: chosen in `collect`, `<data_dir>/deployment.json`), with the margin also chosen per layout from a
pilot and adapted online (c3jepa_wm.comm.deployment). Everything is logged to wandb (`wandb`), and written to
`<data_dir>/results.json`.

The saved power control's predictor is `predictor.kind` (gradient-boosted trees, or an LSTM or a GRU over the past gains,
or with `predictor.steps` over the past steps: c3jepa_wm.comm.sequence); `evaluation.extra_predictors` are compared
with it everywhere. `link.history_data` builds the data with more past gains than `link.history` uses, so runs with
different histories are judged on the same transmissions:

    python comm_power.py predictor.kind=lstm predictor.steps=true      # train and evaluate (collect, trace: done once)
    python comm_power.py stage=train link.history=16 link.history_data=32
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

from c3jepa_wm.comm.channel_data import (build_dataset, collect_trajectories, fading_correlation, trace_layouts, trace_visited_pairs,
                                         with_history)
from c3jepa_wm.comm.deployment import (deployment_layouts, deployment_summary, deployment_table, evaluate_deployment,
                                       layout_edits_of, load_deployment, plot_deployment, save_deployment)
from c3jepa_wm.comm.evaluation import (LEARNED, OPTIMAL, motion_summary, compare_network, evaluate_power_control, fit_baselines, guarantee_summary,
                                       layout_summary, network_summary, plot_frontiers, plot_layout_spread, plot_snr, snr_summary,
                                       summary)
from c3jepa_wm.comm.power_control import Budget, calibrate, evaluate, load_power_control, predict_on, save_power_control
from c3jepa_wm.comm.evaluation import save_lambda_policies
from c3jepa_wm.comm.continual import (METHOD_COLUMNS, continual_rows, continual_summary, evaluate_continual, plot_continual,
                                      task_columns)
from c3jepa_wm.comm.sequence import fit_model, predictor_name

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
def _rho(cfg) -> float:
    "The small-scale fading's ρ between steps while both agents stand still: `link.fading_correlation`, or from the Doppler."
    link = cfg.link
    if link.get('fading_correlation') is not None:
        return float(link.fading_correlation)
    return fading_correlation(link.step_seconds, link.doppler_hz)


def _history_data(cfg) -> int:
    "The past gains the data is built with: `link.history_data` (at least `link.history`), or `link.history`."
    link = cfg.link
    built = int(link.history_data or link.history)
    if built < link.history:
        raise ValueError(f"link.history_data ({built}) is fewer past gains than link.history ({link.history}) uses")
    return built


def _per_layout(cfg, data, layouts, seed, trajectories: str = 'trajectories') -> dict:
    """Each layout's samples (`build_dataset` one layout at a time: the same as all at once, with the fading seeded per
    layout), built with `_history_data` past gains and kept to the `link.history` most recent; with the past steps if
    a predictor reads them; from the trajectories in `data / trajectories`."""
    link = cfg.link
    return {k: with_history(build_dataset(data / 'layouts', data / trajectories, [k], history=_history_data(cfg),
                                          fading=link.fading, fading_correlation=_rho(cfg), symmetric=link.symmetric,
                                          seed=seed, steps=_needs_steps(cfg), channel=link.channel), link.history)
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


def _shifts(cfg) -> dict:
    "`deployment.shifts`: name -> (the behaviour policy's weights, its ε), each collected into `trajectories_<name>/`."
    d = cfg.deployment
    if not d.enabled or not d.get('shifts'):
        return {}
    return {name: (OmegaConf.to_container(s.policy), float(s.get('epsilon', cfg.rollouts.epsilon))) for name, s in d.shifts.items()}


def _taus(cfg) -> list:
    return [math.inf if math.isinf(x) else float(x) for x in cfg.calibration.taus_db]


def _budget(cfg) -> Budget:
    return Budget(cfg.link.snr_target_db, cfg.link.noise_dbm, cfg.link.max_power_dbm)


# --------------------------------------------------------------------------------------------- fitting and judging
def _options(cfg) -> dict:
    "`fit_model`'s options for each kind: the trees' and the sequence predictors'."
    p = cfg.predictor
    return dict(trees=dict(max_iter=p.max_iter, learning_rate=p.learning_rate),
                sequence=OmegaConf.to_container(p.sequence, resolve=True))


def _saved_spec(cfg) -> dict:
    "The saved power control's predictor, as `fit_model`'s arguments."
    p = cfg.predictor
    return dict(kind=p.kind, quantile=p.quantile, steps=bool(p.steps), **_options(cfg))


def _extra(cfg) -> list:
    """`evaluation.extra_predictors` as `fit_model`'s arguments (`mean`: for the mean, else `predictor.quantile`), with
    the trees' quantile model when the saved predictor is not it; the saved predictor itself left out."""
    p, saved = cfg.predictor, _saved_spec(cfg)
    entries = [dict(kind=e.kind, steps=bool(e.get('steps', False)), quantile=None if e.get('mean', False) else p.quantile)
               for e in cfg.evaluation.extra_predictors or []]
    if p.kind != 'trees':
        entries.insert(0, dict(kind='trees', steps=False, quantile=p.quantile))
    same = lambda e: (e['kind'], e['steps'], e['quantile']) == (saved['kind'], saved['steps'], saved['quantile'])
    names = {}
    for e in entries:
        if not same(e):
            names.setdefault(predictor_name(e['kind'], e['quantile'], e['steps']), dict(e, **_options(cfg)))
    return list(names.values())


def _predictor_label(cfg) -> str:
    "The saved predictor's name (`predictor_name`), for the logs: in the tables it is `LEARNED`."
    p = cfg.predictor
    return predictor_name(p.kind, p.quantile, bool(p.steps))


def _check_saved(cfg, saved: dict):
    "The saved power control must be the configured one (its predictor and history): evaluate after training with the same config."
    was = (saved.get('info', {}).get('config') or {})
    p, link = was.get('predictor', {}), was.get('link', {})
    then = (p.get('kind', 'trees'), bool(p.get('steps', False)), p.get('quantile'), link.get('history'),
            link.get('history_data') or link.get('history'))
    now = (cfg.predictor.kind, bool(cfg.predictor.steps), cfg.predictor.quantile, cfg.link.history, _history_data(cfg))
    if was and then != now:
        raise ValueError(f"the saved power control was trained with (kind, steps, quantile, history, history_data) = {then}, "
                         f"the config asks for {now}: run stage=train with this config first (or stage=all)")


def _needs_steps(cfg) -> bool:
    "Whether a predictor reads the past steps (then the data has them, `S`)."
    return (bool(cfg.predictor.steps) or any(e['steps'] for e in _extra(cfg))
            or bool(cfg.continual.enabled and cfg.continual.model.get('steps', False)))


def _continual_spec(cfg) -> dict:
    "The predictor learning over the stream (`continual.model`), as `fit_model`'s arguments."
    m = cfg.continual.model
    return dict(kind=m.kind, steps=bool(m.get('steps', False)), quantile=None if m.get('mean', False) else cfg.predictor.quantile,
                sequence=_options(cfg)['sequence'])


def continual(cfg, sets: dict, stream: dict, groups: dict, budget: Budget) -> dict:
    "`evaluate_continual` with the config's stream, methods and learning (`continual.*`), targets and calibration."
    c, e = cfg.continual, cfg.evaluation
    present = set(np.unique(stream['layout']).tolist())
    groups = {g: [k for k in ids if k in present] for g, ids in groups.items()}
    return evaluate_continual(sets, stream, {g: ids for g, ids in groups.items() if ids}, budget.g_min, model=_continual_spec(cfg),
                              methods=tuple(c.methods), trees=dict(trees=_options(cfg)['trees']) if c.trees else None,
                              trees_every=c.trees_every, order=c.order, runs=c.runs, update_episodes=c.update_episodes,
                              targets=tuple(float(t) for t in e.targets), taus_db=_taus(cfg), slack=cfg.calibration.slack,
                              recalibrate_every=c.recalibrate_every, seed=c.seed,
                              **OmegaConf.to_container(c.learning, resolve=True))


def fit_and_calibrate(cfg, sets: dict) -> tuple:
    "The predictor fit on the training layouts, and its κ and τ chosen on the calibration layouts for `calibration.target`."
    budget = _budget(cfg)
    model = fit_model(sets['train'], **_saved_spec(cfg))
    estimate = predict_on(model, sets['calibration'])
    chosen = calibrate(estimate, sets['calibration']['gain'], budget.g_min, cfg.calibration.target + cfg.calibration.slack,
                       taus_db=_taus(cfg))
    return model, chosen


def judge(cfg, saved: dict, sets: dict, snr_sweep: bool) -> tuple:
    "`evaluate_power_control` on the test layouts (every predictor, every target, per layout), and the baselines' models."
    e = cfg.evaluation
    fitted = fit_baselines(sets['train'], saved['history'], cfg.predictor.quantile or 0.05, cfg.predictor.max_iter,
                           extra=_extra(cfg))
    snrs = [float(x) for x in (e.snr_targets_db or [])] if snr_sweep else []
    comparison = evaluate_power_control(saved, sets, targets=tuple(e.targets), taus_db=_taus(cfg), slack=cfg.calibration.slack,
                                        quantile=cfg.predictor.quantile or 0.05, max_iter=cfg.predictor.max_iter,
                                        message_bits=e.message_bits, bits_per_token=e.bits_per_token, compare=e.baselines,
                                        snr_targets_db=snrs, fitted=fitted, **_over_layouts(cfg))
    return comparison, fitted


def _over_layouts(cfg) -> dict:
    "`calibration.layouts` as `evaluate_power_control`'s arguments (none if `coverage` is null)."
    c = cfg.calibration.layouts
    return dict(layout_coverage=c.coverage, layout_confidence=c.confidence, min_servable=c.min_servable) if c.coverage else {}


def network(cfg, data, saved, sets, comparison, fitted, layouts, seed) -> dict:
    "`compare_network` for each of `evaluation.network_targets`, on `layouts`."
    e = cfg.evaluation
    return {float(t): compare_network(saved, sets, data / 'layouts', data / 'trajectories', layouts, comparison=comparison,
                                      target=float(t), fitted=fitted, fading=cfg.link.fading, channel=cfg.link.channel,
                                      fading_correlation=_rho(cfg), symmetric=cfg.link.symmetric,
                                      message_bits=e.message_bits, bits_per_token=e.bits_per_token, seed=seed,
                                      skip=_history_data(cfg))
            for t in e.network_targets}


def deploy(cfg, saved: dict, sets: dict, deployment: dict, fitted: dict) -> dict:
    "`evaluate_deployment` with the config's targets, calibration and adaptation."
    a, e = cfg.adaptation, cfg.evaluation
    return evaluate_deployment(saved, sets, deployment, fitted, targets=tuple(float(t) for t in e.targets), taus_db=_taus(cfg),
                               slack=cfg.calibration.slack, pilot=cfg.deployment.pilot, confidence=a.confidence,
                               adapt=tuple(a.predictors), min_servable=a.min_servable, steps_db=tuple(a.online.steps_db),
                               failure_scales=tuple(a.online.failure_scales), bounds_db=tuple(a.online.bounds_db),
                               quantile=cfg.predictor.quantile or 0.05, max_iter=cfg.predictor.max_iter,
                               layout_coverage=cfg.calibration.layouts.coverage or None,
                               layout_confidence=cfg.calibration.layouts.confidence)


DEPLOY_METRICS = ('reliability', 'power', 'withheld', 'layout_p10', 'layout_min', 'layout_below', 'layout_significantly_below',
                  'layout_between_std')

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
LAYOUT_METRICS = ('min', 'p10', 'std', 'below', 'significantly_below', 'between_std')

def metrics(comparison: dict) -> list:
    "Rows (predictor, target, metric, value) of a run: per target its reliability, power, withheld, error-free, κ, τ and per-layout spread; the prediction's errors."
    rows = []
    for name, r in comparison['predictors'].items():
        for k, v in r['prediction'].items():
            rows.append((name, None, k, v))
        for t, o in r['targets'].items():
            rows += [(name, t, k, o[k]) for k in (*METRICS, 'margin_db', 'tau_db')]
            rows += [(name, t, f'layout_{k}', r['layouts'][t][k]) for k in LAYOUT_METRICS if k in r['layouts'][t]]
        for t, o in r.get('layout_targets', {}).items():               # κ and τ chosen over layouts
            rows += [(name, t, f'over_layouts/{k}', o[k]) for k in ('reliability', 'power', 'withheld', 'margin_db', 'tau_db')]
            rows += [(name, t, 'over_layouts/met_on_calibration', float(o['met_on_calibration']))]
            rows += [(name, t, f'over_layouts/layout_{k}', o['layouts'][k]) for k in LAYOUT_METRICS if k in o['layouts']]
        for part, m in r.get('motion', {}).items():                       # after a move, and while both stood still
            rows += [(name, None, f'motion/{part}/{k}', m['prediction'][k]) for k in ('bias_db', 'spread_db')]
            rows += [(name, t, f'motion/{part}/{k}', o[k]) for t, o in m['targets'].items() for k in ('reliability', 'power')]
    for name, by_target in _policies(comparison).items():                # Proposition 1's policy: the oracle's, the predicted ones
        for t, o in by_target.items():
            rows += [(name, t, k, o[k]) for k in ('reliability', 'power', 'withheld')]
            rows += [(name, t, f'layout_{k}', o[f'layout_{k}']) for k in LAYOUT_METRICS if f'layout_{k}' in o]
            rows += [(name, t, f'motion/{part}/{k}', m[k]) for part, m in o['motion'].items() for k in ('reliability', 'power')]
    return rows


def _policies(comparison: dict) -> dict:
    "name -> per target the λ-policy's results: the optimal policy (oracle), and each distributional predictor's."
    return {**({OPTIMAL: comparison['optimal']} if 'optimal' in comparison else {}), **comparison.get('distributional', {})}


def guarantee_mc_summary(agg: list, targets) -> str:
    """A text table over the runs: per predictor and target, κ and τ chosen pooled and over layouts: the reliability
    (pooled), the power, and the share of test layouts reaching the target (its mean over runs: the guarantee's check)."""
    get = {(n, t, k): (m, s) for n, t, k, m, s, *_ in agg}
    names = [n for n in dict.fromkeys(n for n, *_ in agg) if (n, targets[0], 'over_layouts/power') in get]
    if not names:
        return ''
    w = max(map(len, names)) + 2
    lines = [f"{'':{w}s}" + ''.join(f"{f'{100 * t:g}%  pooled':>34s}{'over layouts':>34s}" for t in targets)]
    for n in names:
        row = f"{n:{w}s}"
        for t in targets:
            for pre in ('', 'over_layouts/'):
                (r, rs), (p, ps), (b, bs) = (get.get((n, t, pre + k), (math.nan, math.nan))
                                             for k in ('reliability', 'power', 'layout_below'))
                row += f"{100 * r:12.1f}±{100 * rs:3.1f}% {p:6.2f}±{ps:4.2f}× {100 * (1 - b):5.0f}%"
        lines.append(row)
    lines.append('per target and choice: reached (pooled), power, and the share of test layouts reaching the target (mean over runs)')
    return '\n'.join(lines)


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
    outputs = Path(to_absolute_path(cfg.out_dir))                  # the power control, results.json, the plots
    outputs.mkdir(parents=True, exist_ok=True)
    seeds = _split(cfg, 0)
    every = seeds['train'] + seeds['calibration'] + seeds['test']
    mine = every[cfg.shard.index::cfg.shard.count]
    stages = (('collect', 'trace', 'train', 'evaluate') + (('continual',) if cfg.continual.enabled else ())
              if cfg.stage == 'all' else (cfg.stage,))
    deployment = _deployment(cfg, data, seeds['train']) if {'collect', 'trace', 'evaluate', 'continual'} & set(stages) else None
    edits = layout_edits_of(deployment) if deployment else {}
    deployed = [k for ids in deployment['groups'].values() for k in ids] if deployment else []
    mine_deployed = deployed[cfg.shard.index::cfg.shard.count]
    if cfg.shard.count > 1 and {'train', 'evaluate', 'continual'} & set(stages):
        raise ValueError("train, evaluate and continual read every layout: run them once (shard.count=1), after the shards' collect and trace")
    budget = _budget(cfg)
    log = Log(cfg)
    log.summary({'layouts/variations': ','.join(cfg.env.get('variations') or []) or 'none',
                 **{f'layouts/{k}': len(v) for k, v in seeds.items()}})

    if 'collect' in stages:
        start, r = time.time(), cfg.rollouts
        collect_trajectories(cfg.env, mine, data / 'trajectories', episodes=max(r.episodes, r.eval_episodes or 0), max_steps=r.max_steps,
                             policy=OmegaConf.to_container(r.policy), epsilon=r.epsilon, seed=r.seed, overwrite=r.overwrite)
        logger.info(f"{len(mine)} layouts' trajectories in {data / 'trajectories'} ({time.time() - start:.0f} s)")
        if deployment:
            logger.info('deployment layouts (' + str(data / 'deployment.json') + '):\n' + deployment_summary(deployment))
            collect_trajectories(cfg.env, mine_deployed, data / 'trajectories', episodes=cfg.deployment.episodes,
                                 max_steps=r.max_steps, policy=OmegaConf.to_container(r.policy), epsilon=r.epsilon,
                                 seed=r.seed, overwrite=r.overwrite, edits=edits)
            logger.info(f"{len(mine_deployed)} deployment layouts' trajectories ({cfg.deployment.episodes} episodes each)")
            for name, (policy, epsilon) in _shifts(cfg).items():
                collect_trajectories(cfg.env, mine_deployed, data / f'trajectories_{name}', episodes=cfg.deployment.shift_episodes,
                                     max_steps=r.max_steps, policy=policy, epsilon=epsilon, seed=r.seed, overwrite=r.overwrite,
                                     edits=edits)
                logger.info(f"{len(mine_deployed)} deployment layouts' trajectories with the {name} behaviour {policy}, ε = {epsilon}")
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
            for name in _shifts(cfg):                                 # the deployment layouts' pairs the other behaviours visit
                trace_visited_pairs(cfg.env, data / f'trajectories_{name}', mine_deployed, data / 'layouts', edits=edits, **options)
        elif t.mode == 'full':
            trace_layouts(cfg.env, mine, data / 'layouts', overwrite=t.overwrite, edits=edits, **options)
        else:
            raise ValueError(f"tracing.mode: pairs or full, not {t.mode!r}")
        pairs = sum(len(np.load(p)['H']) for k in mine
                    if (p := data / 'layouts' / (f'pairs_{k}.npz' if t.mode == 'pairs' else f'layout_{k}.npz')).exists())
        logger.info(f"{len(mine)} layouts' channels in {data / 'layouts'} ({pairs} traced, {time.time() - start:.0f} s)")
        log.summary({'trace/variant': variant, 'trace/layouts': len(mine), 'trace/traced': pairs,
                     'trace/seconds': time.time() - start})

    cache = {}                                                       # the samples per layout, per fading seed and trajectories
    def sets_of(split: dict, seed: int, trajectories: str = 'trajectories') -> dict:
        if cfg.link.fading is None:
            seed = cfg.link.seed                                     # no fading: the samples don't depend on the seed
        key = (seed, trajectories)
        if key not in cache:
            for k in [k for k in cache if k[0] != seed]:
                del cache[k]
            cache[key] = {}
        need = [k for v in split.values() for k in v if k not in cache[key]]
        cache[key].update(_per_layout(cfg, data, need, seed, trajectories))
        out = {k: _join(cache[key], v) for k, v in split.items()}
        if 'train' in out:                                           # training: each layout's first rollouts.episodes
            keep = out['train']['episode'] < cfg.rollouts.episodes
            out['train'] = {k: v[keep] for k, v in out['train'].items()}
        return out

    if 'train' in stages:
        start = time.time()
        sets = sets_of({k: seeds[k] for k in ('train', 'calibration')}, cfg.link.seed)
        logger.info(f"{len(sets['train']['gain'])} training and {len(sets['calibration']['gain'])} calibration transmissions")
        model, chosen = fit_and_calibrate(cfg, sets)
        if not chosen['met']:
            logger.warning(f"no margin and threshold reach {cfg.calibration.target} on the calibration layouts: "
                           f"the most reliable is kept ({chosen['reliability']:.3f})")
        save_power_control(outputs / 'power_control.pkl', model, cfg.link.history, chosen, budget,
                           config=OmegaConf.to_container(cfg, resolve=True))
        logger.info(f"κ = {chosen['margin_db']:+.2f} dB, τ = {chosen['tau_db']} dB on the calibration layouts: "
                    f"{chosen['reliability']:.3f} reached, {chosen['power']:.2f}× perfect CSI's power")
        log.summary({'train/predictor': _predictor_label(cfg), 'train/history_data': _history_data(cfg),
                     'train/transmissions': len(sets['train']['gain']), 'train/seconds': time.time() - start,
                     **{f'train/calibration_{k}': _plain(v) for k, v in chosen.items()}})

    if 'evaluate' in stages:
        e, mc = cfg.evaluation, cfg.monte_carlo
        saved = load_power_control(outputs / 'power_control.pkl')
        _check_saved(cfg, saved)
        logger.info(f"the saved power control's predictor ({LEARNED} in the tables): {_predictor_label(cfg)}, "
                    f"history {cfg.link.history} (data built with {_history_data(cfg)})")
        log.summary({'evaluate/predictor': _predictor_label(cfg)})

        # run 0: the saved power control, on the split in `layouts`
        start = time.time()
        sets = sets_of(seeds, cfg.link.seed)
        c = saved['calibration']
        test_estimate = predict_on(saved['model'], sets['test'])
        result = evaluate(test_estimate, sets['test']['gain'], budget.g_min, c['margin_db'], c['tau_db'])
        logger.info(f"on the {len(seeds['test'])} test layouts: {result['reliability']:.3f} of the servable transmissions "
                    f"reach Γ (target {cfg.calibration.target}), {result['power']:.2f}× perfect CSI's power, "
                    f"{result['withheld']:.3f} withheld; {result['servable']:.3f} servable")
        comparison, fitted = judge(cfg, saved, sets, snr_sweep=True)
        if comparison.get('distributional'):                 # Proposition 1's policies, for deployment (deploy.py: lambda-policy)
            save_lambda_policies(outputs / 'lambda_policy.pkl', fitted, comparison, saved)
            logger.info(f"the λ-policies saved: {outputs / 'lambda_policy.pkl'} ({', '.join(comparison['distributional'])})")
        sweep = comparison.pop('snr', None)
        out = dict(calibration=c, test=result, comparison=comparison, layouts=seeds)
        logger.info('\n' + summary(comparison))
        logger.info('\n' + layout_summary(comparison))
        for t in e.targets:
            if text := motion_summary(comparison, float(t)):
                logger.info('\n' + text)
        motion = [[n, part, m['share'], m['prediction']['bias_db'], m['prediction']['spread_db'], t, o['reliability'], o['power'],
                   o['withheld']] for n, r in comparison['predictors'].items() for part, m in r.get('motion', {}).items()
                  for t, o in m['targets'].items()]
        motion += [[n, part, m['share'], None, None, t, m['reliability'], m['power'], m['withheld']]
                   for n, by_target in _policies(comparison).items() for t, o in by_target.items() for part, m in o['motion'].items()]
        if motion:
            log.table('evaluate/motion', ['predictor', 'part', 'share', 'bias_db', 'spread_db', 'target', 'reliability', 'power',
                                          'withheld'], motion)
        if comparison.get('optimal'):
            log.table('evaluate/optimal', ['target', 'reliability', 'power', 'withheld', 'lambda'],
                      [[t, o['reliability'], o['power'], o['withheld'], o['lam']] for t, o in comparison['optimal'].items()])
        if comparison.get('distributional'):
            log.table('evaluate/distributional', ['method', 'target', 'reliability', 'power', 'withheld', 'lambda', 'layout_min',
                                                  'layout_p10', 'layout_below'],
                      [[n, t, o['reliability'], o['power'], o['withheld'], o['lam'], *(o.get(f'layout_{k}') for k in ('min', 'p10', 'below'))]
                       for n, by_target in comparison['distributional'].items() for t, o in by_target.items()])
        if comparison.get('layout_coverage'):
            logger.info('\n' + guarantee_summary(comparison))
            log.table('evaluate/over_layouts', ['predictor', 'target', 'reliability', 'power', 'withheld', 'margin_db', 'tau_db',
                                                'met_on_calibration', 'calibration_below', 'allowed', 'test_layouts_reaching',
                                                'significantly_below', 'between_std'],
                      [[n, t, o['reliability'], o['power'], o['withheld'], o['margin_db'], o['tau_db'], o['met_on_calibration'],
                        o['calibration_below'], o['allowed'], 1 - o['layouts'].get('below', math.nan),
                        o['layouts'].get('significantly_below'), o['layouts'].get('between_std')]
                       for n, r in comparison['predictors'].items() for t, o in r.get('layout_targets', {}).items()])
        plot_frontiers(comparison, outputs / 'evaluation.png')
        images = {'evaluate/frontiers': outputs / 'evaluation.png'}
        for t in e.targets:
            path = outputs / f'evaluation_layouts_{100 * float(t):g}.png'
            plot_layout_spread(comparison, float(t), path)
            images[f'evaluate/layouts_{100 * float(t):g}'] = path
        if sweep:
            out['snr'] = sweep
            logger.info('\n' + snr_summary(sweep))
            plot_snr(sweep, outputs / 'evaluation_snr.png')
            images['evaluate/snr'] = outputs / 'evaluation_snr.png'
        log.summary({f'evaluate/test_{k}': v for k, v in result.items()})
        log.table('evaluate/predictors', ['predictor', 'target', 'reliability', 'power', 'withheld', 'error_free',
                                          'margin_db', 'tau_db', 'met_on_calibration', 'layout_min', 'layout_p10', 'layout_below'],
                  [[n, t, o['reliability'], o['power'], o['withheld'], o['error_free'], o['margin_db'], o['tau_db'],
                    o['met_on_calibration'], *(r['layouts'][t].get(k) for k in ('min', 'p10', 'below'))]
                   for n, r in comparison['predictors'].items() for t, o in r['targets'].items()])
        log.table('evaluate/prediction', ['predictor', 'bias_db', 'spread_db', 'coverage'],
                  [[n, *(r['prediction'][k] for k in ('bias_db', 'spread_db', 'coverage'))] for n, r in comparison['predictors'].items()])
        log.table('evaluate/per_layout', ['predictor', 'target', 'layout', 'reliability', 'servable', 'low', 'high'],
                  [[n, t, k, s['per_layout'][k], s['servable'][k], *s.get('interval', {}).get(k, (None, None))]
                   for n, r in comparison['predictors'].items() for t, s in r['layouts'].items() for k in s['per_layout']])
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
                path = outputs / f'deployment_{100 * float(t):g}.png'
                plot_deployment(out['deployment'], float(t), path)
                images[f'deploy/groups_{100 * float(t):g}'] = path
            log.table('deploy/methods', ['method', 'target', 'group', *DEPLOY_METRICS, 'margin_db', 'tau_db', 'step_db', 'success'],
                      [[n, t, g, *(o['groups'][g].get(k) for k in DEPLOY_METRICS), o['margin_db'], o['tau_db'],
                        o.get('step_db'), o.get('success')]
                       for n, r in out['deployment']['methods'].items() for t, o in r['targets'].items() for g in o['groups']])
            log.table('deploy/prediction', ['method', 'group', 'bias_db', 'spread_db', 'coverage'],
                      [[n, g, *(p[k] for k in ('bias_db', 'spread_db', 'coverage'))]
                       for n, r in out['deployment']['methods'].items() for g, p in r['prediction'].items()])
            log.images({k: v for k, v in images.items() if k.startswith('deploy/')})
            log.summary({'deploy/seconds': time.time() - start, 'deploy/transmissions': out['deployment']['transmissions']})
            shift_rows = []                                          # the same calibration, the agents moving differently
            for name, (policy, epsilon) in _shifts(cfg).items():
                s_sets = dict(d_sets, deployment=sets_of({'deployment': deployed}, cfg.link.seed, f'trajectories_{name}')['deployment'])
                r = deploy(cfg, saved, s_sets, deployment, fitted)
                out.setdefault('deployment_shift', {})[name] = r
                moving = float(s_sets['deployment']['fresh'].mean()) if 'fresh' in s_sets['deployment'] else math.nan
                for t in e.targets:
                    logger.info(f"deployment with the {name} behaviour ({100 * moving:.0f}% of transmissions after a move):\n"
                                + deployment_table(r, float(t)))
                shift_rows += [[name, moving, n, t, g, *(o['groups'][g].get(k) for k in DEPLOY_METRICS)]
                               for n, m in r['methods'].items() for t, o in m['targets'].items() for g in o['groups']]
            if shift_rows:
                moving = float(d_sets['deployment']['fresh'].mean()) if 'fresh' in d_sets['deployment'] else math.nan
                shift_rows += [['training mix', moving, n, t, g, *(o['groups'][g].get(k) for k in DEPLOY_METRICS)]
                               for n, m in out['deployment']['methods'].items() for t, o in m['targets'].items() for g in o['groups']]
                log.table('deploy_shift/methods', ['behaviour', 'after_a_move', 'method', 'target', 'group', *DEPLOY_METRICS],
                          shift_rows)

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
        if cfg.calibration.layouts.coverage:
            logger.info(f"κ and τ over layouts (for {cfg.calibration.layouts.coverage} of them), over {mc.runs} runs:\n"
                        + guarantee_mc_summary(agg, [float(t) for t in e.targets]))
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
        (outputs / 'results.json').write_text(json.dumps(out, indent=1, default=str))
        log.files('comm-power-results', [outputs / 'results.json', outputs / 'power_control.pkl',
                                         *[outputs / 'lambda_policy.pkl'] * (outputs / 'lambda_policy.pkl').exists(), *images.values()])

    if 'continual' in stages:                       # the predictor learning over a stream of layouts (c3jepa_wm.comm.continual)
        start, c = time.time(), cfg.continual
        sets = sets_of(seeds, cfg.link.seed)
        if c.layouts == 'deployment':
            if not deployment:
                raise ValueError("continual.layouts=deployment needs the deployment layouts (deployment.enabled)")
            groups = {g: list(ids) for g, ids in deployment['groups'].items()}
            stream = sets_of({'deployment': deployed}, cfg.link.seed)['deployment']
        elif c.layouts == 'test':
            groups, stream = {'test': list(seeds['test'])}, sets['test']
        else:
            raise ValueError(f"continual.layouts: deployment or test, not {c.layouts!r}")
        result = continual(cfg, sets, stream, groups, budget)
        targets = [float(t) for t in cfg.evaluation.targets]
        for t in targets:
            logger.info('\n' + continual_summary(result, t))
        images = {}
        for t in targets:
            path = outputs / f'continual_{100 * t:g}.png'
            plot_continual(result, t, path)
            images[f'continual/stream_{100 * t:g}'] = path
        tasks, methods = continual_rows(result)
        log.table('continual/tasks', task_columns(result), tasks)
        log.table('continual/methods', METHOD_COLUMNS, methods)
        log.images(images)
        summary_rows = {}
        for r, m, t, kind, g, rel, power, *_ in methods:
            if g in ('all', 'test'):
                summary_rows.setdefault((m, t, kind), []).append((rel, power))
        log.summary({'continual/predictor': result['predictor'], 'continual/seconds': time.time() - start,
                     **{f'continual/{m}/{t}/{kind}_{k}_mean': float(np.nanmean([v[i] for v in vs]))
                        for (m, t, kind), vs in summary_rows.items() for i, k in enumerate(('reliability', 'power'))}})
        (outputs / 'continual.json').write_text(json.dumps(result, indent=1, default=str))
        log.files('comm-power-continual', [outputs / 'continual.json', *images.values()])
    log.finish()


if __name__ == "__main__":
    main()
