"""Diagnostics for the CSI-history regression.

Answers three questions the raw metrics don't:
  1. Is the gain predictor better than trivial baselines?
  2. What is the scale/spread of the gain, so absolute errors mean something?
  3. Where should rho_max sit for the scheduler to actually do anything?
"""

import numpy as np


def gain_stats(series_list):
    """Scale and spread of the gain, plus the step-to-step smoothness that
    determines how strong the persistence baseline will be."""
    allg = np.concatenate([np.asarray(s, float).ravel() for s in series_list])
    allg = allg[np.isfinite(allg)]
    d = np.concatenate([np.diff(np.asarray(s, float).ravel()) for s in series_list])
    d = d[np.isfinite(d)]
    return {
        "n": allg.size,
        "mean": allg.mean(),
        "std": allg.std(),
        "min": allg.min(),
        "p50": np.percentile(allg, 50),
        "max": allg.max(),
        "dynamic_range_db": 10 * np.log10(allg.max() / max(allg.min(), 1e-30)),
        "lag1_corr": float(np.corrcoef(allg[:-1], allg[1:])[0, 1]),
        "mean_abs_step": np.abs(d).mean(),
    }


def baseline_comparison(policy, series_list):
    """Ridge vs. two trivial predictors, on identical windows.

    persistence : y_hat = y_{t-1}     (the one that actually matters)
    mean        : y_hat = train mean  (the do-nothing predictor)

    R2 is computed against the mean predictor. If ridge_r2 <= persistence_r2,
    the L-lag regression is buying you nothing over a one-line rule.
    """
    y_true, y_ridge, y_persist = [], [], []
    for s in series_list:
        s = np.asarray(s, float).ravel()
        for t in range(policy.L, s.size):
            hist = s[t - policy.L:t][::-1]
            if not np.isfinite(hist).all() or not np.isfinite(s[t]):
                continue
            y_true.append(s[t])
            y_ridge.append(policy.predict_gain(hist))
            y_persist.append(hist[0])

    y = np.array(y_true)
    r = np.array(y_ridge)
    p = np.array(y_persist)
    ss_tot = ((y - y.mean()) ** 2).sum()

    def rep(pred):
        err = pred - y
        return {
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "nrmse": float(np.sqrt((err ** 2).mean()) / y.std()),
            "r2": float(1 - (err ** 2).sum() / ss_tot),
            "bias": float(err.mean()),
            "over_predict_rate": float((pred > y).mean()),
        }

    return {"ridge": rep(r), "persistence": rep(p), "n": y.size}


def power_distribution(policy, series_list, qs=(50, 75, 90, 95, 99)):
    """Genie-required power (eq. 18) across all steps, in W and dBm.

    Pick rho_max from these percentiles: setting it at the qth percentile makes
    the genie duty cycle q%, which is the regime where the scheduler is
    actually making a decision instead of always saying yes.
    """
    g = np.concatenate([np.asarray(s, float).ravel() for s in series_list])
    g = g[np.isfinite(g) & (g > 0)]
    rho = policy.snr_lin * policy.noise_lin / g

    out = {"noise_W": policy.noise_lin, "snr_lin": policy.snr_lin,
           "rho_max_current_W": policy.rho_max,
           "frac_feasible_at_current_rho_max": float((rho <= policy.rho_max).mean())}
    for q in qs:
        v = float(np.percentile(rho, q))
        out[f"rho_p{q}_W"] = v
        out[f"rho_p{q}_dBm"] = float(10 * np.log10(v * 1e3))
    return out


def outage_vs_backoff(policy, series_list, backoffs_db=(0, 1, 3, 6, 10)):
    """The curve worth putting in the paper.

    Backoff shrinks the predicted gain before inversion, so power goes up and
    outage goes down. Reports mean transmit power alongside outage so the
    trade-off is visible.
    """
    rows = []
    for b in backoffs_db:
        f = 10 ** (b / 10.0)
        outs, pows, n = 0, [], 0
        for s in series_list:
            s = np.asarray(s, float).ravel()
            for t in range(policy.L, s.size):
                hist = s[t - policy.L:t][::-1]
                if not np.isfinite(hist).all() or not np.isfinite(s[t]):
                    continue
                y_hat = policy.predict_gain(hist) / f     # pessimistic gain
                if y_hat <= 0:
                    continue
                rho = policy.snr_lin * policy.noise_lin / y_hat
                if rho > policy.rho_max:
                    continue                              # would be deferred
                n += 1
                pows.append(rho)
                if s[t] * rho / policy.noise_lin < policy.snr_lin:
                    outs += 1
        rows.append({
            "backoff_db": b,
            "outage_rate": outs / max(n, 1),
            "mean_power_W": float(np.mean(pows)) if pows else float("nan"),
            "n_tx": n,
        })
    return rows