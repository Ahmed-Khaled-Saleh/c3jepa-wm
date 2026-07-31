"""Communication policy: predict the instantaneous channel gain from a window of
L past gain observations, then invert the channel for the minimum power that
meets a target SNR, and schedule the transmission if that power is feasible.

Implements eqs. (21)-(23), (35)-(39) of C3-WM.

Conventions
-----------
* Gains are *linear* scalars y = |h|^2 (not dB).
* Powers and noise are in *linear watts*.
* One policy instance per directed link j -> i (or one shared instance if you
  pool links; see `fit` docstring).
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "gain_from_csi",
    "trajectory_gain_series",
    "make_windows",
    "build_channel_dataset",
    "ridge_closed_form",
    "CommPolicy",
]


# ---------------------------------------------------------------------------
# 1. Turning ray-traced CFRs into scalar gain time series
# ---------------------------------------------------------------------------

def gain_from_csi(h, reduce="mean"):
    """|h|^2 collapsed over subcarriers.

    reduce="mean" gives the average per-subcarrier gain, which is what eq. (8)
    assumes under flat fading. Use "sum" only if your SNR target is defined on
    the wideband energy -- otherwise the gain scales with subcarrier count and
    rho_max stops being calibrated.
    """
    p = np.abs(np.asarray(h)) ** 2
    return float(p.mean() if reduce == "mean" else p.sum())


def trajectory_gain_series(H_full, grid_to_idx, tx_traj, rx_traj,
                           np_get_csi, reduce="mean"):
    """Scalar gain series for one link along a pair of agent trajectories.

    tx_traj, rx_traj : sequences of (gx, gy) of equal length T.
    Returns an array of shape (T,). Time steps where the two agents occupy the
    same cell, or where a position is missing from grid_to_idx, are marked NaN
    so `make_windows` can drop any window that touches them.
    """
    T = len(tx_traj)
    out = np.full(T, np.nan)
    for t in range(T):
        tx, rx = tuple(tx_traj[t]), tuple(rx_traj[t])
        if tx == rx or tx not in grid_to_idx or rx not in grid_to_idx:
            continue
        out[t] = gain_from_csi(
            np_get_csi(H_full, grid_to_idx, tx, rx), reduce=reduce
        )
    return out


# ---------------------------------------------------------------------------
# 2. Sliding-window dataset  (eqs. 21-22)
# ---------------------------------------------------------------------------

def make_windows(series, L, drop_nonfinite=True):
    """Sliding windows of length L over a 1-D gain series.

    Row t of X is [y_{t-1}, ..., y_{t-L}] -- most recent first, matching
    eq. (22) -- and y[t] is the gain at t. Returns (X, y) with
    X.shape == (T - L, L).
    """
    s = np.asarray(series, dtype=float).ravel()
    T = s.size
    if T <= L:
        return np.empty((0, L)), np.empty((0,))

    # windows[k] = s[k : k+L], target = s[k+L]
    idx = np.arange(T - L)[:, None] + np.arange(L)[None, :]
    X = s[idx][:, ::-1]          # reverse -> most recent lag in column 0
    y = s[L:]

    if drop_nonfinite:
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]
    return X, y


def build_channel_dataset(series_list, L, drop_nonfinite=True):
    """Stack windows from many independent series (episodes and/or links).

    Windows never straddle a boundary because each series is windowed
    separately -- important, since concatenating raw series first would
    manufacture bogus lags across episode resets.
    """
    Xs, ys = [], []
    for s in series_list:
        X, y = make_windows(s, L, drop_nonfinite=drop_nonfinite)
        if len(y):
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return np.empty((0, L)), np.empty((0,))
    return np.vstack(Xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# 3. Ridge closed form  (eq. 36)
# ---------------------------------------------------------------------------

def ridge_closed_form(X, y, alpha=1e-3, add_bias=True):
    """w* = (X^T X + alpha * T * I)^{-1} X^T y.

    The alpha*T scaling matches eq. (36), where the data term carries the 1/T
    factor. With add_bias=True an unpenalised intercept column is appended;
    gains are strictly positive and usually far from zero-mean, so the
    intercept matters in practice even though eq. (23) omits it.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    T = X.shape[0]
    if T == 0:
        raise ValueError("empty design matrix -- no usable windows")

    if add_bias:
        X = np.hstack([X, np.ones((T, 1))])

    reg = alpha * T * np.eye(X.shape[1])
    if add_bias:
        reg[-1, -1] = 0.0  # don't shrink the intercept

    # solve, not inv: A is symmetric PD once regularised
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


# ---------------------------------------------------------------------------
# 4. The policy  (eqs. 18-20, 37-39)
# ---------------------------------------------------------------------------

@dataclass
class CommPolicy:
    L: int = 8                      # CSI history window
    target_snr_db: float = 15.0
    noise_power_dbm: float = -114.0
    rho_max: float = 1.0            # linear watts
    alpha: float = 1e-3
    add_bias: bool = True
    w: np.ndarray = field(default=None, repr=False)

    # --- derived linear-scale constants -----------------------------------
    @property
    def snr_lin(self):
        return 10 ** (self.target_snr_db / 10.0)

    @property
    def noise_lin(self):
        return 10 ** ((self.noise_power_dbm - 30) / 10.0)

    # --- training ---------------------------------------------------------
    def fit(self, series_list):
        """Fit on a list of 1-D gain series.

        Pool series across links only if the links are statistically alike
        (same mobility statistics, comparable path loss). Otherwise fit one
        policy per link -- the regression is learning a temporal correlation,
        and mixing links with different mean gains will bias the intercept.
        """
        X, y = build_channel_dataset(series_list, self.L)
        self.w = ridge_closed_form(X, y, self.alpha, self.add_bias)
        return self

    # --- inference --------------------------------------------------------
    def predict_gain(self, history):
        """Eq. (37): max(0, w^T x). `history` is the L most recent gains,
        most-recent-first. Returns a non-negative scalar."""
        x = np.asarray(history, dtype=float).ravel()
        if self.w is None:
            raise RuntimeError("call fit() first")
        if x.size != self.L:
            raise ValueError(f"expected {self.L} lags, got {x.size}")
        if self.add_bias:
            x = np.append(x, 1.0)
        return max(0.0, float(self.w @ x))

    def power(self, history):
        """Eq. (38): channel inversion on the predicted gain, with the
        rho_max/2 fallback when fewer than L observations are available or the
        prediction collapses to zero.

        `history` may be shorter than L (warm-up / intermittent CSI loss).
        Returns (rho_hat, used_fallback).
        """
        hist = [h for h in np.asarray(history, dtype=float).ravel()
                if np.isfinite(h)]
        if len(hist) < self.L:
            return self.rho_max / 2.0, True

        y_hat = self.predict_gain(hist[:self.L])
        if y_hat <= 0.0:
            return self.rho_max / 2.0, True
        return self.snr_lin * self.noise_lin / y_hat, False

    def schedule(self, history):
        """Eqs. (39)-(20). Returns (delta, rho_executed, rho_hat)."""
        rho_hat, _ = self.power(history)
        delta = int(0.0 <= rho_hat <= self.rho_max)
        return delta, delta * rho_hat, rho_hat

    # --- oracle, for the ablation ----------------------------------------
    def genie(self, true_gain):
        """Same decision with perfect instantaneous CSI (eqs. 18-19).
        Use as the upper bound the learned policy is measured against."""
        if true_gain <= 0:
            return 0, 0.0, np.inf
        rho = self.snr_lin * self.noise_lin / float(true_gain)
        delta = int(rho <= self.rho_max)
        return delta, delta * rho, rho


# ---------------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(policy, series_list):
    """Roll the policy along held-out series and report the metrics that
    actually matter for the paper: prediction error, achieved-SNR outage
    (allocated power too low because the gain was over-predicted), and the
    duty cycle."""
    n_mae, n_out, n_dec, n_tx = 0.0, 0, 0, 0
    sq_err, denom = 0.0, 0.0

    for s in series_list:
        s = np.asarray(s, dtype=float).ravel()
        for t in range(policy.L, s.size):
            hist = s[t - policy.L:t][::-1]   # most-recent-first
            if not np.isfinite(hist).all() or not np.isfinite(s[t]):
                continue

            y_hat = policy.predict_gain(hist)
            sq_err += (y_hat - s[t]) ** 2
            n_mae += abs(y_hat - s[t])
            denom += 1

            delta, rho, _ = policy.schedule(hist)
            n_dec += 1
            n_tx += delta
            if delta:
                achieved = s[t] * rho / policy.noise_lin
                n_out += int(achieved < policy.snr_lin)

    if denom == 0:
        return {}
    return {
        "gain_mae": n_mae / denom,
        "gain_rmse": np.sqrt(sq_err / denom),
        "duty_cycle": n_tx / max(n_dec, 1),
        "snr_outage_rate": n_out / max(n_tx, 1),
    }