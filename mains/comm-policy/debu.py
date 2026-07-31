"""Inspect H_full directly. If the gains are normalised, this shows it in
three lines of output.
"""

import numpy as np


def inspect_h_full(H_full, valid_2d, np_get_csi, n_probe=200, seed=0):
    grid_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(valid_2d)}
    cells = [(int(x), int(y)) for x, y in valid_2d]
    rng = np.random.default_rng(seed)

    print("H_full shape :", H_full.shape)
    print("H_full dtype :", H_full.dtype)
    p = np.abs(H_full) ** 2
    print("raw |H|^2  mean/min/max: %.4e %.4e %.4e"
          % (p.mean(), p.min(), p.max()))
    del p

    rows = []
    for _ in range(n_probe):
        tx, rx = rng.choice(len(cells), 2, replace=False)
        tx, rx = cells[tx], cells[rx]
        h = np.asarray(np_get_csi(H_full, grid_to_idx, tx, rx))
        d = np.hypot(tx[0] - rx[0], tx[1] - rx[1])
        rows.append((d,
                     float((np.abs(h) ** 2).mean()),
                     float((np.abs(h) ** 2).sum()),
                     h.size))

    d = np.array([r[0] for r in rows])
    gm = np.array([r[1] for r in rows])
    gs = np.array([r[2] for r in rows])
    n_sub = rows[0][3]

    print("n_subcarriers:", n_sub)
    print("mean|h|^2  : mean=%.6e  min=%.6e  max=%.6e  range=%.2f dB"
          % (gm.mean(), gm.min(), gm.max(), 10 * np.log10(gm.max() / gm.min())))
    print("sum |h|^2  : mean=%.6e  (== n_sub if per-subcarrier normalised)"
          % gs.mean())

    # Path loss sanity: log-gain should fall roughly linearly with log-distance.
    ok = (d > 0) & (gm > 0)
    if ok.sum() > 10:
        slope = np.polyfit(np.log10(d[ok]), 10 * np.log10(gm[ok]), 1)[0]
        corr = np.corrcoef(np.log10(d[ok]), 10 * np.log10(gm[ok]))[0, 1]
        print("path-loss slope: %.2f dB/decade   corr(log d, dB gain)=%.3f"
              % (slope, corr))
        print("  -> expect roughly -20 to -40 dB/decade and corr near -0.8;")
        print("     slope ~0 and corr ~0 means path loss has been normalised out.")
    return {"mean_gain": gm, "dist": d, "n_sub": n_sub}


if __name__ == "__main__":
    from c3jepa_wm.data.utils import get_valid_2d, get_h_full, np_get_csi

    valid_2d = get_valid_2d()
    H_full = get_h_full()
    inspect_h_full(H_full, valid_2d, np_get_csi)