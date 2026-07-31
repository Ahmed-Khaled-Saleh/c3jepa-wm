
from policy import CommPolicy, evaluate, trajectory_gain_series
from synth_traj import make_episode_position_pairs
from c3jepa_wm.data.utils import get_valid_2d, get_h_full, np_get_csi

valid_2d    = get_valid_2d()
H_full      = get_h_full()
grid_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(valid_2d)}

episode_position_pairs = make_episode_position_pairs(valid_2d, n_episodes=2000, T=150)

series = [trajectory_gain_series(H_full, grid_to_idx, tx, rx, np_get_csi)
          for tx, rx in episode_position_pairs]

split = int(0.8 * len(series))
policy = CommPolicy(L=8, target_snr_db=15.0, rho_max=1.0).fit(series[:split])
print(evaluate(policy, series[split:]))


from diagnostics import gain_stats, baseline_comparison, power_distribution, outage_vs_backoff

print(gain_stats(series[split:]))
print(baseline_comparison(policy, series[split:]))
print(power_distribution(policy, series[split:]))
for r in outage_vs_backoff(policy, series[split:]): print(r)