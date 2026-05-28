import os
import numpy as np

from src import market_outcomes as mo
from src.ebd import I_MU, Simulation

OUTDIR = "figs/_test_3_gr"
BURN_IN = 100
os.makedirs(OUTDIR, exist_ok=True)

group_shares = np.array([0.7, 0.2, 0.1], dtype=np.float64)
N_G = len(group_shares)

fk = {
    "n": 120,
    "n_g": N_G,
}
wk = {
    "n": 180,
    "group_shares": group_shares,
}

sim = Simulation(
    fk,
    wk,
    replace_firms=False,
    horizon=120,
    wage_dist_which="chosen",
    wage_dist_scope="all",
)
sim.simulate(base_seed=40)

print("group shares target:", group_shares)
print("group shares actual:", np.bincount(sim.workers.groups, minlength=sim.ng) / sim.nw)

sim.plot_wage_cdf(f"{OUTDIR}/wage_cdf.png", survival=False)
sim.plot_wage_cdf(f"{OUTDIR}/wage_above.png", survival=True)
sim.plot_regret_over_time(f"{OUTDIR}/regret_over_time.png")
sim.plot_regret_over_time(f"{OUTDIR}/cumulative_regret.png", cumulative=True)

mo.write_market_outcomes(
    sim,
    f"{OUTDIR}/market_outcomes.json",
    f"{OUTDIR}/market_outcomes.tex",
    burn_in=BURN_IN,
    wage_which="chosen",
    base_seed=40,
)
mo.plot_wage_cdf(
    sim,
    f"{OUTDIR}/wage_cdf_post_burnin.png",
    burn_in=BURN_IN,
    which="chosen",
)

for g in range(sim.ng):
    w = mo.collect_wages_from_sim(sim, burn_in=BURN_IN, which="chosen")[g]
    w = w[np.isfinite(w)]
    if w.size == 0:
        print(f"group {g}: no post-burn-in wages")
        continue
    print(
        f"group {g}: n={w.size}, mean={w.mean():.3f}, "
        f"P(wage > 4)={float(np.mean(w > 4.0)):.1%}"
    )

# Keep original diagnostic for group index 1 and also identify lowest among all groups
mu_g1 = sim.firms.beliefs[:, 1, I_MU]
f_g1 = int(np.argmin(mu_g1))
sim.prod_timeline(f_g1, f"{OUTDIR}/irm_timeline_lowest_g1_firm{f_g1}.png")
print(f"lowest group-1 mu firm: {f_g1} (mu={mu_g1[f_g1]:.4f})")

mean_mu_by_firm = sim.firms.beliefs[:, :, I_MU].mean(axis=1)
f_any = int(np.argmin(mean_mu_by_firm))
sim.prod_timeline(f_any, f"{OUTDIR}/irm_timeline_lowest_avg_mu_firm{f_any}.png")
print(f"lowest average mu firm: {f_any} (mean mu={mean_mu_by_firm[f_any]:.4f})")
