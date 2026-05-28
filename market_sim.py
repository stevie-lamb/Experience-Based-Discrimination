import numpy as np

from src import market_outcomes as mo
from src.ebd import I_MU, Simulation

N_FIRMS = 1000
HORIZON = 1000  # must be > BURN_IN
BURN_IN = 100
WAGE_DIST_SCOPE = "final"  # "all" | "final" — periods included in legacy ebd CDF / means

# Example: 3 groups via share vector (G inferred from length)
group_shares = np.array([0.9, 0.1], dtype=float)
wk = {"sigma_p": 2.5, "sigma_signal": 2.5, "n": 2000, "group_shares": group_shares}

# Firms now construct defaults from shares internally.
# Optional direct array overrides (backup path):
fk = {
    "n": N_FIRMS,
    "mu_0": np.array([5.0, 5.0], dtype=float),
    "nu_0": np.array([1.0, 1.0], dtype=float),
    "alpha_0": np.array([4.0, 2.0], dtype=float),
    "beta_0": np.array([2.0, 4.0], dtype=float),
    "delta_0": np.array([2.0, 2.0], dtype=float),
    "kappa_0": np.array([2.0, 2.0], dtype=float),
}

sim = Simulation(
    fk,
    wk,
    replace_firms=False,
    horizon=HORIZON,
    wage_dist_which="chosen",
    wage_dist_scope=WAGE_DIST_SCOPE,
)
sim.simulate(base_seed=40)

actual_shares = np.bincount(sim.workers.groups, minlength=sim.ng) / sim.nw
print(f"group shares target: {group_shares}")
print(f"group shares actual: {actual_shares}")
print(f"wage distribution scope (legacy plots): {WAGE_DIST_SCOPE}")

sim.plot_wage_cdf("figs/wage_cdf.png", survival=False)
sim.plot_wage_cdf("figs/wage_above.png", survival=True)
sim.plot_regret_over_time("figs/regret_over_time.png")
sim.plot_regret_over_time("figs/cumulative_regret.png", cumulative=True)

mo.write_market_outcomes(
    sim,
    "results/market_outcomes.json",
    "results/market_outcomes.tex",
    burn_in=BURN_IN,
    wage_which="chosen",
    base_seed=40,
)
print("Wrote results/market_outcomes.json and results/market_outcomes.tex")

mo.plot_wage_cdf(
    sim,
    "figs/wage_cdf_post_burnin.png",
    burn_in=BURN_IN,
    which="chosen",
)
mo.plot_wage_cdf(
    sim,
    "figs/wage_above_post_burnin.png",
    burn_in=BURN_IN,
    which="chosen",
    survival=True,
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
#
#sim.trace_firm_choice()
#

#sim.model_wide_stats()
#print(sim.emp_record)
if sim.ng > 1:
    mu_g1 = sim.firms.beliefs[:, 1, I_MU]
    lowest_firms = np.argsort(mu_g1)[:3]
    for f in lowest_firms:
        mu_val = mu_g1[f]
        out = f"figs/irm_timeline_firm{int(f)}_mu{mu_val:.2f}.png"
        sim.prod_timeline(int(f), out)
        print(f"firm {f}: group-1 mu={mu_val:.4f} -> {out}")
