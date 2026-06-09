import numpy as np
from pathlib import Path

from src import firm_outcomes as fo
from src import market_outcomes as mo
from src.ebd import (
    I_MU,
    Simulation,
    find_positive_then_down_firm,
    plot_prod_timeline_pair,
)

FIGS_910 = Path("figs_910")
RESULTS_910 = Path("results_910")

N_FIRMS = 1000
WORKER_FIRM_RATIO = 2.0
HORIZON = 1000  # must be > BURN_IN
BURN_IN = 0
BASE_SEED = 198734
WAGE_DIST_SCOPE = "all"  # "all" | "final"
WAGE_SHAVE = 0.0
SIGNAL_BIAS_G1 = 2.0

FIGS_910.mkdir(parents=True, exist_ok=True)
RESULTS_910.mkdir(parents=True, exist_ok=True)

group_shares = np.array([0.9, 0.1], dtype=float)
wk = {
    "sigma_p": 1.5,
    "sigma_signal": 1.5,
    "n": int(N_FIRMS * WORKER_FIRM_RATIO),
    "group_shares": group_shares,
}

fk = {
    "n": N_FIRMS,
    "mu_0": np.array([5.0, 5.0], dtype=float),
    "nu_0": np.array([10.0, 1.0], dtype=float),
    "alpha_0": np.array([4.0, 2.0], dtype=float),
    "beta_0": np.array([2.0, 4.0], dtype=float),
    "delta_0": np.array([2.0, 2.0], dtype=float),
    "kappa_0": np.array([2.0, 2.0], dtype=float),
}

sim_kwargs = {
    "replace_firms": False,
    "wage_dist_which": "chosen",
    "wage_dist_scope": WAGE_DIST_SCOPE,
    "min_wage": 2.0,
    "wage_shave": WAGE_SHAVE,
    "low_memory": True,
    "log_belief_history": False,
    "ucb_c": 5.0,
}

scenarios = Simulation.run_scenario_suite(
    fk,
    wk,
    horizon=HORIZON,
    burn_in=BURN_IN,
    base_seed=BASE_SEED,
    signal_bias_g1=SIGNAL_BIAS_G1,
    sim_kwargs=sim_kwargs,
)

sim_base = scenarios["baseline"]
sim_policy = scenarios["policy"]
sim_bias = scenarios["policy_bias"]

actual_shares = (
    np.bincount(sim_policy.workers.groups, minlength=sim_policy.ng) / sim_policy.nw
)
print(f"group shares target: {group_shares}")
print(f"group shares actual: {actual_shares}")
print(f"signal_bias_g1 (policy_bias): {SIGNAL_BIAS_G1}")
print(
    f"policy_start={sim_policy.policy_start}, burn_in={BURN_IN}, "
    f"worker_seed={sim_policy.worker_seed}, base_seed={BASE_SEED}"
)

mo.plot_wage_cdf_scenarios(
    scenarios,
    FIGS_910 / "wage_cdf_three_scenarios.png",
    burn_in=BURN_IN,
    which="chosen",
)
mo.plot_wage_cdf_scenarios(
    scenarios,
    FIGS_910 / "wage_above_three_scenarios.png",
    burn_in=BURN_IN,
    which="chosen",
    survival=True,
)
print("Wrote figs_910/wage_cdf_three_scenarios.png (+ survival)")

mo.plot_wage_cdf_scenarios_faceted(
    scenarios,
    FIGS_910 / "wage_cdf_four_scenarios.png",
    burn_in=BURN_IN,
    which="chosen",
)
mo.plot_wage_cdf_scenarios_faceted(
    scenarios,
    FIGS_910 / "wage_above_four_scenarios.png",
    burn_in=BURN_IN,
    which="chosen",
    survival=True,
)
print("Wrote figs_910/wage_cdf_four_scenarios.png (+ survival)")

fo.plot_regret_scenarios(scenarios, FIGS_910 / "regret_over_time.png")
print("Wrote figs_910/regret_over_time.png (paired panels, smoothed)")

fo.plot_cumulative_regret_scenarios(
    scenarios,
    FIGS_910 / "cumulative_regret.png",
    burn_in=BURN_IN,
)
print("Wrote figs_910/cumulative_regret.png")

fo.plot_total_regret_scenarios(scenarios, FIGS_910 / "total_regret.png")
print("Wrote figs_910/total_regret.png")

fo.plot_beliefs_mu_vs_profit_scenarios(
    scenarios,
    FIGS_910 / "beliefs_mu_vs_profit.png",
    burn_in=BURN_IN,
)
print("Wrote figs_910/beliefs_mu_vs_profit.png")

fo.write_scenario_outcomes(
    scenarios,
    RESULTS_910 / "firm_outcomes.json",
    RESULTS_910 / "firm_outcomes.tex",
    burn_in=BURN_IN,
    base_seed=BASE_SEED,
)
print("Wrote results_910/firm_outcomes.json and results_910/firm_outcomes.tex")

market_out = mo.write_scenario_comparison(
    scenarios,
    RESULTS_910 / "market_outcomes.json",
    RESULTS_910 / "market_outcomes.tex",
    burn_in=BURN_IN,
    wage_which="chosen",
    base_seed=BASE_SEED,
    signal_bias_g1=SIGNAL_BIAS_G1,
)
print("Wrote results_910/market_outcomes.json and results_910/market_outcomes.tex")

for sid in ("baseline", "policy", "baseline_bias", "policy_bias"):
    eq = market_out["scenarios"][sid]["equity"]
    minority_g = eq["minority_group"]
    print(
        f"Equity [{sid}] (group {minority_g} minority): "
        f"choice share={eq['minority_choice_share']:.1%}, "
        f"total welfare={eq['totals']['total_welfare']:,.0f}"
    )

for g in range(sim_policy.ng):
    w = mo.collect_wages_from_sim(sim_policy, burn_in=BURN_IN, which="chosen")[g]
    w = w[np.isfinite(w)]
    if w.size == 0:
        print(f"group {g}: no post-burn-in wages (policy)")
        continue
    print(
        f"group {g} (policy): n={w.size}, mean={w.mean():.3f}, "
        f"P(wage > 4)={float(np.mean(w > 4.0)):.1%}"
    )

if sim_policy.ng > 1 and sim_policy.log_belief_history:
    mu_g1 = sim_policy.firms.beliefs[:, 1, I_MU]
    lowest_firms = np.argsort(mu_g1)[:3]
    for f in lowest_firms:
        f = int(f)
        out = FIGS_910 / f"timeline_pair_firm{f}_mu{mu_g1[f]:.2f}.png"
        plot_prod_timeline_pair(sim_policy, sim_base, f, out)
        print(f"firm {f}: group-1 mu={mu_g1[f]:.4f} -> {out}")

    pos_then_down = find_positive_then_down_firm(sim_policy, group=1)
    if pos_then_down is not None:
        out = FIGS_910 / f"timeline_pair_positive_then_down_firm{pos_then_down}.png"
        plot_prod_timeline_pair(sim_policy, sim_base, pos_then_down, out)
        print(
            f"positive-then-down firm {pos_then_down}: "
            f"group-1 mu={mu_g1[pos_then_down]:.4f} -> {out}"
        )
    else:
        print("No positive-then-down group-1 firm found.")
elif sim_policy.ng > 1:
    print("Skipped timeline figures (log_belief_history=False saves RAM).")

from scripts.extract_firm_tables import extract_tables as extract_firm_tables
from scripts.extract_market_tables import extract_tables as extract_market_tables

extract_market_tables(
    RESULTS_910 / "market_outcomes.tex",
    RESULTS_910,
)
extract_firm_tables(
    RESULTS_910 / "firm_outcomes.tex",
    RESULTS_910,
)
print(f"Extracted tables to {RESULTS_910}/")
