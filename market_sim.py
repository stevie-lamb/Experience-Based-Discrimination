from src.ebd import Simulation

N_FIRMS = 1000
GROUP_1_SHARE = 0.1  # minority share; change to sweep parameters
WAGE_DIST_SCOPE = "final"  # "all" | "final" — periods included in CDF / means

fk = {"n": N_FIRMS, "n_g": 2}
wk = {"n": 2 * N_FIRMS, "n_g": 2, "group_1_share": GROUP_1_SHARE}

sim = Simulation(
    fk,
    wk,
    horizon=1000,
    wage_dist_which="all_offers",
    wage_dist_scope=WAGE_DIST_SCOPE,
)
sim.simulate(base_seed=42)

print(f"worker pool: group 1 share = {sim.workers.groups.mean():.3f} (target {GROUP_1_SHARE})")
print(f"candidate slots (all periods): group 1 share = {(sim.cand_groups_log == 1).mean():.3f}")
print(f"wage distribution scope: {WAGE_DIST_SCOPE}")

sim.plot_wage_cdf("figs/wage_cdf.png", survival=False)
sim.plot_wage_cdf("figs/wage_above.png", survival=True)

for g in range(sim.ng):
    w = sim.wage_by_group[g]
    print(
        f"group {g}: n={w.size}, mean={w.mean():.3f}, "
        f"P(wage > 4)={sim.fraction_above(g, 4.0):.1%}"
    )
