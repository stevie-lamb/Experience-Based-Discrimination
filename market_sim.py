from src.ebd import Simulation

N_FIRMS = 10000
GROUP_1_SHARE = 0.1  # minority share; change to sweep parameters
WAGE_DIST_SCOPE = "final"  # "all" | "final" — periods included in CDF / means

fk = {"n": N_FIRMS, "n_g": 2}
wk = {"n": 15000, "n_g": 2, "group_1_share": GROUP_1_SHARE}

sim = Simulation(
    fk,
    wk,
    horizon=100,
    wage_dist_which="chosen",
    wage_dist_scope=WAGE_DIST_SCOPE,
)
sim.simulate(base_seed=40)

print(f"worker pool: group 1 share = {sim.workers.groups.mean():.3f} (target {GROUP_1_SHARE})")
print(f"candidate slots (all periods): group 1 share = {(sim.cand_groups_log == 1).mean():.3f}")
print(f"wage distribution scope: {WAGE_DIST_SCOPE}")

sim.plot_wage_cdf("figs/wage_cdf.png", survival=False)
sim.plot_wage_cdf("figs/wage_above.png", survival=True)
sim.plot_regret_over_time("figs/regret_over_time.png")
sim.plot_regret_over_time("figs/cumulative_regret.png", cumulative=True)

for g in range(sim.ng):
    w = sim.wage_by_group[g]
    print(
        f"group {g}: n={w.size}, mean={w.mean():.3f}, "
        f"P(wage > 4)={sim.fraction_above(g, 4.0):.1%}"
    )
#
#sim.trace_firm_choice()
#

#sim.model_wide_stats()
#print(sim.emp_record)

#######################################
#Sim2 w/o ucb
#######################################


sim2 = Simulation(
    fk,
    wk,
    horizon=1000,
    wage_dist_which="chosen",
    wage_dist_scope=WAGE_DIST_SCOPE,
    ucb=True
)
sim2.simulate(base_seed=40)

print(f"worker pool: group 1 share = {sim2.workers.groups.mean():.3f} (target {GROUP_1_SHARE})")
print(f"candidate slots (all periods): group 1 share = {(sim2.cand_groups_log == 1).mean():.3f}")
print(f"wage distribution scope: {WAGE_DIST_SCOPE}")

sim2.plot_wage_cdf("figs/wage_cdf_ucb.png", survival=False)
sim2.plot_wage_cdf("figs/wage_above_ucb.png", survival=True)
sim2.plot_regret_over_time("figs/regret_over_time_ucb.png")
sim2.plot_regret_over_time("figs/cumulative_regret_ucb.png", cumulative=True)

for g in range(sim2.ng):
    w = sim2.wage_by_group[g]
    print(
        f"group {g}: n={w.size}, mean={w.mean():.3f}, "
        f"P(wage > 4)={sim2.fraction_above(g, 4.0):.1%}"
    )
#
#sim2.trace_firm_choice()
#

#sim2.model_wide_stats()
#print(sim2.emp_record)