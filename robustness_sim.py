#!/usr/bin/env python3
"""Run robustness sweeps (OAT + CI grid) and write compact figures/tables."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import robustness as rb

# Main specification (90/10 minority share, aligned with market_sim.py).
MAIN_SPEC = {
    "minority_share": 0.1,
    "sigma_p": 1.5,
    "sigma_signal": 1.5,
    "ucb_c": 1.0,
    "prior_uncertainty": "high",
    "worker_firm_ratio": 1.5,
    "wage_shave": 0.0,
    "min_wage": 2.0,
    "signal_bias_g1": 1.0,
}

# One-at-a-time grids (hold other params at MAIN_SPEC).
OAT_GRID = {
    "minority_share": [0.01, 0.1, 0.25, 0.5],
    "sigma_p": [0.0, 1.0, 2.0, 3.0],
    "sigma_signal": [0.0, 1.0, 2.0, 3.0],
    "ucb_c": [0.0, 5.0, 10.0, 25.0, 50.0],
    "prior_uncertainty": list(rb.PRIOR_UNCERTAINTY_LEVELS.keys()),
    "worker_firm_ratio": [1.0, 2.0, 3.0],
}

# CI figure: minority share x policy aggressiveness.
CI_GRID_X = "minority_share"
CI_GRID_Y = "ucb_c"
CI_VALUES_X = [0.01, 0.1, 0.25, 0.5]
CI_VALUES_Y = [0.0, 5.0, 10.0, 25.0, 50.0]

N_FIRMS = 500
HORIZON = 100
N_SEEDS = 10
BURN_IN = 0

RESULTS_DIR = Path("results")
FIGS_DIR = Path("figs")


def run_sweeps(*, quick: bool = False) -> None:
    n_firms = 100 if quick else N_FIRMS
    horizon = HORIZON
    n_seeds = 2 if quick else N_SEEDS
    oat_grid = {k: v[:2] for k, v in OAT_GRID.items()} if quick else OAT_GRID
    ci_x = CI_VALUES_X[:2] if quick else CI_VALUES_X
    ci_y = CI_VALUES_Y[:2] if quick else CI_VALUES_Y

    n_oat = sum(len(v) for v in oat_grid.values()) * n_seeds
    n_ci = len(ci_x) * len(ci_y) * n_seeds
    total_runs = n_oat + n_ci

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(
        f"Robustness sweep: n_firms={n_firms}, horizon={horizon}, "
        f"n_seeds={n_seeds}, burn_in={BURN_IN}, "
        f"total={total_runs} paired runs (OAT={n_oat}, CI={n_ci})",
        flush=True,
    )

    oat_records = rb.sweep_oat(
        MAIN_SPEC,
        oat_grid,
        n_firms=n_firms,
        horizon=horizon,
        burn_in=BURN_IN,
        n_seeds=n_seeds,
    )
    print(f"OAT: {len(oat_records)} runs completed", flush=True)

    ci_records = rb.sweep_grid(
        MAIN_SPEC,
        CI_GRID_X,
        CI_GRID_Y,
        ci_x,
        ci_y,
        n_firms=n_firms,
        horizon=horizon,
        burn_in=BURN_IN,
        n_seeds=n_seeds,
        seed_base=77,
        grid_name="CI grid",
        run_index_start=len(oat_records),
    )
    print(f"CI grid: {len(ci_records)} runs completed", flush=True)

    all_records = oat_records + ci_records
    rb.write_records_json(all_records, RESULTS_DIR / "robustness_records.json")
    rb.write_records_csv(all_records, RESULTS_DIR / "robustness_records.csv")

    agg = rb.aggregate(all_records)
    rb.write_records_json(agg, RESULTS_DIR / "robustness_agg.json")
    rb.write_records_csv(agg, RESULTS_DIR / "robustness_agg.csv")

    for param in oat_grid:
        rb.plot_oat_panel(
            agg,
            varied_param=param,
            main_spec_value=MAIN_SPEC[param],
            path=FIGS_DIR / f"robustness_oat_{param}.png",
        )
        print(f"Wrote figs/robustness_oat_{param}.png", flush=True)

    rb.plot_ucb_c_by_minority_ci(
        agg,
        values_minority=ci_x,
        values_ucb_c=ci_y,
        metrics=rb.PRIMARY_METRICS,
        path=FIGS_DIR / "robustness_ucb_c_by_minority_ci.png",
    )
    print("Wrote figs/robustness_ucb_c_by_minority_ci.png", flush=True)

    rb.write_robustness_tex(
        agg,
        main_spec=MAIN_SPEC,
        path=RESULTS_DIR / "robustness.tex",
    )
    print(f"Wrote {RESULTS_DIR / 'robustness.tex'}", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"Done in {elapsed / 60:.1f} min ({len(all_records)} paired runs)", flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness parameter sweeps")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced grid for smoke testing (2 seeds, 100 firms, horizon 100)",
    )
    args = parser.parse_args()
    run_sweeps(quick=args.quick)


if __name__ == "__main__":
    main()
