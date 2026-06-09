import numpy as np
import matplotlib.pyplot as plt
from src.ebd import Workers, Firms

if __name__ == "__main__":

    horizon = 1000
    n_runs = 100
    n_groups = Firms().n_g

    # wages_by_run_group[run, group, t] = mean wage offer to that group at time t
    wages_by_run_group = np.full((n_runs, n_groups, horizon), np.nan, dtype=float)

    for run in range(n_runs):
        y = Firms()
        for t in range(horizon):
            x = Workers()
            wage_offers = [
                y.wage_offer(0, x, worker) for worker in x.workers_info.keys()
            ]
            signals = [x.workers_info[worker][1] for worker in x.workers_info.keys()]

            # Record all offers this period (irrespective of acceptance/hiring).
            for g in range(n_groups):
                group_offers = [
                    wage_offers[worker]
                    for worker in x.workers_info.keys()
                    if x.workers_info[worker][2] == g
                ]
                if len(group_offers) > 0:
                    wages_by_run_group[run, g, t] = float(np.mean(group_offers))

            expected_profit = [s - w for w, s in zip(wage_offers, signals)]
            matched_worker = np.argmax(expected_profit)
            offered_group = x.workers_info[matched_worker][2]
            offered_wage = wage_offers[matched_worker]
            accepted = x.accept_wage(matched_worker, offered_wage)

            if accepted:
                y.update_priors(0, x, matched_worker)

    expected_wages = np.nanmean(wages_by_run_group, axis=0)
    time = np.arange(horizon)

    window = 25
    kernel = np.ones(window)

    plt.figure(figsize=(12, 6))
    for g in range(n_groups):
        # Plot individual run paths in background
        for run in range(n_runs):
            plt.plot(
                time,
                wages_by_run_group[run, g, :],
                alpha=0.025,
                linewidth=1,
                color=f"C{g}",
            )

        # Plot raw expected wage path
        plt.plot(
            time,
            expected_wages[g, :],
            color=f"C{g}",
            linewidth=1.5,
            label=f"Group {g} expected wage",
        )

        # Smooth expected wage path via NaN-aware moving average
        expected_series = expected_wages[g, :]
        values_filled = np.nan_to_num(expected_series, nan=0.0)
        obs_counts = (~np.isnan(expected_series)).astype(float)
        smooth_num = np.convolve(values_filled, kernel, mode="same")
        smooth_den = np.convolve(obs_counts, kernel, mode="same")
        smooth_series = np.divide(
            smooth_num,
            smooth_den,
            out=np.full_like(smooth_num, np.nan, dtype=float),
            where=smooth_den > 0,
        )

        plt.plot(
            time,
            smooth_series,
            color=f"C{g}",
            linewidth=3,
            label=f"Group {g} expected wage (smoothed)",
        )

    plt.title("Expected Wage Offers by Group (100 runs x 1000 periods)")
    plt.xlabel("Time")
    plt.ylabel("Offered Wage")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/wage_paths.png", dpi=150)
    print("Saved plot to figs/wage_paths.png")
    plt.close()
