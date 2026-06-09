#!/usr/bin/env python3
"""Re-render regret figures from results/firm_outcomes.json (no simulation re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.priors import (
    COLOR_MEAN,
    COLOR_TIMELINE_BLUE,
    COLOR_TRUE,
    FIGSIZE_TIMESERIES,
    dissertation_title,
    legend_dissertation,
    plot_rc_context,
    save_dissertation_figure,
    style_axis,
)
from src.firm_outcomes import _policy_label, _smooth_uniform


def _load_pair(path: Path) -> tuple[dict, dict, dict]:
    with path.open(encoding="utf-8") as f:
        out = json.load(f)
    if "comparison" not in out:
        raise ValueError(f"{path} is not a policy-vs-baseline comparison export")
    return out["baseline"], out["policy"], out["comparison"]


def plot_from_json(
    json_path: Path,
    *,
    cumulative_path: Path,
    over_time_path: Path,
    total_regret_path: Path,
    smooth_window: int | None = None,
) -> None:
    baseline, policy, _ = _load_pair(json_path)
    n_firms = int(baseline["metadata"]["n_firms"])
    horizon = int(baseline["metadata"]["horizon"])
    policy_start = policy["metadata"].get("policy_start")

    mean_cum_base = np.asarray(baseline["mean_cumulative_regret"], dtype=np.float64)
    mean_cum_pol = np.asarray(policy["mean_cumulative_regret"], dtype=np.float64)
    if mean_cum_base.size != horizon or mean_cum_pol.size != horizon:
        raise ValueError("mean_cumulative_regret length does not match horizon")

    total_cum_base = mean_cum_base * n_firms
    total_cum_pol = mean_cum_pol * n_firms
    time = np.arange(horizon)

    # Per-period mean regret from diff of mean cumulative series.
    mean_reg_base = np.empty(horizon, dtype=np.float64)
    mean_reg_base[0] = mean_cum_base[0]
    mean_reg_base[1:] = np.diff(mean_cum_base)
    mean_reg_pol = np.empty(horizon, dtype=np.float64)
    mean_reg_pol[0] = mean_cum_pol[0]
    mean_reg_pol[1:] = np.diff(mean_cum_pol)

    total_per_base = mean_reg_base * n_firms
    total_per_pol = mean_reg_pol * n_firms

    window = smooth_window if smooth_window is not None else max(5, horizon // 40)
    smooth_base = _smooth_uniform(mean_reg_base, window)
    smooth_pol = _smooth_uniform(mean_reg_pol, window)
    smooth_total_base = _smooth_uniform(total_per_base, window)
    smooth_total_pol = _smooth_uniform(total_per_pol, window)

    cumulative_path.parent.mkdir(parents=True, exist_ok=True)
    over_time_path.parent.mkdir(parents=True, exist_ok=True)
    total_regret_path.parent.mkdir(parents=True, exist_ok=True)

    with plot_rc_context():
        fig, ax = plt.subplots(figsize=FIGSIZE_TIMESERIES, layout="constrained")
        ax.plot(
            time,
            total_cum_base,
            color=COLOR_TRUE,
            linewidth=1.2,
            linestyle="--",
            label=rf"Baseline (total at $T$ = {total_cum_base[-1]:,.0f})",
        )
        ax.plot(
            time,
            total_cum_pol,
            color=COLOR_TIMELINE_BLUE,
            linewidth=1.5,
            label=rf"{_policy_label(policy_start)} "
            rf"(total at $T$ = {total_cum_pol[-1]:,.0f})",
        )
        if policy_start is not None and int(policy_start) > 0:
            ax.axvline(
                int(policy_start),
                color=COLOR_MEAN,
                linestyle=":",
                linewidth=0.8,
                alpha=0.75,
            )
        dissertation_title(ax, rf"Total cumulative regret ($F={n_firms:,}$ firms)")
        ax.set_xlabel(r"period $t$")
        ax.set_ylabel(r"total cumulative regret")
        style_axis(ax)
        legend_dissertation(ax)
        save_dissertation_figure(fig, cumulative_path)

    with plot_rc_context():
        fig, ax = plt.subplots(figsize=FIGSIZE_TIMESERIES, layout="constrained")
        ax.plot(
            time,
            smooth_base,
            color=COLOR_TRUE,
            linewidth=1.2,
            linestyle="--",
            label="Baseline (myopic)",
        )
        ax.plot(
            time,
            smooth_pol,
            color=COLOR_TIMELINE_BLUE,
            linewidth=1.5,
            label=_policy_label(policy_start),
        )
        if policy_start is not None and int(policy_start) > 0:
            ax.axvline(
                int(policy_start),
                color=COLOR_MEAN,
                linestyle=":",
                linewidth=0.8,
                alpha=0.75,
            )
        dissertation_title(
            ax,
            rf"Mean per-period regret ($F={n_firms:,}$ firms, "
            rf"{window}-period moving average)",
        )
        ax.set_xlabel(r"period $t$")
        ax.set_ylabel(r"mean regret (smoothed)")
        style_axis(ax)
        legend_dissertation(ax)
        save_dissertation_figure(fig, over_time_path)

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_TIMESERIES[0], 2.0 * FIGSIZE_TIMESERIES[1]),
            layout="constrained",
            sharex=True,
        )
        axes[0].plot(
            time,
            smooth_total_base,
            color=COLOR_TRUE,
            linewidth=1.2,
            linestyle="--",
            label="Baseline (myopic)",
        )
        axes[0].plot(
            time,
            smooth_total_pol,
            color=COLOR_TIMELINE_BLUE,
            linewidth=1.5,
            label=_policy_label(policy_start),
        )
        if policy_start is not None and int(policy_start) > 0:
            axes[0].axvline(
                int(policy_start),
                color=COLOR_MEAN,
                linestyle=":",
                linewidth=0.8,
                alpha=0.75,
            )
        dissertation_title(
            axes[0],
            rf"Total per-period regret ($F={n_firms:,}$ firms, "
            rf"{window}-period moving average)",
        )
        axes[0].set_ylabel(r"total regret (smoothed)")
        axes[0].tick_params(labelbottom=False)
        style_axis(axes[0])
        legend_dissertation(axes[0])

        axes[1].plot(
            time,
            total_cum_base,
            color=COLOR_TRUE,
            linewidth=1.2,
            linestyle="--",
            label=rf"Baseline (total at $T$ = {total_cum_base[-1]:,.0f})",
        )
        axes[1].plot(
            time,
            total_cum_pol,
            color=COLOR_TIMELINE_BLUE,
            linewidth=1.5,
            label=rf"{_policy_label(policy_start)} "
            rf"(total at $T$ = {total_cum_pol[-1]:,.0f})",
        )
        if policy_start is not None and int(policy_start) > 0:
            axes[1].axvline(
                int(policy_start),
                color=COLOR_MEAN,
                linestyle=":",
                linewidth=0.8,
                alpha=0.75,
            )
        dissertation_title(
            axes[1],
            rf"Total cumulative regret ($F={n_firms:,}$ firms)",
        )
        axes[1].set_xlabel(r"period $t$")
        axes[1].set_ylabel(r"total cumulative regret")
        style_axis(axes[1])
        legend_dissertation(axes[1])
        save_dissertation_figure(fig, total_regret_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render regret figures from firm_outcomes.json"
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="results/firm_outcomes.json",
        type=Path,
    )
    parser.add_argument(
        "--cumulative",
        type=Path,
        default=Path("figs/cumulative_regret.png"),
    )
    parser.add_argument(
        "--over-time",
        type=Path,
        default=Path("figs/regret_over_time.png"),
    )
    parser.add_argument(
        "--total-regret",
        type=Path,
        default=Path("figs/total_regret.png"),
    )
    parser.add_argument("--smooth-window", type=int, default=None)
    args = parser.parse_args()

    plot_from_json(
        args.json_path,
        cumulative_path=args.cumulative,
        over_time_path=args.over_time,
        total_regret_path=args.total_regret,
        smooth_window=args.smooth_window,
    )
    print(f"Wrote {args.cumulative}")
    print(f"Wrote {args.over_time}")
    print(f"Wrote {args.total_regret}")


if __name__ == "__main__":
    main()
