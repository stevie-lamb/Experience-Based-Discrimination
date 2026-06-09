"""Firm-level outcomes from a completed Simulation (regret, cumulative regret plots)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from src.priors import (
    COLOR_MEAN,
    COLOR_TIMELINE_BLUE,
    COLOR_TRUE,
    FIGSIZE_TIMESERIES,
    PANEL_TITLES,
    SCENARIO_IDS,
    SCENARIO_LABELS,
    SCENARIO_LINE_COLORS,
    SCENARIO_PAIRS,
    SCENARIO_TEX_LABELS,
    dissertation_title,
    group_line_color,
    legend_dissertation,
    plot_rc_context,
    save_dissertation_figure,
    style_axis,
)

if TYPE_CHECKING:
    from src.ebd import Simulation

DEFAULT_BURN_IN = 100
ORACLE_DEFINITION = (
    "oracle = max_j(prod_j - wage_j) over both bandit arms; "
    "actual = chosen_prod - chosen_wage if hired else no_hire_cost; "
    "regret = oracle - actual"
)
PROD_ORACLE_DEFINITION = (
    "prod_oracle = max_j(prod_j) over both bandit arms; "
    "actual_prod = chosen_prod if hired else no_hire_cost; "
    "prod_regret = prod_oracle - actual_prod"
)
REGRET_QUANTILES = (1, 25, 50, 75, 99)
# Half-width (in percentile points) of the band used to group firms around a
# target regret quantile when summarising their mean final beliefs.
BELIEF_BAND_HALFWIDTH = 10.0


def _validate_burn_in(horizon: int, burn_in: int) -> None:
    if horizon <= burn_in:
        raise ValueError(f"horizon ({horizon}) must be > burn_in ({burn_in})")


def mean_cumulative_regret(regret: np.ndarray) -> np.ndarray:
    """Mean cumulative regret across firms. regret shape (T, F) -> (T,)."""
    return np.cumsum(regret, axis=0).mean(axis=1)


def total_cumulative_regret(regret: np.ndarray) -> np.ndarray:
    """Total cumulative regret summed across firms. regret shape (T, F) -> (T,)."""
    return np.cumsum(regret, axis=0).sum(axis=1)


def mean_per_period_regret(regret: np.ndarray) -> np.ndarray:
    """Mean per-period regret across firms. regret shape (T, F) -> (T,)."""
    return regret.mean(axis=1)


def total_per_period_regret(regret: np.ndarray) -> np.ndarray:
    """Total per-period regret summed across firms. regret shape (T, F) -> (T,)."""
    return regret.sum(axis=1)


def _smooth_uniform(y: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth (same length as input)."""
    window = max(1, int(window))
    if window <= 1:
        return np.asarray(y, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def _policy_label(policy_start: int | None) -> str:
    if policy_start is None or policy_start <= 0:
        return "Policy (BayesUCB)"
    return rf"Policy (BayesUCB from $t={policy_start}$)"


def _sim_metadata(sim: Simulation, *, base_seed: int | None = None) -> dict:
    return {
        "horizon": sim.horizon,
        "n_firms": sim.nf,
        "no_hire_cost": sim.no_hire_cost,
        "policy_intervention": bool(sim.policy_intervention),
        "policy_start": sim.policy_start,
        "ucb": bool(sim.ucb),
        "base_seed": base_seed,
        "worker_seed": getattr(sim, "worker_seed", None),
        "signal_bias_g1": float(sim.workers.signal_bias_g1),
        "oracle_definition": ORACLE_DEFINITION,
        "prod_oracle_definition": PROD_ORACLE_DEFINITION,
    }


def _rejection_rate_post_burnin(sim: Simulation, burn_in: int) -> float:
    accepted = sim.accepted_log[burn_in:]
    if accepted.size == 0:
        return float("nan")
    return float(np.mean(~accepted))


def _zero_regret_share(regret_slice: np.ndarray, tol: float = 1e-9) -> float:
    if regret_slice.size == 0:
        return float("nan")
    return float(np.mean(np.abs(regret_slice) <= tol))


def _quantile_dict(
    values: np.ndarray, quantiles: tuple[int, ...] = REGRET_QUANTILES
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{q}": None for q in quantiles}
    return {f"p{q}": float(np.percentile(values, q)) for q in quantiles}


def _firm_total_regret_quantiles(regret: np.ndarray, burn_in: int) -> dict:
    totals = regret[burn_in:].sum(axis=0)
    return _quantile_dict(totals)


def _most_productive_choice_rate(sim: Simulation, burn_in: int) -> dict:
    """Share of firm choices that pick the more productive of the two candidates."""
    cand_prod = sim.cand_prod_log[burn_in:]
    chosen = sim.chosen_arm_log[burn_in:]
    accepted = sim.accepted_log[burn_in:]
    if cand_prod.size == 0:
        return {"overall": None, "among_accepted": None}
    best_arm = np.argmax(cand_prod, axis=2)
    is_best = chosen == best_arm
    overall = float(is_best.mean())
    among_accepted = float(is_best[accepted].mean()) if accepted.any() else None
    return {"overall": overall, "among_accepted": among_accepted}


def _productivity_regret(sim: Simulation, burn_in: int) -> dict:
    """Hire-quality regret in pure productivity units (ignores wages)."""
    cand_prod = sim.cand_prod_log[burn_in:]
    chosen = sim.chosen_arm_log[burn_in:]
    accepted = sim.accepted_log[burn_in:]
    if cand_prod.size == 0:
        return {
            "mean_per_period": None,
            "cumulative_at_horizon": None,
            "firm_total_quantiles": _quantile_dict(np.array([])),
        }
    prod_oracle = np.max(cand_prod, axis=2)
    chosen_prod = np.take_along_axis(cand_prod, chosen[..., None], axis=2).squeeze(-1)
    actual_prod = np.where(accepted, chosen_prod, sim.no_hire_cost)
    prod_regret = prod_oracle - actual_prod
    firm_totals = prod_regret.sum(axis=0)
    return {
        "mean_per_period": float(prod_regret.mean()),
        "cumulative_at_horizon": float(firm_totals.mean()),
        "firm_total_quantiles": _quantile_dict(firm_totals),
    }


def _profit_summary(sim: Simulation, burn_in: int) -> dict:
    profit_post = sim.profit[burn_in:]
    if profit_post.size == 0:
        return {
            "mean_per_period_profit": None,
            "mean_total_profit_per_firm": None,
            "firm_total_profit_quantiles": _quantile_dict(np.array([])),
            "mean_cum_profit_full_run": float(sim.cum_profit.mean()),
        }
    firm_totals = profit_post.sum(axis=0)
    return {
        "mean_per_period_profit": float(profit_post.mean()),
        "mean_total_profit_per_firm": float(firm_totals.mean()),
        "firm_total_profit_quantiles": _quantile_dict(firm_totals),
        "mean_cum_profit_full_run": float(sim.cum_profit.mean()),
    }


def _belief_param_indices() -> dict:
    from src.ebd import I_ALPHA, I_BETA, I_MU, I_NU

    return {"mu": I_MU, "nu": I_NU, "alpha": I_ALPHA, "beta": I_BETA}


def _final_beliefs_quantiles(sim: Simulation) -> dict:
    """Per-group quantiles of each final belief hyperparameter."""
    params = _belief_param_indices()
    beliefs = sim.firms.beliefs
    return {
        str(g): {
            name: _quantile_dict(beliefs[:, g, idx]) for name, idx in params.items()
        }
        for g in range(sim.ng)
    }


def _beliefs_by_regret_quantile(sim: Simulation, burn_in: int) -> dict:
    """Mean final beliefs of firms within +/-BELIEF_BAND_HALFWIDTH pct points
    of each target regret quantile, linking belief content to performance."""
    params = _belief_param_indices()
    beliefs = sim.firms.beliefs
    firm_total_regret = sim.regret[burn_in:].sum(axis=0)
    profit_post = sim.profit[burn_in:].sum(axis=0)
    cum_profit = sim.cum_profit

    out: dict = {}
    if firm_total_regret.size == 0:
        return out

    for q in REGRET_QUANTILES:
        lo = max(0.0, q - BELIEF_BAND_HALFWIDTH)
        hi = min(100.0, q + BELIEF_BAND_HALFWIDTH)
        lo_val = np.percentile(firm_total_regret, lo)
        hi_val = np.percentile(firm_total_regret, hi)
        mask = (firm_total_regret >= lo_val) & (firm_total_regret <= hi_val)
        if not mask.any():
            continue
        out[f"p{q}"] = {
            "percentile": q,
            "band_pct": [lo, hi],
            "n_firms": int(mask.sum()),
            "mean_total_regret": float(firm_total_regret[mask].mean()),
            "mean_total_profit_post_burn_in": float(profit_post[mask].mean()),
            "mean_cum_profit_full_run": float(cum_profit[mask].mean()),
            "by_group": {
                str(g): {
                    name: float(beliefs[mask, g, idx].mean())
                    for name, idx in params.items()
                }
                for g in range(sim.ng)
            },
        }
    return out


def compute_firm_outcomes(
    sim: Simulation,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    base_seed: int | None = None,
) -> dict:
    _validate_burn_in(sim.horizon, burn_in)

    regret = sim.regret
    mean_cum = mean_cumulative_regret(regret)
    post_slice = slice(burn_in, sim.horizon)
    post_regret = regret[post_slice]

    per_period_mean_post = float(post_regret.mean()) if post_regret.size else None
    total_per_firm_post = post_regret.sum(axis=0)
    std_total_post = (
        float(total_per_firm_post.std(ddof=1)) if total_per_firm_post.size > 1 else 0.0
    )

    return {
        "metadata": _sim_metadata(sim, base_seed=base_seed),
        "burn_in": burn_in,
        "n_periods_measured": sim.horizon - burn_in,
        "mean_cumulative_regret": mean_cum.tolist(),
        "post_burn_in": {
            "mean_cumulative_regret_at_horizon": float(mean_cum[-1]),
            "mean_per_period_regret": per_period_mean_post,
            "std_total_regret_across_firms": std_total_post,
            "rejection_rate": _rejection_rate_post_burnin(sim, burn_in),
            "zero_regret_share": _zero_regret_share(post_regret),
            "firm_total_regret_quantiles": _firm_total_regret_quantiles(
                regret, burn_in
            ),
        },
        "choice_quality": _most_productive_choice_rate(sim, burn_in),
        "productivity_regret": _productivity_regret(sim, burn_in),
        "profit": _profit_summary(sim, burn_in),
        "final_beliefs": _final_beliefs_quantiles(sim),
        "beliefs_by_regret_quantile": _beliefs_by_regret_quantile(sim, burn_in),
    }


def compute_policy_comparison(
    sim_baseline: Simulation,
    sim_policy: Simulation,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    base_seed: int | None = None,
) -> dict:
    _validate_burn_in(sim_baseline.horizon, burn_in)
    if sim_baseline.horizon != sim_policy.horizon:
        raise ValueError("baseline and policy simulations must share horizon")
    if sim_baseline.nf != sim_policy.nf:
        raise ValueError("baseline and policy simulations must share n_firms")

    base_out = compute_firm_outcomes(sim_baseline, burn_in=burn_in, base_seed=base_seed)
    policy_out = compute_firm_outcomes(sim_policy, burn_in=burn_in, base_seed=base_seed)

    mean_cum_base = np.asarray(base_out["mean_cumulative_regret"])
    mean_cum_pol = np.asarray(policy_out["mean_cumulative_regret"])

    pre_match = bool(
        np.allclose(mean_cum_base[:burn_in], mean_cum_pol[:burn_in], rtol=0, atol=1e-9)
    )

    lift_at_horizon = float(mean_cum_base[-1] - mean_cum_pol[-1])

    return {
        "metadata": {
            "burn_in": burn_in,
            "base_seed": base_seed,
            "pre_burn_in_paths_match": pre_match,
            "oracle_definition": ORACLE_DEFINITION,
            "prod_oracle_definition": PROD_ORACLE_DEFINITION,
        },
        "baseline": base_out,
        "policy": policy_out,
        "comparison": {
            "policy_lift_cumulative_regret_at_horizon": lift_at_horizon,
            "mean_cumulative_regret_baseline": mean_cum_base.tolist(),
            "mean_cumulative_regret_policy": mean_cum_pol.tolist(),
        },
    }


def compute_scenario_comparison(
    scenarios: dict[str, Simulation],
    *,
    burn_in: int = DEFAULT_BURN_IN,
    base_seed: int | None = None,
) -> dict:
    """Firm outcomes for baseline, policy, and policy+signal-bias scenarios."""
    missing = [sid for sid in SCENARIO_IDS if sid not in scenarios]
    if missing:
        raise ValueError(f"missing scenario keys: {missing}")

    ref = scenarios[SCENARIO_IDS[0]]
    scenario_out: dict[str, dict] = {}
    for sid in SCENARIO_IDS:
        sim = scenarios[sid]
        if sim.horizon != ref.horizon or sim.nf != ref.nf:
            raise ValueError("all scenarios must share horizon and n_firms")
        scenario_out[sid] = compute_firm_outcomes(
            sim, burn_in=burn_in, base_seed=base_seed
        )

    lifts = {}
    for pair_id, (base_key, pol_key) in SCENARIO_PAIRS:
        base_cum = float(scenario_out[base_key]["mean_cumulative_regret"][-1])
        pol_cum = float(scenario_out[pol_key]["mean_cumulative_regret"][-1])
        lifts[f"{pair_id}_lift_cumulative_regret_at_horizon"] = base_cum - pol_cum

    return {
        "metadata": {
            "burn_in": burn_in,
            "base_seed": base_seed,
            "oracle_definition": ORACLE_DEFINITION,
            "prod_oracle_definition": PROD_ORACLE_DEFINITION,
            "signal_bias_g1": float(scenarios["policy_bias"].workers.signal_bias_g1),
            "pairs": {
                pair_id: {"baseline": base_key, "policy": pol_key}
                for pair_id, (base_key, pol_key) in SCENARIO_PAIRS
            },
        },
        "scenarios": scenario_out,
        "comparison": lifts,
    }


def _validate_scenario_sims(scenarios: dict[str, Simulation]) -> None:
    ref = scenarios[SCENARIO_IDS[0]]
    for sid in SCENARIO_IDS:
        sim = scenarios[sid]
        if sim.horizon != ref.horizon:
            raise ValueError(f"{sid}: horizon mismatch")
        if sim.nf != ref.nf:
            raise ValueError(f"{sid}: n_firms mismatch")


def _plot_pair_lines(
    ax,
    scenarios: dict[str, Simulation],
    base_key: str,
    pol_key: str,
    time: np.ndarray,
    series_by_key: dict[str, np.ndarray],
    *,
    label_suffix: str = "",
) -> None:
    for sid, ls in ((base_key, "--"), (pol_key, "-")):
        lw = 1.2 if sid.startswith("baseline") else 1.5
        label = SCENARIO_LABELS[sid]
        if label_suffix:
            label = f"{label} ({label_suffix})"
        ax.plot(
            time,
            series_by_key[sid],
            color=SCENARIO_LINE_COLORS[sid],
            linewidth=lw,
            linestyle=ls,
            label=label,
        )


def plot_regret_scenarios(
    scenarios: dict[str, Simulation],
    path: str | Path,
    *,
    smooth_window: int | None = None,
) -> None:
    """Smoothed mean per-period regret: unbiased pair (top), biased pair (bottom)."""
    _validate_scenario_sims(scenarios)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref = scenarios["baseline"]
    horizon = ref.horizon
    window = smooth_window if smooth_window is not None else max(5, horizon // 40)
    time = np.arange(horizon)

    series = {
        sid: _smooth_uniform(mean_per_period_regret(scenarios[sid].regret), window)
        for sid in SCENARIO_IDS
    }

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_TIMESERIES[0], 2.0 * FIGSIZE_TIMESERIES[1]),
            layout="constrained",
            sharex=True,
        )
        for ax, (pair_id, (base_key, pol_key)) in zip(axes, SCENARIO_PAIRS):
            _plot_pair_lines(ax, scenarios, base_key, pol_key, time, series)
            dissertation_title(
                ax,
                rf"{PANEL_TITLES[pair_id]} "
                rf"($F={ref.nf:,}$ firms, {window}-period moving average)",
            )
            ax.set_ylabel(r"mean regret (smoothed)")
            style_axis(ax)
            legend_dissertation(ax)
        axes[-1].set_xlabel(r"period $t$")
        save_dissertation_figure(fig, path)


def plot_cumulative_regret_scenarios(
    scenarios: dict[str, Simulation],
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
) -> None:
    """Total cumulative regret: unbiased pair (top), biased pair (bottom)."""
    _validate_scenario_sims(scenarios)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref = scenarios["baseline"]
    time = np.arange(ref.horizon)
    series = {
        sid: total_cumulative_regret(scenarios[sid].regret) for sid in SCENARIO_IDS
    }

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_TIMESERIES[0], 2.0 * FIGSIZE_TIMESERIES[1]),
            layout="constrained",
            sharex=True,
        )
        for ax, (pair_id, (base_key, pol_key)) in zip(axes, SCENARIO_PAIRS):
            for sid, ls in ((base_key, "--"), (pol_key, "-")):
                total_cum = series[sid]
                ax.plot(
                    time,
                    total_cum,
                    color=SCENARIO_LINE_COLORS[sid],
                    linewidth=1.2 if sid.startswith("baseline") else 1.5,
                    linestyle=ls,
                    label=rf"{SCENARIO_LABELS[sid]} (total at $T$ = {total_cum[-1]:,.0f})",
                )
            dissertation_title(
                ax,
                rf"{PANEL_TITLES[pair_id]} ($F={ref.nf:,}$ firms)",
            )
            ax.set_ylabel(r"total cumulative regret")
            style_axis(ax)
            legend_dissertation(ax, fontsize=8)
        axes[-1].set_xlabel(r"period $t$")
        save_dissertation_figure(fig, path)


def _total_regret_ylim(pair_series: dict[str, np.ndarray]) -> tuple[float, float]:
    """Independent y limits from each panel's trimmed series (small padding)."""
    if not pair_series:
        return 0.0, 1.0
    ymin_data = min(float(s.min()) for s in pair_series.values())
    ymax = max(float(s.max()) for s in pair_series.values())
    span = ymax - ymin_data
    pad = max(span * 0.08, abs(ymax) * 0.005, 1.0)
    if span <= 0:
        pad = max(abs(ymax) * 0.01, 1.0)
    return ymin_data - pad, ymax + pad


def plot_total_regret_scenarios(
    scenarios: dict[str, Simulation],
    path: str | Path,
    *,
    smooth_window: int | None = None,
    ymin: float | None = None,
    trim_ma_edges: bool = True,
    trim_zero_edges: bool = True,
    zero_tol: float = 1e-9,
) -> None:
    """Smoothed total per-period regret: unbiased pair (top), biased pair (bottom)."""
    _validate_scenario_sims(scenarios)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref = scenarios["baseline"]
    horizon = ref.horizon
    window = smooth_window if smooth_window is not None else max(5, horizon // 40)

    raw_series = {
        sid: total_per_period_regret(scenarios[sid].regret) for sid in SCENARIO_IDS
    }
    series: dict[str, np.ndarray] = {}
    time_by_pair: dict[str, np.ndarray] = {}

    for pair_id, (base_key, pol_key) in SCENARIO_PAIRS:
        pair_raw = np.stack([raw_series[base_key], raw_series[pol_key]])
        t_slice = slice(None)
        if trim_zero_edges:
            nonzero = np.any(pair_raw > zero_tol, axis=0)
            idx = np.flatnonzero(nonzero)
            if idx.size > 0:
                t_slice = slice(int(idx[0]), int(idx[-1]) + 1)
        if trim_ma_edges:
            start = t_slice.start if t_slice.start is not None else 0
            stop = t_slice.stop if t_slice.stop is not None else horizon
            start = min(start + window, stop)
            stop = max(stop - window, start)
            if start >= stop:
                t_slice = slice(None)
            else:
                t_slice = slice(start, stop)
        time_by_pair[pair_id] = np.arange(horizon)[t_slice]
        for sid in (base_key, pol_key):
            smoothed = _smooth_uniform(raw_series[sid], window)
            series[sid] = smoothed[t_slice]

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_TIMESERIES[0], 2.0 * FIGSIZE_TIMESERIES[1]),
            layout="constrained",
            sharex=False,
            sharey=False,
        )
        for ax, (pair_id, (base_key, pol_key)) in zip(axes, SCENARIO_PAIRS):
            pair_series = {sid: series[sid] for sid in (base_key, pol_key)}
            time = time_by_pair[pair_id]
            if not time.size or not pair_series[base_key].size:
                ax.text(0.5, 0.5, "No data in trimmed window", ha="center", va="center")
                continue
            if ymin is not None:
                yhi = max(float(s.max()) for s in pair_series.values()) * 1.01
                ylo = (
                    ymin
                    if ymin < yhi
                    else min(float(s.min()) for s in pair_series.values()) * 0.99
                )
            else:
                ylo, yhi = _total_regret_ylim(pair_series)
            _plot_pair_lines(ax, scenarios, base_key, pol_key, time, pair_series)
            dissertation_title(
                ax,
                rf"{PANEL_TITLES[pair_id]} "
                rf"($F={ref.nf:,}$ firms, {window}-period moving average)",
            )
            ax.set_ylabel(r"total regret (smoothed)")
            ax.set_ylim(bottom=ylo, top=yhi)
            style_axis(ax)
            legend_dissertation(ax)
        axes[-1].set_xlabel(r"period $t$")
        save_dissertation_figure(fig, path)


def plot_beliefs_mu_vs_profit_scenarios(
    scenarios: dict[str, Simulation],
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
) -> None:
    """Per-group scatter of final mu belief vs cum profit for policy scenarios."""
    _validate_scenario_sims(scenarios)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref = scenarios["baseline"]
    i_mu = _belief_param_indices()["mu"]
    plot_ids = ("policy", "policy_bias")
    n_groups = ref.ng
    width = max(FIGSIZE_TIMESERIES[0], 3.0 * n_groups * len(plot_ids))

    with plot_rc_context():
        fig, axes = plt.subplots(
            len(plot_ids),
            n_groups,
            figsize=(width / len(plot_ids) * n_groups, 2.2 * len(plot_ids)),
            layout="constrained",
            squeeze=False,
        )
        for ri, sid in enumerate(plot_ids):
            sim = scenarios[sid]
            for g in range(n_groups):
                ax = axes[ri, g]
                ax.scatter(
                    sim.cum_profit,
                    sim.firms.beliefs[:, g, i_mu],
                    color=SCENARIO_LINE_COLORS[sid],
                    s=12,
                    alpha=0.6,
                    edgecolors="none",
                )
                ax.set_xlabel(r"cumulative profit")
                if g == 0:
                    ax.set_ylabel(r"final $\mu$ belief")
                style_axis(ax)
                dissertation_title(
                    ax,
                    rf"{SCENARIO_LABELS[sid]}, group {g}: final $\mu$ vs cum.\ profit",
                )
        save_dissertation_figure(fig, path)


def plot_cumulative_regret(
    sim: Simulation,
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    baseline_sim: Simulation | None = None,
    policy_intervention: bool | None = None,
) -> None:
    """Plot total cumulative regret (summed across firms).

    With ``baseline_sim``, overlays myopic baseline vs BayesUCB policy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    time = np.arange(sim.horizon)
    total_cum_policy = total_cumulative_regret(sim.regret)
    policy_start = sim.policy_start if sim.policy_start is not None else burn_in

    with plot_rc_context():
        fig, ax = plt.subplots(figsize=FIGSIZE_TIMESERIES, layout="constrained")

        if baseline_sim is not None:
            if baseline_sim.horizon != sim.horizon:
                raise ValueError("baseline_sim and sim must share the same horizon")
            total_cum_base = total_cumulative_regret(baseline_sim.regret)
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
                total_cum_policy,
                color=COLOR_TIMELINE_BLUE,
                linewidth=1.5,
                label=rf"{_policy_label(policy_start)} "
                rf"(total at $T$ = {total_cum_policy[-1]:,.0f})",
            )
            if policy_start is not None and policy_start > 0:
                ax.axvline(
                    policy_start,
                    color=COLOR_MEAN,
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.75,
                )
            dissertation_title(
                ax,
                rf"Total cumulative regret ($F={sim.nf:,}$ firms)",
            )
        else:
            ax.plot(
                time,
                total_cum_policy,
                color=COLOR_TIMELINE_BLUE,
                linewidth=1.5,
                label=rf"Total cumulative regret (at $T$ = {total_cum_policy[-1]:,.0f})",
            )
            dissertation_title(ax, rf"Total cumulative regret ($F={sim.nf:,}$ firms)")

        ax.set_xlabel(r"period $t$")
        ax.set_ylabel(r"total cumulative regret")
        style_axis(ax)
        legend_dissertation(ax)
        save_dissertation_figure(fig, path)


def plot_regret_over_time(
    sim_baseline: Simulation,
    sim_policy: Simulation,
    path: str | Path,
    *,
    smooth_window: int | None = None,
) -> None:
    """Smoothed mean per-period regret: baseline vs policy on one panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if sim_baseline.horizon != sim_policy.horizon:
        raise ValueError("baseline and policy simulations must share horizon")
    if sim_baseline.nf != sim_policy.nf:
        raise ValueError("baseline and policy simulations must share n_firms")

    horizon = sim_policy.horizon
    window = smooth_window if smooth_window is not None else max(5, horizon // 40)
    time = np.arange(horizon)

    mean_base = _smooth_uniform(mean_per_period_regret(sim_baseline.regret), window)
    mean_pol = _smooth_uniform(mean_per_period_regret(sim_policy.regret), window)
    policy_start = sim_policy.policy_start

    with plot_rc_context():
        fig, ax = plt.subplots(figsize=FIGSIZE_TIMESERIES, layout="constrained")

        ax.plot(
            time,
            mean_base,
            color=COLOR_TRUE,
            linewidth=1.2,
            linestyle="--",
            label="Baseline (myopic)",
        )
        ax.plot(
            time,
            mean_pol,
            color=COLOR_TIMELINE_BLUE,
            linewidth=1.5,
            label=_policy_label(policy_start),
        )
        if policy_start is not None and policy_start > 0:
            ax.axvline(
                policy_start,
                color=COLOR_MEAN,
                linestyle=":",
                linewidth=0.8,
                alpha=0.75,
            )

        dissertation_title(
            ax,
            rf"Mean per-period regret ($F={sim_policy.nf:,}$ firms, "
            rf"{window}-period moving average)",
        )
        ax.set_xlabel(r"period $t$")
        ax.set_ylabel(r"mean regret (smoothed)")
        style_axis(ax)
        legend_dissertation(ax)
        save_dissertation_figure(fig, path)


def _plot_regret_pair_on_axes(
    axes,
    time: np.ndarray,
    *,
    series_base: np.ndarray,
    series_pol: np.ndarray,
    policy_start: int | None,
    title: str,
    ylabel: str,
    show_xlabel: bool,
    legend_base: str = "Baseline (myopic)",
    legend_pol: str | None = None,
) -> None:
    if legend_pol is None:
        legend_pol = _policy_label(policy_start)
    ax = axes
    ax.plot(
        time,
        series_base,
        color=COLOR_TRUE,
        linewidth=1.2,
        linestyle="--",
        label=legend_base,
    )
    ax.plot(
        time,
        series_pol,
        color=COLOR_TIMELINE_BLUE,
        linewidth=1.5,
        label=legend_pol,
    )
    if policy_start is not None and policy_start > 0:
        ax.axvline(
            policy_start,
            color=COLOR_MEAN,
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
        )
    dissertation_title(ax, title)
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(r"period $t$")
    else:
        ax.tick_params(labelbottom=False)
    style_axis(ax)
    legend_dissertation(ax)


def plot_total_regret(
    sim_baseline: Simulation,
    sim_policy: Simulation,
    path: str | Path,
    *,
    smooth_window: int | None = None,
) -> None:
    """Total regret across all firms: per-period sum (smoothed) and cumulative sum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if sim_baseline.horizon != sim_policy.horizon:
        raise ValueError("baseline and policy simulations must share horizon")
    if sim_baseline.nf != sim_policy.nf:
        raise ValueError("baseline and policy simulations must share n_firms")

    horizon = sim_policy.horizon
    n_firms = sim_policy.nf
    window = smooth_window if smooth_window is not None else max(5, horizon // 40)
    time = np.arange(horizon)
    policy_start = sim_policy.policy_start

    per_base = total_per_period_regret(sim_baseline.regret)
    per_pol = total_per_period_regret(sim_policy.regret)
    cum_base = total_cumulative_regret(sim_baseline.regret)
    cum_pol = total_cumulative_regret(sim_policy.regret)

    smooth_per_base = _smooth_uniform(per_base, window)
    smooth_per_pol = _smooth_uniform(per_pol, window)

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_TIMESERIES[0], 2.0 * FIGSIZE_TIMESERIES[1]),
            layout="constrained",
            sharex=True,
        )
        _plot_regret_pair_on_axes(
            axes[0],
            time,
            series_base=smooth_per_base,
            series_pol=smooth_per_pol,
            policy_start=policy_start,
            title=rf"Total per-period regret ($F={n_firms:,}$ firms, "
            rf"{window}-period moving average)",
            ylabel=r"total regret (smoothed)",
            show_xlabel=False,
        )
        _plot_regret_pair_on_axes(
            axes[1],
            time,
            series_base=cum_base,
            series_pol=cum_pol,
            policy_start=policy_start,
            title=rf"Total cumulative regret ($F={n_firms:,}$ firms)",
            ylabel=r"total cumulative regret",
            show_xlabel=True,
            legend_base=rf"Baseline (total at $T$ = {cum_base[-1]:,.0f})",
            legend_pol=rf"{_policy_label(policy_start)} "
            rf"(total at $T$ = {cum_pol[-1]:,.0f})",
        )
        save_dissertation_figure(fig, path)


def plot_beliefs_mu_vs_profit(
    sim: Simulation,
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
) -> None:
    """Per-group scatter of each firm's final mu belief vs full-run cumulative profit.

    Note: with replace_firms=True, bankrupt firms reset cum_profit and beliefs,
    which would distort this link; market_sim.py uses replace_firms=False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    i_mu = _belief_param_indices()["mu"]
    cum_profit = sim.cum_profit
    n_groups = sim.ng
    width = max(FIGSIZE_TIMESERIES[0], 3.0 * n_groups)

    with plot_rc_context():
        fig, axes = plt.subplots(
            1,
            n_groups,
            figsize=(width, FIGSIZE_TIMESERIES[1]),
            layout="constrained",
            squeeze=False,
        )
        for g in range(n_groups):
            ax = axes[0, g]
            ax.scatter(
                cum_profit,
                sim.firms.beliefs[:, g, i_mu],
                color=group_line_color(g),
                s=12,
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel(r"cumulative profit")
            if g == 0:
                ax.set_ylabel(r"final $\mu$ belief")
            style_axis(ax)
            dissertation_title(ax, rf"group {g}: final $\mu$ vs cum.\ profit")
        save_dissertation_figure(fig, path)


def write_firm_outcomes_json(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _tex_fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "---"
    return f"{float(x):.{nd}f}"


_QUANTILE_KEYS = tuple(f"p{q}" for q in REGRET_QUANTILES)
_QUANTILE_HEADER = " & ".join(f"$p_{{{q}}}$" for q in REGRET_QUANTILES)


def _quantile_cells(qd: dict) -> str:
    return " & ".join(_tex_fmt(qd.get(k)) for k in _QUANTILE_KEYS)


def _tex_scalar_table(o: dict) -> list[str]:
    """Choice quality, productivity (hire-quality) regret and profit scalars."""
    cq = o["choice_quality"]
    pr = o["productivity_regret"]
    pf = o["profit"]
    return [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Choice quality, productivity regret and profit (post burn-in)}",
        "\\begin{tabular}{|l|r|}",
        "\\hline",
        "Metric & Value \\\\",
        "\\hline",
        f"Most-productive choice rate (overall) & {_tex_fmt(cq['overall'])} \\\\",
        f"Most-productive choice rate (accepted hires) & {_tex_fmt(cq['among_accepted'])} \\\\",
        f"Mean per-period productivity regret & {_tex_fmt(pr['mean_per_period'])} \\\\",
        f"Cumulative productivity regret at $T$ & {_tex_fmt(pr['cumulative_at_horizon'])} \\\\",
        f"Mean per-period profit & {_tex_fmt(pf['mean_per_period_profit'])} \\\\",
        f"Mean total profit per firm & {_tex_fmt(pf['mean_total_profit_per_firm'])} \\\\",
        f"Mean full-run cumulative profit & {_tex_fmt(pf['mean_cum_profit_full_run'])} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]


def _tex_quantile_table(o: dict) -> list[str]:
    """Per-firm quantiles of total regret, total profit and productivity regret."""
    reg_q = o["post_burn_in"]["firm_total_regret_quantiles"]
    prof_q = o["profit"]["firm_total_profit_quantiles"]
    prodreg_q = o["productivity_regret"]["firm_total_quantiles"]
    return [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Per-firm quantiles (post burn-in totals)}",
        "\\begin{tabular}{|l|r|r|r|r|r|}",
        "\\hline",
        f"Metric & {_QUANTILE_HEADER} \\\\",
        "\\hline",
        f"Total regret & {_quantile_cells(reg_q)} \\\\",
        f"Total profit & {_quantile_cells(prof_q)} \\\\",
        f"Total productivity regret & {_quantile_cells(prodreg_q)} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]


def _tex_final_belief_table(o: dict) -> list[str]:
    """Per-group quantiles of final belief hyperparameters."""
    fb = o["final_beliefs"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Final belief quantiles by group}",
        "\\begin{tabular}{|l|l|r|r|r|r|r|}",
        "\\hline",
        f"Group & Param & {_QUANTILE_HEADER} \\\\",
        "\\hline",
    ]
    for g in sorted(fb, key=int):
        for param in ("mu", "nu", "alpha", "beta"):
            lines.append(f"{g} & $\\{param}$ & {_quantile_cells(fb[g][param])} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return lines


def _tex_regret_band_table(o: dict, caption_suffix: str = "") -> list[str]:
    """Mean final beliefs of firms banded around each regret quantile."""
    bands = o["beliefs_by_regret_quantile"]
    if not bands:
        return []
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Mean final beliefs by regret-quantile band "
        f"($\\pm$10 percentile points){caption_suffix}}}",
        "\\begin{tabular}{|l|l|r|r|r|r|r|r|}",
        "\\hline",
        "Band & Group & $\\mu$ & $\\nu$ & $\\alpha$ & $\\beta$ "
        "& Mean regret & Mean cum.\\ profit \\\\",
        "\\hline",
    ]
    for key in _QUANTILE_KEYS:
        band = bands.get(key)
        if band is None:
            continue
        groups = sorted(band["by_group"], key=int)
        for i, g in enumerate(groups):
            bg = band["by_group"][g]
            if i == 0:
                label = f"$p_{{{band['percentile']}}}$ (n={band['n_firms']})"
                reg = _tex_fmt(band["mean_total_regret"])
                prof = _tex_fmt(band["mean_cum_profit_full_run"])
            else:
                label = reg = prof = ""
            lines.append(
                f"{label} & {g} & {_tex_fmt(bg['mu'])} & {_tex_fmt(bg['nu'])} "
                f"& {_tex_fmt(bg['alpha'])} & {_tex_fmt(bg['beta'])} "
                f"& {reg} & {prof} \\\\"
            )
        lines.append("\\hline")
    if lines[-1] == "\\hline":
        lines[-1] = "\\hline"
    else:
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    return lines


def _tex_quantile_table_pair(base_o: dict, pol_o: dict) -> list[str]:
    """Per-firm quantiles (regret/profit/prod regret), baseline vs policy rows."""
    metrics = [
        (
            "Total regret",
            lambda o: o["post_burn_in"]["firm_total_regret_quantiles"],
        ),
        ("Total profit", lambda o: o["profit"]["firm_total_profit_quantiles"]),
        (
            "Total productivity regret",
            lambda o: o["productivity_regret"]["firm_total_quantiles"],
        ),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Per-firm quantiles (post burn-in totals)}",
        "\\begin{tabular}{|l|l|r|r|r|r|r|}",
        "\\hline",
        f"Metric & Regime & {_QUANTILE_HEADER} \\\\",
        "\\hline",
    ]
    for label, getter in metrics:
        lines.append(f"{label} & Baseline & {_quantile_cells(getter(base_o))} \\\\")
        lines.append(f" & Policy & {_quantile_cells(getter(pol_o))} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    return lines


def _tex_mu_belief_table_pair(base_o: dict, pol_o: dict, *, section: str) -> list[str]:
    """Final $\\mu$ belief quantiles by group, baseline vs policy."""
    base_fb = base_o["final_beliefs"]
    pol_fb = pol_o["final_beliefs"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Final $\\mu$ belief quantiles by group --- {section.lower()}}}",
        "\\begin{tabular}{|l|l|l|r|r|r|r|r|}",
        "\\hline",
        f"Group & Regime & {_QUANTILE_HEADER} \\\\",
        "\\hline",
    ]
    for g in sorted(pol_fb, key=int):
        lines.append(f"{g} & Baseline & {_quantile_cells(base_fb[g]['mu'])} \\\\")
        lines.append(f" & Policy & {_quantile_cells(pol_fb[g]['mu'])} \\\\")
        lines.append("\\hline")
    if lines[-1] == "\\hline":
        lines.pop()
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return lines


def write_scenario_outcomes_tex(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = _tex_fmt
    meta = out["metadata"]
    scenarios = out["scenarios"]
    comp = out.get("comparison", {})
    pair_section_titles = {
        "unbiased": "Unbiased signals",
        "biased": "Downward-biased signals",
    }

    lines = [
        "% Auto-generated firm outcomes (four scenarios, paired); "
        "\\input{results_910/firm_outcomes.tex}",
        "",
        "\\subsection*{Firm outcomes (regret)}",
        f"Burn-in $t={meta['burn_in']}$.",
        "",
    ]

    def _post_val(o: dict, key: str) -> str:
        return fmt(o["post_burn_in"][key])

    for pair_id, (base_key, pol_key) in SCENARIO_PAIRS:
        section = pair_section_titles[pair_id]
        base_o = scenarios[base_key]
        pol_o = scenarios[pol_key]

        lines += [
            f"\\subsubsection*{{{section}}}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Regret summaries (post burn-in) --- {section.lower()}}}",
            "\\begin{tabular}{|l|r|r|}",
            "\\hline",
            " & Baseline & Policy \\\\",
            "\\hline",
            f"Mean cum.\\ regret at $T$ & "
            f"{_post_val(base_o, 'mean_cumulative_regret_at_horizon')} "
            f"& {_post_val(pol_o, 'mean_cumulative_regret_at_horizon')} \\\\",
            f"Mean per-period regret & "
            f"{_post_val(base_o, 'mean_per_period_regret')} "
            f"& {_post_val(pol_o, 'mean_per_period_regret')} \\\\",
            f"Rejection rate & "
            f"{_post_val(base_o, 'rejection_rate')} "
            f"& {_post_val(pol_o, 'rejection_rate')} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Choice quality, productivity regret and profit --- {section.lower()}}}",
            "\\begin{tabular}{|l|r|r|}",
            "\\hline",
            " & Baseline & Policy \\\\",
            "\\hline",
            "Most-productive choice rate (overall) & "
            f"{fmt(base_o['choice_quality']['overall'])} "
            f"& {fmt(pol_o['choice_quality']['overall'])} \\\\",
            "Mean per-period productivity regret & "
            f"{fmt(base_o['productivity_regret']['mean_per_period'])} "
            f"& {fmt(pol_o['productivity_regret']['mean_per_period'])} \\\\",
            "Mean per-period profit & "
            f"{fmt(base_o['profit']['mean_per_period_profit'])} "
            f"& {fmt(pol_o['profit']['mean_per_period_profit'])} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

        lift_key = f"{pair_id}_lift_cumulative_regret_at_horizon"
        if comp.get(lift_key) is not None:
            lines.append(
                f"Policy lift (baseline $-$ policy cum.): {fmt(comp[lift_key])}."
            )
            lines.append("")

        lines += _tex_quantile_table_pair(base_o, pol_o)
        lines.append("")
        lines += _tex_mu_belief_table_pair(base_o, pol_o, section=section)
        lines.append("")
        for sid in (base_key, pol_key):
            lines += _tex_regret_band_table(
                scenarios[sid],
                caption_suffix=f" ({SCENARIO_LABELS[sid]}, {section.lower()})",
            )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_firm_outcomes_tex(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "scenarios" in out:
        write_scenario_outcomes_tex(out, path)
        return

    fmt = _tex_fmt

    lines = [
        "% Auto-generated firm outcomes; \\input{results/firm_outcomes.tex}",
        "",
        "\\subsection*{Firm outcomes (regret)}",
    ]

    if "comparison" in out:
        meta = out["metadata"]
        comp = out["comparison"]
        pol_post = out["policy"]["post_burn_in"]
        base_post = out["baseline"]["post_burn_in"]
        lines += [
            f"Burn-in $t={meta['burn_in']}$; pre-burn-in paths match: "
            f"{meta['pre_burn_in_paths_match']}.",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Policy vs baseline regret (post burn-in)}",
            "\\begin{tabular}{|l|r|r|}",
            "\\hline",
            " & Baseline (myopic) & Policy (UCB after burn-in) \\\\",
            "\\hline",
            f"Mean cum.\\ regret at $T$ & {fmt(base_post['mean_cumulative_regret_at_horizon'])} "
            f"& {fmt(pol_post['mean_cumulative_regret_at_horizon'])} \\\\",
            f"Mean per-period regret & {fmt(base_post['mean_per_period_regret'])} "
            f"& {fmt(pol_post['mean_per_period_regret'])} \\\\",
            f"Rejection rate & {fmt(base_post['rejection_rate'])} "
            f"& {fmt(pol_post['rejection_rate'])} \\\\",
            "\\hline",
            f"Policy lift (baseline $-$ policy cum.) & \\multicolumn{{2}}{{c}}{{{fmt(comp['policy_lift_cumulative_regret_at_horizon'])}}} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]

        base_cq = out["baseline"]["choice_quality"]
        pol_cq = out["policy"]["choice_quality"]
        base_pr = out["baseline"]["productivity_regret"]
        pol_pr = out["policy"]["productivity_regret"]
        base_pf = out["baseline"]["profit"]
        pol_pf = out["policy"]["profit"]
        lines += [
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Policy vs baseline: choice quality, productivity regret "
            "and profit (post burn-in)}",
            "\\begin{tabular}{|l|r|r|}",
            "\\hline",
            " & Baseline (myopic) & Policy (UCB after burn-in) \\\\",
            "\\hline",
            f"Most-productive choice rate (overall) & {fmt(base_cq['overall'])} "
            f"& {fmt(pol_cq['overall'])} \\\\",
            f"Mean per-period productivity regret & {fmt(base_pr['mean_per_period'])} "
            f"& {fmt(pol_pr['mean_per_period'])} \\\\",
            f"Mean per-period profit & {fmt(base_pf['mean_per_period_profit'])} "
            f"& {fmt(pol_pf['mean_per_period_profit'])} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]

        lines.append("")
        lines += _tex_quantile_table_pair(out["baseline"], out["policy"])
        lines.append("")
        lines.append(
            "Belief quantiles are shown for the baseline (myopic) and policy "
            "(BayesUCB) runs; the regret-band beliefs follow as one table per run."
        )
        lines.append("")
        lines += _tex_mu_belief_table_pair(
            out["baseline"], out["policy"], section="Unbiased signals"
        )
        lines.append("")
        lines += _tex_regret_band_table(out["policy"], caption_suffix=" (policy)")
        lines.append("")
        lines += _tex_regret_band_table(out["baseline"], caption_suffix=" (baseline)")
    else:
        post = out["post_burn_in"]
        meta = out["metadata"]
        lines += [
            f"Horizon $T={meta['horizon']}$, burn-in $t={out['burn_in']}$.",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Firm regret summaries (post burn-in)}",
            "\\begin{tabular}{|l|r|}",
            "\\hline",
            "Metric & Value \\\\",
            "\\hline",
            f"Mean cumulative regret at $T$ & {fmt(post['mean_cumulative_regret_at_horizon'])} \\\\",
            f"Mean per-period regret & {fmt(post['mean_per_period_regret'])} \\\\",
            f"Std total regret across firms & {fmt(post['std_total_regret_across_firms'])} \\\\",
            f"Rejection rate & {fmt(post['rejection_rate'])} \\\\",
            f"Zero-regret share & {fmt(post['zero_regret_share'])} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
        lines.append("")
        lines += _tex_scalar_table(out)
        lines.append("")
        lines += _tex_quantile_table(out)
        lines.append("")
        lines += _tex_final_belief_table(out)
        lines.append("")
        lines += _tex_regret_band_table(out)

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scenario_outcomes(
    scenarios: dict[str, Simulation],
    json_path: str | Path,
    tex_path: str | Path | None = None,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    base_seed: int | None = None,
) -> dict:
    out = compute_scenario_comparison(
        scenarios,
        burn_in=burn_in,
        base_seed=base_seed,
    )
    write_firm_outcomes_json(out, json_path)
    if tex_path is not None:
        write_scenario_outcomes_tex(out, tex_path)
    return out


def write_firm_outcomes(
    sim: Simulation,
    json_path: str | Path,
    tex_path: str | Path | None = None,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    base_seed: int | None = None,
    baseline_sim: Simulation | None = None,
) -> dict:
    if baseline_sim is not None:
        out = compute_policy_comparison(
            baseline_sim,
            sim,
            burn_in=burn_in,
            base_seed=base_seed,
        )
    else:
        out = compute_firm_outcomes(sim, burn_in=burn_in, base_seed=base_seed)

    write_firm_outcomes_json(out, json_path)
    if tex_path is not None:
        write_firm_outcomes_tex(out, tex_path)
    return out
