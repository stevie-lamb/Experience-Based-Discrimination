"""Compact robustness sweeps: policy effect on discrimination gaps."""

from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src import market_outcomes as mo
from src.ebd import I_MU, Simulation
from src.priors import (
    FIGSIZE_TIMESERIES,
    SCENARIO_IDS,
    SCENARIO_LABELS,
    SCENARIO_LINE_COLORS,
    dissertation_title,
    group_line_color,
    plot_rc_context,
    save_dissertation_figure,
    style_axis,
)

BURN_IN = 0
# Primary metrics for report tables and headline plots.
PRIMARY_METRICS = ("mean_wage", "belief_mu")
HEADLINE_METRIC_LABELS = {
    "mean_wage": r"Mean wage gap ($g_0 - g_1$)",
    "wage_variance": r"Wage variance gap",
    "employment_rate": r"Employment-rate gap",
    "belief_mu": r"Belief $\mu$ gap ($g_0 - g_1$)",
    "minority_choice_share": r"Choice-share gap ($g_0 - g_1$)",
}
TEX_METRIC_SHORT = {
    "mean_wage": "Wage",
    "belief_mu": r"Belief $\mu$",
}
# Shared annotation for robustness figures.
DELTA_GAP_NOTE = (
    r"$\Delta\mathrm{gap} = \mathrm{gap}_{\mathrm{policy}} "
    r"- \mathrm{gap}_{\mathrm{baseline}}$, "
    r"$\mathrm{gap} = g_0 - g_1$ "
    r"(positive $\Rightarrow$ policy widens the majority--minority gap)"
)
DELTA_GAP_YLABEL = r"$\Delta$gap (policy $-$ baseline)"
WITHIN_GAP_YLABEL = r"gap ($g_0 - g_1$)"

# How each metric is plotted in robustness sweeps.
METRIC_DISPLAY = {
    "employment_rate": "within_gap",
    "mean_wage": "delta",
    "wage_variance": "delta",
    "belief_mu": "delta",
    "minority_choice_share": "delta",
}
DEFAULT_SIGNAL_BIAS_G1 = 1.0

# Group-1 prior uncertainty levels (alpha_1, beta_1); group 0 fixed at (4, 2).
PRIOR_UNCERTAINTY_LEVELS: dict[str, tuple[float, float]] = {
    "same": (4.0, 2.0),
    "mid": (3.0, 3.0),
    "high": (2.0, 4.0),
}

_PARAM_PRINT_KEYS = (
    "minority_share",
    "sigma_p",
    "sigma_signal",
    "ucb_c",
    "prior_uncertainty",
    "worker_firm_ratio",
)


def _params_summary(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={params[k]}" for k in _PARAM_PRINT_KEYS if k in params)


def _gap(g0: float, g1: float) -> float:
    return float(g0 - g1)


def _scenario_gaps(sim, *, burn_in: int = BURN_IN) -> dict[str, float]:
    """Per-metric majority-minority gap for one simulation (groups 0 vs 1)."""
    wages = mo.collect_wages_from_sim(sim, burn_in=burn_in, which="chosen")
    w0 = wages[0][np.isfinite(wages[0])]
    w1 = wages[1][np.isfinite(wages[1])]

    emp_slice = slice(burn_in, sim.horizon)
    er0 = float(np.nanmean(sim.employment_rate_log[emp_slice, 0]))
    er1 = float(np.nanmean(sim.employment_rate_log[emp_slice, 1]))

    mu0 = float(sim.firms.beliefs[:, 0, I_MU].mean())
    mu1 = float(sim.firms.beliefs[:, 1, I_MU].mean())

    eq = mo.compute_equity_outcomes(sim, burn_in=burn_in)
    cs0 = eq["by_group"]["0"]["choice_share"]
    cs1 = eq["by_group"]["1"]["choice_share"]

    mean_w0 = float(w0.mean()) if w0.size else float("nan")
    mean_w1 = float(w1.mean()) if w1.size else float("nan")
    var_w0 = (
        float(w0.var(ddof=1))
        if w0.size > 1
        else (0.0 if w0.size == 1 else float("nan"))
    )
    var_w1 = (
        float(w1.var(ddof=1))
        if w1.size > 1
        else (0.0 if w1.size == 1 else float("nan"))
    )

    return {
        "mean_wage": _gap(mean_w0, mean_w1),
        "wage_variance": _gap(var_w0, var_w1),
        "employment_rate": _gap(er0, er1),
        "belief_mu": _gap(mu0, mu1),
        "minority_choice_share": _gap(cs0, cs1),
    }


def headline_metrics(
    scenarios: dict[str, Simulation],
    *,
    burn_in: int = BURN_IN,
) -> dict[str, dict[str, float]]:
    """Within-scenario gaps and paired policy effects (unbiased and biased)."""
    gaps = {
        sid: _scenario_gaps(scenarios[sid], burn_in=burn_in) for sid in SCENARIO_IDS
    }
    out: dict[str, dict[str, float]] = {}
    for key in HEADLINE_METRIC_LABELS:
        base = gaps["baseline"][key]
        pol = gaps["policy"][key]
        base_b = gaps["baseline_bias"][key]
        pol_b = gaps["policy_bias"][key]
        out[key] = {
            "gap_baseline": base,
            "gap_policy": pol,
            "gap_baseline_bias": base_b,
            "gap_policy_bias": pol_b,
            "delta_policy": pol - base,
            "delta_policy_bias": pol_b - base_b,
        }
    return out


def build_sim_config(
    *,
    n_firms: int,
    minority_share: float,
    sigma_p: float,
    sigma_signal: float,
    prior_uncertainty: str,
    worker_firm_ratio: float,
    wage_shave: float = 0.0,
    min_wage: float = 2.0,
) -> tuple[dict, dict, dict]:
    """Build firm_kwargs, worker_kwargs, sim_kwargs from scalar parameters."""
    if prior_uncertainty not in PRIOR_UNCERTAINTY_LEVELS:
        raise ValueError(
            f"unknown prior_uncertainty={prior_uncertainty!r}; "
            f"use one of {list(PRIOR_UNCERTAINTY_LEVELS)}"
        )
    alpha_1, beta_1 = PRIOR_UNCERTAINTY_LEVELS[prior_uncertainty]
    majority_share = 1.0 - minority_share
    n_workers = max(2, int(round(worker_firm_ratio * n_firms)))

    firm_kwargs = {
        "n": n_firms,
        "mu_0": np.array([5.0, 5.0], dtype=float),
        "nu_0": np.array([10.0, 1.0], dtype=float),
        "alpha_0": np.array([4.0, alpha_1], dtype=float),
        "beta_0": np.array([2.0, beta_1], dtype=float),
        "delta_0": np.array([2.0, 2.0], dtype=float),
        "kappa_0": np.array([2.0, 2.0], dtype=float),
    }
    worker_kwargs = {
        "n": n_workers,
        "sigma_p": sigma_p,
        "sigma_signal": sigma_signal,
        "group_shares": np.array([majority_share, minority_share], dtype=float),
    }
    sim_kwargs = {
        "replace_firms": False,
        "wage_dist_which": "chosen",
        "wage_dist_scope": "all",
        "min_wage": min_wage,
        "wage_shave": wage_shave,
        "low_memory": True,
        "log_belief_history": False,
    }
    return firm_kwargs, worker_kwargs, sim_kwargs


def run_one_cell(
    params: dict[str, Any],
    *,
    n_firms: int,
    horizon: int,
    burn_in: int,
    base_seed: int,
    run_index: int | None = None,
    sweep_label: str = "",
) -> dict[str, Any]:
    """Run one paired baseline/policy simulation and return flat record."""
    label = f"[{run_index}] " if run_index is not None else ""
    prefix = f"{label}{sweep_label} ".strip()
    print(
        f"  run {prefix} starting | seed={base_seed}, "
        f"n_firms={n_firms}, horizon={horizon} | {_params_summary(params)}",
        flush=True,
    )
    t0 = time.perf_counter()
    fk, wk, sk = build_sim_config(
        n_firms=n_firms,
        minority_share=params["minority_share"],
        sigma_p=params["sigma_p"],
        sigma_signal=params["sigma_signal"],
        prior_uncertainty=params["prior_uncertainty"],
        worker_firm_ratio=params["worker_firm_ratio"],
        wage_shave=params.get("wage_shave", 0.0),
        min_wage=params.get("min_wage", 2.0),
    )
    signal_bias_g1 = float(params.get("signal_bias_g1", DEFAULT_SIGNAL_BIAS_G1))
    scenarios = Simulation.run_scenario_suite(
        fk,
        wk,
        horizon=horizon,
        burn_in=burn_in,
        base_seed=base_seed,
        signal_bias_g1=signal_bias_g1,
        sim_kwargs=sk,
        ucb_c=float(params["ucb_c"]),
    )
    metrics = headline_metrics(scenarios, burn_in=burn_in)
    record: dict[str, Any] = {
        **params,
        "base_seed": base_seed,
        "n_firms": n_firms,
        "n_workers": wk["n"],
        "horizon": horizon,
        "burn_in": burn_in,
    }
    for metric, vals in metrics.items():
        for k, v in vals.items():
            record[f"{metric}.{k}"] = v
    dt = time.perf_counter() - t0
    wage_d = metrics["mean_wage"]["delta_policy"]
    wage_d_b = metrics["mean_wage"]["delta_policy_bias"]
    belief_d = metrics["belief_mu"]["delta_policy"]
    belief_d_b = metrics["belief_mu"]["delta_policy_bias"]
    print(
        f"  run {prefix} done ({dt:.1f}s) | "
        f"Δgap wage (unbiased)={wage_d:+.3f}, (biased)={wage_d_b:+.3f}; "
        f"belief_μ (unbiased)={belief_d:+.3f}, (biased)={belief_d_b:+.3f}",
        flush=True,
    )
    return record


def _seed_list(n_seeds: int, seed_base: int = 0) -> list[int]:
    rng = np.random.default_rng(seed_base)
    return [int(x) for x in rng.integers(1, 2_000_000_000, size=n_seeds)]


def sweep_oat(
    main_spec: dict[str, Any],
    grid: dict[str, list[Any]],
    *,
    n_firms: int,
    horizon: int,
    burn_in: int = BURN_IN,
    n_seeds: int = 10,
    seed_base: int = 42,
) -> list[dict[str, Any]]:
    """One-at-a-time sweep: vary one parameter, hold others at main_spec."""
    records: list[dict[str, Any]] = []
    seeds = _seed_list(n_seeds, seed_base)
    n_cells = sum(len(v) for v in grid.values())
    print(
        f"OAT sweep: {n_cells} cells × {n_seeds} seeds = {n_cells * n_seeds} paired runs",
        flush=True,
    )
    run_index = 0
    for param, values in grid.items():
        for value in values:
            cell = deepcopy(main_spec)
            cell[param] = value
            sweep_label = f"OAT {param}={value!r}"
            for seed in seeds:
                run_index += 1
                rec = run_one_cell(
                    cell,
                    n_firms=n_firms,
                    horizon=horizon,
                    burn_in=burn_in,
                    base_seed=seed,
                    run_index=run_index,
                    sweep_label=sweep_label,
                )
                rec["sweep_type"] = "oat"
                rec["varied_param"] = param
                rec["varied_value"] = value
                records.append(rec)
    return records


def sweep_grid(
    main_spec: dict[str, Any],
    param_x: str,
    param_y: str,
    values_x: list[Any],
    values_y: list[Any],
    *,
    n_firms: int,
    horizon: int,
    burn_in: int = BURN_IN,
    n_seeds: int = 10,
    seed_base: int = 99,
    grid_name: str = "grid",
    run_index_start: int = 0,
) -> list[dict[str, Any]]:
    """2D grid over (param_x, param_y) with multiple seeds per cell."""
    records: list[dict[str, Any]] = []
    seeds = _seed_list(n_seeds, seed_base)
    n_cells = len(values_x) * len(values_y)
    print(
        f"{grid_name}: {n_cells} cells × {n_seeds} seeds = "
        f"{n_cells * n_seeds} paired runs ({param_x} × {param_y})",
        flush=True,
    )
    run_index = run_index_start
    for vx in values_x:
        for vy in values_y:
            cell = deepcopy(main_spec)
            cell[param_x] = vx
            cell[param_y] = vy
            sweep_label = f"{grid_name} {param_x}={vx!r}, {param_y}={vy!r}"
            for seed in seeds:
                run_index += 1
                rec = run_one_cell(
                    cell,
                    n_firms=n_firms,
                    horizon=horizon,
                    burn_in=burn_in,
                    base_seed=seed,
                    run_index=run_index,
                    sweep_label=sweep_label,
                )
                rec["sweep_type"] = "grid"
                rec["varied_param"] = f"{param_x}x{param_y}"
                rec["varied_value"] = f"{vx},{vy}"
                rec[param_x] = vx
                rec[param_y] = vy
                records.append(rec)
    return records


def _metric_series_keys(metric: str) -> tuple[str, ...]:
    display = METRIC_DISPLAY.get(metric, "delta")
    if display == "within_gap":
        return (
            "gap_baseline",
            "gap_policy",
            "gap_baseline_bias",
            "gap_policy_bias",
        )
    return ("delta_policy", "delta_policy_bias")


def _series_label(metric: str, key: str) -> str:
    if key == "gap_baseline":
        return rf"{SCENARIO_LABELS['baseline']} (unbiased)"
    if key == "gap_policy":
        return rf"{SCENARIO_LABELS['policy']} (unbiased)"
    if key == "gap_baseline_bias":
        return rf"{SCENARIO_LABELS['baseline_bias']} (biased)"
    if key == "gap_policy_bias":
        return rf"{SCENARIO_LABELS['policy_bias']} (biased)"
    if key == "delta_policy":
        return r"$\Delta$ (unbiased)"
    if key == "delta_policy_bias":
        return r"$\Delta$ (biased)"
    return key


def _series_color(key: str) -> str:
    if key.startswith("gap_"):
        sid = key.replace("gap_", "")
        return SCENARIO_LINE_COLORS[sid]
    if key == "delta_policy":
        return SCENARIO_LINE_COLORS["policy"]
    if key == "delta_policy_bias":
        return SCENARIO_LINE_COLORS["policy_bias"]
    return SCENARIO_LINE_COLORS["baseline"]


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and 95% CI per sweep cell and metric series."""
    if not records:
        return []

    keys = sorted(
        {(r["sweep_type"], r["varied_param"], r["varied_value"]) for r in records}
    )
    agg: list[dict[str, Any]] = []
    for sweep_type, varied_param, varied_value in keys:
        subset = [
            r
            for r in records
            if r["sweep_type"] == sweep_type
            and r["varied_param"] == varied_param
            and r["varied_value"] == varied_value
        ]
        if not subset:
            continue
        row: dict[str, Any] = {
            "sweep_type": sweep_type,
            "varied_param": varied_param,
            "varied_value": varied_value,
            "n_seeds": len(subset),
        }
        skip_keys = {
            "sweep_type",
            "varied_param",
            "varied_value",
            "base_seed",
        }
        for k, v in subset[0].items():
            if k in skip_keys or "." in k:
                continue
            row[k] = v

        for metric in HEADLINE_METRIC_LABELS:
            for series_key in _metric_series_keys(metric):
                field = f"{metric}.{series_key}"
                vals = np.array(
                    [r[field] for r in subset if field in r],
                    dtype=np.float64,
                )
                vals = vals[np.isfinite(vals)]
                prefix = f"{metric}.{series_key}"
                if vals.size == 0:
                    row[f"{prefix}.mean"] = None
                    row[f"{prefix}.ci_lo"] = None
                    row[f"{prefix}.ci_hi"] = None
                    row[f"{prefix}.sign_positive_share"] = None
                    continue
                mean = float(vals.mean())
                se = (
                    float(vals.std(ddof=1) / np.sqrt(vals.size))
                    if vals.size > 1
                    else 0.0
                )
                row[f"{prefix}.mean"] = mean
                row[f"{prefix}.ci_lo"] = mean - 1.96 * se
                row[f"{prefix}.ci_hi"] = mean + 1.96 * se
                row[f"{prefix}.sign_positive_share"] = float(np.mean(vals > 0))
            # Legacy alias for tables expecting delta_gap on non-employment metrics
            if METRIC_DISPLAY.get(metric, "delta") == "delta":
                row[f"{metric}.delta_gap"] = row.get(f"{metric}.delta_policy.mean")
                row[f"{metric}.mean"] = row.get(f"{metric}.delta_policy.mean")
                row[f"{metric}.ci_lo"] = row.get(f"{metric}.delta_policy.ci_lo")
                row[f"{metric}.ci_hi"] = row.get(f"{metric}.delta_policy.ci_hi")
                row[f"{metric}.sign_positive_share"] = row.get(
                    f"{metric}.delta_policy.sign_positive_share"
                )
        agg.append(row)
    return agg


def write_records_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_records_json(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _parse_varied_value(value: Any) -> float | str:
    if isinstance(value, (int, float)):
        return value
    s = str(value)
    if "," in s:
        return s
    try:
        return float(s)
    except ValueError:
        return s


def plot_oat_panel(
    agg: list[dict[str, Any]],
    *,
    varied_param: str,
    main_spec_value: Any,
    path: str | Path,
) -> None:
    """Small-multiples: delta_gap vs parameter value, one panel per metric."""
    subset = [
        r for r in agg if r["sweep_type"] == "oat" and r["varied_param"] == varied_param
    ]
    if not subset:
        return

    def sort_key(r: dict) -> tuple:
        v = _parse_varied_value(r["varied_value"])
        if isinstance(v, (int, float)):
            return (0, float(v))
        return (1, str(v))

    subset = sorted(subset, key=sort_key)
    x_labels = [r["varied_value"] for r in subset]
    x = np.arange(len(subset))
    n_metrics = len(HEADLINE_METRIC_LABELS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with plot_rc_context():
        fig, axes = plt.subplots(
            n_metrics,
            1,
            figsize=(FIGSIZE_TIMESERIES[0], 2.2 * n_metrics),
            layout="constrained",
            sharex=True,
        )
        if n_metrics == 1:
            axes = [axes]

        main_idx = None
        for i, lbl in enumerate(x_labels):
            if str(lbl) == str(main_spec_value):
                main_idx = i
                break

        for ax, (metric, ylab) in zip(axes, HEADLINE_METRIC_LABELS.items()):
            for series_key in _metric_series_keys(metric):
                means = [r.get(f"{metric}.{series_key}.mean") for r in subset]
                lo = [r.get(f"{metric}.{series_key}.ci_lo") for r in subset]
                hi = [r.get(f"{metric}.{series_key}.ci_hi") for r in subset]
                yerr = [
                    [
                        m - l if m is not None and l is not None else 0
                        for m, l in zip(means, lo)
                    ],
                    [
                        h - m if m is not None and h is not None else 0
                        for m, h in zip(means, hi)
                    ],
                ]
                ax.errorbar(
                    x,
                    means,
                    yerr=yerr,
                    fmt="o-",
                    color=_series_color(series_key),
                    capsize=3,
                    linewidth=1.2,
                    markersize=4,
                    label=_series_label(metric, series_key),
                )
            ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
            if main_idx is not None:
                ax.axvline(main_idx, color="0.65", linewidth=0.8, linestyle=":")
            ylabel = (
                WITHIN_GAP_YLABEL
                if METRIC_DISPLAY.get(metric) == "within_gap"
                else DELTA_GAP_YLABEL
            )
            ax.set_ylabel(ylabel)
            dissertation_title(ax, ylab)
            style_axis(ax)
            if metric == list(HEADLINE_METRIC_LABELS.keys())[0]:
                ax.legend(loc="best", fontsize=7, framealpha=0.9)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([str(v) for v in x_labels], rotation=0)
        axes[-1].set_xlabel(varied_param.replace("_", " "))
        save_dissertation_figure(fig, path)


def plot_ucb_c_by_minority_ci(
    agg: list[dict[str, Any]],
    *,
    values_minority: list[Any],
    values_ucb_c: list[Any],
    metrics: tuple[str, ...] = PRIMARY_METRICS,
    path: str | Path,
) -> None:
    """Lines of $\\Delta$gap vs policy $c$, one line per minority share, with 95% CIs."""
    subset = [
        r
        for r in agg
        if r["sweep_type"] == "grid" and r["varied_param"] == "minority_sharexucb_c"
    ]
    if not subset:
        return

    lookup: dict[tuple[Any, Any], dict] = {}
    for r in subset:
        lookup[(r["minority_share"], r["ucb_c"])] = r

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_metrics = len(metrics)
    x = np.array([float(v) for v in values_ucb_c], dtype=np.float64)
    nrows, ncols = 1, n_metrics
    fig_h = 3.6

    with plot_rc_context():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(FIGSIZE_TIMESERIES[0], fig_h),
            layout="constrained",
            squeeze=False,
        )
        for mi, metric in enumerate(metrics):
            ax = axes[0, mi]
            display = METRIC_DISPLAY.get(metric, "delta")
            if display == "within_gap":
                for series_key in _metric_series_keys(metric):
                    means, lo, hi = [], [], []
                    for uc in values_ucb_c:
                        rows = [
                            lookup.get((ms, uc))
                            for ms in values_minority
                            if lookup.get((ms, uc)) is not None
                        ]
                        if not rows:
                            means.append(np.nan)
                            lo.append(np.nan)
                            hi.append(np.nan)
                            continue
                        prefix = f"{metric}.{series_key}"
                        vals = [
                            r.get(f"{prefix}.mean")
                            for r in rows
                            if r.get(f"{prefix}.mean") is not None
                        ]
                        if not vals:
                            means.append(np.nan)
                            lo.append(np.nan)
                            hi.append(np.nan)
                            continue
                        means.append(float(np.mean(vals)))
                        lo.append(
                            float(np.mean([r.get(f"{prefix}.ci_lo") for r in rows]))
                        )
                        hi.append(
                            float(np.mean([r.get(f"{prefix}.ci_hi") for r in rows]))
                        )
                    means = np.array(means, dtype=np.float64)
                    lo = np.array(lo, dtype=np.float64)
                    hi = np.array(hi, dtype=np.float64)
                    yerr = np.vstack([means - lo, hi - means])
                    ax.errorbar(
                        x,
                        means,
                        yerr=yerr,
                        fmt="o-",
                        color=_series_color(series_key),
                        capsize=3,
                        linewidth=1.2,
                        markersize=4,
                        label=_series_label(metric, series_key),
                    )
            else:
                for gi, ms in enumerate(values_minority):
                    for series_key, ls in (
                        ("delta_policy", "-"),
                        ("delta_policy_bias", "--"),
                    ):
                        means, lo, hi = [], [], []
                        for uc in values_ucb_c:
                            row = lookup.get((ms, uc))
                            if row is None:
                                means.append(np.nan)
                                lo.append(np.nan)
                                hi.append(np.nan)
                                continue
                            prefix = f"{metric}.{series_key}"
                            means.append(row.get(f"{prefix}.mean"))
                            lo.append(row.get(f"{prefix}.ci_lo"))
                            hi.append(row.get(f"{prefix}.ci_hi"))
                        means = np.array(means, dtype=np.float64)
                        lo = np.array(lo, dtype=np.float64)
                        hi = np.array(hi, dtype=np.float64)
                        yerr = np.vstack([means - lo, hi - means])
                        ax.errorbar(
                            x,
                            means,
                            yerr=yerr,
                            fmt="o-",
                            color=group_line_color(gi),
                            linestyle=ls,
                            capsize=3,
                            linewidth=1.2,
                            markersize=4,
                            label=rf"$s={ms}$ {_series_label(metric, series_key)}",
                        )
            ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
            ax.set_xlabel(r"Policy aggressiveness $c$")
            ylabel = WITHIN_GAP_YLABEL if display == "within_gap" else DELTA_GAP_YLABEL
            if mi == 0:
                ax.set_ylabel(ylabel)
            dissertation_title(ax, HEADLINE_METRIC_LABELS[metric])
            style_axis(ax)
            if mi == 0:
                ax.legend(loc="best", fontsize=6, framealpha=0.9)
        save_dissertation_figure(fig, path)


def plot_minority_vs_sigma_p_ci(
    agg: list[dict[str, Any]],
    *,
    values_minority: list[Any],
    values_sigma_p: list[Any],
    metrics: tuple[str, ...] = PRIMARY_METRICS,
    path: str | Path,
) -> None:
    """Lines of $\\Delta$gap vs $\\sigma_p$, one line per minority share, with 95% CIs."""
    subset = [
        r
        for r in agg
        if r["sweep_type"] == "grid" and r["varied_param"] == "minority_sharexsigma_p"
    ]
    if not subset:
        return

    lookup: dict[tuple[Any, Any], dict] = {}
    for r in subset:
        lookup[(r["minority_share"], r["sigma_p"])] = r

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_metrics = len(metrics)
    x = np.array([float(v) for v in values_sigma_p], dtype=np.float64)
    if n_metrics <= 2:
        nrows, ncols = 1, n_metrics
        fig_h = 3.2
    else:
        nrows, ncols = n_metrics, 1
        fig_h = 2.4 * n_metrics

    with plot_rc_context():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(FIGSIZE_TIMESERIES[0], fig_h),
            layout="constrained",
            squeeze=False,
        )
        for mi, metric in enumerate(metrics):
            if n_metrics <= 2:
                ax = axes[0, mi]
            else:
                ax = axes[mi, 0]
            for gi, ms in enumerate(values_minority):
                means, lo, hi = [], [], []
                for sp in values_sigma_p:
                    row = lookup.get((ms, sp))
                    if row is None:
                        means.append(np.nan)
                        lo.append(np.nan)
                        hi.append(np.nan)
                        continue
                    m = row.get(f"{metric}.mean")
                    means.append(m)
                    lo.append(row.get(f"{metric}.ci_lo"))
                    hi.append(row.get(f"{metric}.ci_hi"))
                means = np.array(means, dtype=np.float64)
                lo = np.array(lo, dtype=np.float64)
                hi = np.array(hi, dtype=np.float64)
                yerr = np.vstack([means - lo, hi - means])
                color = group_line_color(gi)
                ax.errorbar(
                    x,
                    means,
                    yerr=yerr,
                    fmt="o-",
                    color=color,
                    capsize=3,
                    linewidth=1.2,
                    markersize=4,
                    label=rf"minority share $={ms}$",
                )
            ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
            ax.set_xlabel(r"$\sigma_p$ (productivity dispersion)")
            if mi == 0 or (n_metrics > 2 and mi == 0):
                ax.set_ylabel(r"$\Delta$gap")
            dissertation_title(ax, HEADLINE_METRIC_LABELS[metric])
            style_axis(ax)
            if mi == 0:
                ax.legend(loc="best", fontsize=8, framealpha=0.9)
        fig.suptitle(
            r"Policy effect on discrimination gap by minority share ($95\%$ CI over seeds)",
            fontsize=10,
        )
        save_dissertation_figure(fig, path)


def plot_grid_heatmap(
    agg: list[dict[str, Any]],
    *,
    param_x: str,
    param_y: str,
    values_x: list[Any],
    values_y: list[Any],
    metric: str = "mean_wage",
    path: str | Path,
) -> None:
    """Heatmap of mean delta_gap over a 2D parameter grid."""
    subset = [
        r
        for r in agg
        if r["sweep_type"] == "grid" and r["varied_param"] == f"{param_x}x{param_y}"
    ]
    if not subset:
        return

    lookup: dict[tuple[Any, Any], float | None] = {}
    for r in subset:
        lookup[(r[param_x], r[param_y])] = r.get(f"{metric}.mean")

    z = np.full((len(values_y), len(values_x)), np.nan)
    for iy, vy in enumerate(values_y):
        for ix, vx in enumerate(values_x):
            z[iy, ix] = lookup.get((vx, vy))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plot_rc_context():
        fig, ax = plt.subplots(figsize=FIGSIZE_TIMESERIES, layout="constrained")
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_xticks(np.arange(len(values_x)))
        ax.set_yticks(np.arange(len(values_y)))
        ax.set_xticklabels([str(v) for v in values_x])
        ax.set_yticklabels([str(v) for v in values_y])
        ax.set_xlabel(param_x.replace("_", " "))
        ax.set_ylabel(param_y.replace("_", " "))
        fig.colorbar(im, ax=ax, label=r"$\Delta$gap")
        title_y = r"$\sigma_p$" if param_y == "sigma_p" else param_y.replace("_", " ")
        title_x = (
            "minority share"
            if param_x == "minority_share"
            else param_x.replace("_", " ")
        )
        dissertation_title(
            ax,
            f"{HEADLINE_METRIC_LABELS[metric]} — {title_x} $\\times$ {title_y}",
        )
        style_axis(ax)
        save_dissertation_figure(fig, path)


def _tex_param_label(param: str) -> str:
    labels = {
        "minority_share": "Minority share",
        "sigma_p": r"$\sigma_p$",
        "sigma_signal": r"$\sigma_{\mathrm{signal}}$",
        "ucb_c": r"Policy $c$",
        "prior_uncertainty": "Prior uncertainty",
        "worker_firm_ratio": "Workers per firm",
    }
    return labels.get(param, param.replace("_", " "))


def write_robustness_tex(
    agg: list[dict[str, Any]],
    *,
    main_spec: dict[str, Any],
    path: str | Path,
    metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> None:
    """Narrow sign-stability tables: one per varied parameter, two metric columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    oat = [r for r in agg if r["sweep_type"] == "oat"]
    if not oat:
        path.write_text("% No robustness results\n", encoding="utf-8")
        return

    n_seeds = oat[0].get("n_seeds", "?")
    params = sorted({r["varied_param"] for r in oat})
    metric_headers = " & ".join(
        rf"\shortstack{{{TEX_METRIC_SHORT[m]} \\ $\Delta>0$}}" for m in metrics
    )

    lines = [
        "% Auto-generated robustness summary (narrow tables)",
        "",
        "\\subsection*{Robustness: policy effect on discrimination gaps}",
        (
            f"Each cell reports the share of {n_seeds} seeds with "
            r"$\Delta$gap $> 0$ (policy widens the majority--minority gap). "
            r"\textbf{Bold} values mark the main specification."
        ),
        "",
    ]

    for param in params:
        rows = sorted(
            [r for r in oat if r["varied_param"] == param],
            key=lambda r: str(r["varied_value"]),
        )
        lines += [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            f"\\caption{{Sign-stability: {_tex_param_label(param)}}}",
            "\\begin{tabular}{lr" + "r" * len(metrics) + "}",
            "\\hline",
            f"Value & $n$ seeds & {metric_headers} \\\\",
            "\\hline",
        ]
        for r in rows:
            val = r["varied_value"]
            is_main = str(val) == str(main_spec.get(param))
            val_str = f"\\textbf{{{val}}}" if is_main else str(val)
            cells = []
            for m in metrics:
                share = r.get(f"{m}.sign_positive_share")
                cells.append("---" if share is None else f"{100 * share:.0f}\\%")
            lines.append(
                f"{val_str} & {r.get('n_seeds', n_seeds)} & "
                + " & ".join(cells)
                + " \\\\"
            )
        lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
