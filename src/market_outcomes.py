"""Market-level outcomes from a completed Simulation (employment, wages, KS, reports)."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from src.ebd import Simulation

DEFAULT_BURN_IN = 100
WAGE_WHICH_CHOICES = ("chosen", "accepted", "all_offers")


def _validate_burn_in(horizon: int, burn_in: int) -> None:
    if horizon <= burn_in:
        raise ValueError(
            f"horizon ({horizon}) must be > burn_in ({burn_in})"
        )


def _validate_wage_which(which: str) -> None:
    if which not in WAGE_WHICH_CHOICES:
        raise ValueError(
            f"invalid wage_which={which!r}; use one of {WAGE_WHICH_CHOICES}"
        )


def period_range(
    horizon: int,
    burn_in: int = 0,
    scope: str = "all",
) -> range | list[int]:
    if scope == "all":
        start = max(0, burn_in)
        if start >= horizon:
            raise ValueError(
                f"burn_in={burn_in} >= horizon={horizon}; no periods to measure"
            )
        return range(start, horizon)
    if scope == "final":
        return [horizon - 1]
    raise ValueError(f'unknown scope={scope!r}; use "all" or "final"')


def collect_wages_from_sim(
    sim: Simulation,
    *,
    burn_in: int,
    which: str,
) -> dict[int, np.ndarray]:
    _validate_burn_in(sim.horizon, burn_in)
    _validate_wage_which(which)

    buckets: dict[int, list[float]] = {g: [] for g in range(sim.ng)}

    for t in range(burn_in, sim.horizon):
        if which == "chosen":
            for f in range(sim.nf):
                g = int(sim.chosen_group_log[t, f])
                buckets[g].append(sim.chosen_wage_log[t, f])
        elif which == "all_offers":
            for f in range(sim.nf):
                for j in range(2):
                    g = int(sim.cand_groups_log[t, f, j])
                    buckets[g].append(sim.wage_offers[t, f, j])
        elif which == "accepted":
            for f in range(sim.nf):
                if sim.accepted_log[t, f]:
                    g = int(sim.chosen_group_log[t, f])
                    buckets[g].append(sim.accepted_wages[t, f])

    return {g: np.asarray(buckets[g], dtype=np.float64) for g in buckets}


def _summarize_series(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "std": None, "final": None}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "final": float(x[-1]),
    }


def _wage_moments(w: np.ndarray) -> dict:
    w = w[np.isfinite(w)]
    if w.size == 0:
        return {
            "n": 0,
            "mean": None,
            "variance": None,
            "skewness": None,
            "kurtosis": None,
        }
    return {
        "n": int(w.size),
        "mean": float(w.mean()),
        "variance": float(w.var(ddof=1)) if w.size > 1 else 0.0,
        "skewness": float(stats.skew(w, bias=False)) if w.size > 2 else None,
        "kurtosis": float(stats.kurtosis(w, fisher=True, bias=False))
        if w.size > 3
        else None,
    }


def compute_employment_outcomes(
    sim: Simulation,
    *,
    burn_in: int,
    include_time_series: bool = True,
) -> dict:
    _validate_burn_in(sim.horizon, burn_in)
    emp_slice = slice(burn_in, sim.horizon)

    by_group: dict[str, dict] = {}
    for g in range(sim.ng):
        er = sim.employment_rate_log[emp_slice, g]
        ur = sim.unemployment_rate_log[emp_slice, g]
        by_group[str(g)] = {
            "employment_rate": _summarize_series(er),
            "unemployment_rate": _summarize_series(ur),
        }

    block: dict = {"by_group": by_group}
    if include_time_series:
        block["time_series"] = {
            "employment_rate": [
                [int(t)]
                + [
                    float(sim.employment_rate_log[t, g])
                    if np.isfinite(sim.employment_rate_log[t, g])
                    else None
                    for g in range(sim.ng)
                ]
                for t in range(burn_in, sim.horizon)
            ],
            "unemployment_rate": [
                [int(t)]
                + [
                    float(sim.unemployment_rate_log[t, g])
                    if np.isfinite(sim.unemployment_rate_log[t, g])
                    else None
                    for g in range(sim.ng)
                ]
                for t in range(burn_in, sim.horizon)
            ],
        }
    return block


def compute_wage_outcomes(
    sim: Simulation,
    *,
    burn_in: int,
    which: str,
) -> dict:
    wages_by_group = collect_wages_from_sim(sim, burn_in=burn_in, which=which)

    by_group: dict[str, dict] = {}
    for g, w in wages_by_group.items():
        by_group[str(g)] = _wage_moments(w)

    ks_pairwise: dict[str, dict] = {}
    for i, j in combinations(range(sim.ng), 2):
        wi = wages_by_group[i]
        wj = wages_by_group[j]
        key = f"{i}_vs_{j}"
        if wi.size < 2 or wj.size < 2:
            ks_pairwise[key] = {
                "insufficient_data": True,
                "n_i": int(wi.size),
                "n_j": int(wj.size),
            }
            continue
        res = stats.ks_2samp(wi, wj, method="auto")
        ks_pairwise[key] = {
            "insufficient_data": False,
            "statistic": float(res.statistic),
            "pvalue": float(res.pvalue),
            "n_i": int(wi.size),
            "n_j": int(wj.size),
        }

    return {"by_group": by_group, "ks_pairwise": ks_pairwise}


def compute_market_outcomes(
    sim: Simulation,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    wage_which: str = "chosen",
    include_time_series: bool = True,
    base_seed: int | None = None,
) -> dict:
    _validate_burn_in(sim.horizon, burn_in)
    _validate_wage_which(wage_which)

    actual_shares = np.bincount(sim.workers.groups, minlength=sim.ng) / sim.nw

    return {
        "metadata": {
            "horizon": sim.horizon,
            "burn_in": burn_in,
            "n_periods_measured": sim.horizon - burn_in,
            "wage_which": wage_which,
            "n_firms": sim.nf,
            "n_workers": sim.nw,
            "n_groups": sim.ng,
            "group_shares_target": sim.workers.group_shares.tolist(),
            "group_shares_actual": actual_shares.tolist(),
            "pool_sizes": sim.pool_sizes.tolist(),
            "base_seed": base_seed,
            "ucb": sim.ucb,
        },
        "employment": compute_employment_outcomes(
            sim,
            burn_in=burn_in,
            include_time_series=include_time_series,
        ),
        "wages": compute_wage_outcomes(sim, burn_in=burn_in, which=wage_which),
    }


def build_cdf_by_group(
    wages_by_group: dict[int, np.ndarray],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cdf_by_group: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for g, w in wages_by_group.items():
        w = w[np.isfinite(w)]
        if w.size == 0:
            cdf_by_group[g] = (np.array([]), np.array([]))
            continue
        x = np.sort(w)
        y = np.arange(1, w.size + 1, dtype=np.float64) / w.size
        cdf_by_group[g] = (x, y)
    return cdf_by_group


def write_market_outcomes_json(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def write_market_outcomes_tex(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = out["metadata"]
    ng = meta["n_groups"]

    def fmt(x, nd: int = 4) -> str:
        if x is None:
            return "---"
        return f"{x:.{nd}f}"

    lines = [
        "% Auto-generated market outcomes; \\input{results/market_outcomes.tex}",
        f"% horizon={meta['horizon']}, burn_in={meta['burn_in']}, "
        f"wage_which={meta['wage_which']}",
        "",
        "\\subsection*{Market outcomes (post burn-in)}",
        f"Measurements use periods $t \\geq {meta['burn_in']}$ "
        f"({meta['n_periods_measured']} periods). "
        f"Wages pooled with \\texttt{{{meta['wage_which']}}}.",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Employment and unemployment rates by group "
        "(time average post burn-in)}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Group & $\\bar{e}_g$ & $\\bar{u}_g$ & $e_g$ (final) \\\\",
        "\\midrule",
    ]

    for g in range(ng):
        row = out["employment"]["by_group"][str(g)]
        er = row["employment_rate"]
        ur = row["unemployment_rate"]
        lines.append(
            f"{g} & {fmt(er['mean'])} & {fmt(ur['mean'])} & {fmt(er['final'])} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wage distribution moments by group}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Group & $n$ & mean & var & skew & kurt \\\\",
        "\\midrule",
    ]

    for g in range(ng):
        w = out["wages"]["by_group"][str(g)]
        lines.append(
            f"{g} & {w['n']} & {fmt(w['mean'])} & {fmt(w['variance'])} "
            f"& {fmt(w['skewness'])} & {fmt(w['kurtosis'])} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Pairwise Kolmogorov--Smirnov tests on wages}",
        "\\begin{tabular}{llr}",
        "\\toprule",
        "Pair & $D$ & $p$-value \\\\",
        "\\midrule",
    ]

    for key, ks in out["wages"]["ks_pairwise"].items():
        label = key.replace("_", " ")
        if ks.get("insufficient_data"):
            lines.append(f"{label} & --- & --- \\\\")
        else:
            p = ks["pvalue"]
            p_str = f"{p:.2e}" if p < 0.001 else fmt(p)
            lines.append(
                f"{label} & {fmt(ks['statistic'])} & {p_str} \\\\"
            )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_market_outcomes(
    sim: Simulation,
    json_path: str | Path,
    tex_path: str | Path | None = None,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    wage_which: str = "chosen",
    include_time_series: bool = True,
    base_seed: int | None = None,
) -> dict:
    out = compute_market_outcomes(
        sim,
        burn_in=burn_in,
        wage_which=wage_which,
        include_time_series=include_time_series,
        base_seed=base_seed,
    )
    write_market_outcomes_json(out, json_path)
    if tex_path is not None:
        write_market_outcomes_tex(out, tex_path)
    return out


def plot_wage_cdf(
    sim: Simulation,
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    which: str = "chosen",
    survival: bool = False,
) -> None:
    wages_by_group = collect_wages_from_sim(sim, burn_in=burn_in, which=which)
    cdf_by_group = build_cdf_by_group(wages_by_group)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    period_label = f"periods t >= {burn_in}"
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, (x, y) in cdf_by_group.items():
        if x.size == 0:
            continue
        if survival:
            y_plot = 1.0 - y
            ylab = r"P(wage > $w$)"
            title = f"Wage exceedance by group ({which}, {period_label})"
        else:
            y_plot = y
            ylab = r"P(wage $\leq$ $w$)"
            title = f"Wage CDF by group ({which}, {period_label})"
        n = wages_by_group[g].size
        ax.step(x, y_plot, where="post", label=f"group {g} (n={n})")

    ax.set_xlabel("Wage")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
