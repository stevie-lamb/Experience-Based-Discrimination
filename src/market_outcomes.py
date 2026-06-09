"""Market-level outcomes from a completed Simulation (employment, wages, KS, reports)."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.priors import (
    FIGSIZE_CDF,
    PANEL_TITLES,
    SCENARIO_FACET_TITLES,
    SCENARIO_IDS,
    SCENARIO_LABELS,
    SCENARIO_PAIRS,
    SCENARIO_TEX_LABELS,
    dissertation_title,
    group_line_color,
    legend_dissertation,
    plot_rc_context,
    save_dissertation_figure,
    scenario_group_color,
    style_axis,
)

if TYPE_CHECKING:
    from src.ebd import Simulation

DEFAULT_BURN_IN = 100
WAGE_WHICH_CHOICES = ("chosen", "accepted", "all_offers")


def _validate_burn_in(horizon: int, burn_in: int) -> None:
    if horizon <= burn_in:
        raise ValueError(f"horizon ({horizon}) must be > burn_in ({burn_in})")


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
        "kurtosis": (
            float(stats.kurtosis(w, fisher=True, bias=False)) if w.size > 3 else None
        ),
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
                    (
                        float(sim.employment_rate_log[t, g])
                        if np.isfinite(sim.employment_rate_log[t, g])
                        else None
                    )
                    for g in range(sim.ng)
                ]
                for t in range(burn_in, sim.horizon)
            ],
            "unemployment_rate": [
                [int(t)]
                + [
                    (
                        float(sim.unemployment_rate_log[t, g])
                        if np.isfinite(sim.unemployment_rate_log[t, g])
                        else None
                    )
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


def _minority_group_index(sim: Simulation) -> int:
    """Index of the smallest worker pool (minority under unequal shares)."""
    return int(np.argmin(sim.pool_sizes))


def compute_equity_outcomes(sim: Simulation, *, burn_in: int) -> dict:
    """Choice shares, total wages by group, and aggregate welfare (wages + profit)."""
    _validate_burn_in(sim.horizon, burn_in)
    sl = slice(burn_in, sim.horizon)

    chosen = sim.chosen_group_log[sl]
    accepted = sim.accepted_log[sl]
    accepted_wages = sim.accepted_wages[sl]
    profit = sim.profit[sl]

    minority_g = _minority_group_index(sim)
    by_group: dict[str, dict] = {}
    total_wages_by_group: dict[str, float] = {}

    group_mask = chosen[..., np.newaxis] == np.arange(sim.ng)
    profit_by_group = {
        str(g): float(profit[group_mask[..., g]].sum()) for g in range(sim.ng)
    }

    for g in range(sim.ng):
        g_str = str(g)
        choice_share = float((chosen == g).mean())
        wages_received = float(accepted_wages[accepted & (chosen == g)].sum())
        profit_received = profit_by_group[g_str]
        total_wages_by_group[g_str] = wages_received
        by_group[g_str] = {
            "choice_share": choice_share,
            "total_wages_received": wages_received,
            "total_profit_received": profit_received,
            "total_welfare": wages_received + profit_received,
        }

    total_wages = float(sum(total_wages_by_group.values()))
    total_profit = float(profit.sum())
    total_welfare = total_wages + total_profit

    for g in range(sim.ng):
        g_str = str(g)
        by_group[g_str]["wage_share_of_total"] = (
            float(total_wages_by_group[g_str] / total_wages)
            if total_wages > 0
            else None
        )

    return {
        "minority_group": minority_g,
        "minority_choice_share": by_group[str(minority_g)]["choice_share"],
        "by_group": by_group,
        "totals": {
            "total_wages": total_wages,
            "total_profit": total_profit,
            "total_welfare": total_welfare,
        },
    }


def compute_equity_deltas(equity_baseline: dict, equity_policy: dict) -> dict:
    """Policy minus baseline for equity metrics."""
    ng = len(equity_baseline["by_group"])
    by_group: dict[str, dict] = {}
    for g in range(ng):
        g_str = str(g)
        base_g = equity_baseline["by_group"][g_str]
        pol_g = equity_policy["by_group"][g_str]
        by_group[g_str] = {
            "choice_share": pol_g["choice_share"] - base_g["choice_share"],
            "total_wages_received": (
                pol_g["total_wages_received"] - base_g["total_wages_received"]
            ),
            "total_profit_received": (
                pol_g["total_profit_received"] - base_g["total_profit_received"]
            ),
            "total_welfare": pol_g["total_welfare"] - base_g["total_welfare"],
        }

    base_tot = equity_baseline["totals"]
    pol_tot = equity_policy["totals"]
    return {
        "minority_choice_share": (
            equity_policy["minority_choice_share"]
            - equity_baseline["minority_choice_share"]
        ),
        "by_group": by_group,
        "totals": {
            key: pol_tot[key] - base_tot[key]
            for key in ("total_wages", "total_profit", "total_welfare")
        },
    }


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
            "signal_bias_g1": float(sim.workers.signal_bias_g1),
        },
        "employment": compute_employment_outcomes(
            sim,
            burn_in=burn_in,
            include_time_series=include_time_series,
        ),
        "wages": compute_wage_outcomes(sim, burn_in=burn_in, which=wage_which),
        "equity": compute_equity_outcomes(sim, burn_in=burn_in),
    }


def compute_market_comparison(
    sim_baseline: Simulation,
    sim_policy: Simulation,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    wage_which: str = "chosen",
    include_time_series: bool = True,
    base_seed: int | None = None,
) -> dict:
    """Market outcomes for a baseline and policy run, packaged for paired tables."""
    if sim_baseline.ng != sim_policy.ng:
        raise ValueError("baseline and policy simulations must share n_groups")

    base_out = compute_market_outcomes(
        sim_baseline,
        burn_in=burn_in,
        wage_which=wage_which,
        include_time_series=include_time_series,
        base_seed=base_seed,
    )
    policy_out = compute_market_outcomes(
        sim_policy,
        burn_in=burn_in,
        wage_which=wage_which,
        include_time_series=include_time_series,
        base_seed=base_seed,
    )

    return {
        "metadata": {
            "horizon": sim_policy.horizon,
            "burn_in": burn_in,
            "n_periods_measured": sim_policy.horizon - burn_in,
            "wage_which": wage_which,
            "n_groups": sim_policy.ng,
            "base_seed": base_seed,
            "signal_bias_g1": float(sim_policy.workers.signal_bias_g1),
        },
        "baseline": base_out,
        "policy": policy_out,
        "equity_deltas": compute_equity_deltas(
            base_out["equity"], policy_out["equity"]
        ),
    }


def compute_scenario_comparison(
    scenarios: dict[str, Simulation],
    *,
    burn_in: int = DEFAULT_BURN_IN,
    wage_which: str = "chosen",
    include_time_series: bool = True,
    base_seed: int | None = None,
    signal_bias_g1: float | None = None,
) -> dict:
    """Market outcomes for baseline, policy, and policy+signal-bias scenarios."""
    missing = [sid for sid in SCENARIO_IDS if sid not in scenarios]
    if missing:
        raise ValueError(f"missing scenario keys: {missing}")

    ref = scenarios[SCENARIO_IDS[0]]
    scenario_out: dict[str, dict] = {}
    for sid in SCENARIO_IDS:
        sim = scenarios[sid]
        if sim.ng != ref.ng:
            raise ValueError("all scenarios must share n_groups")
        scenario_out[sid] = compute_market_outcomes(
            sim,
            burn_in=burn_in,
            wage_which=wage_which,
            include_time_series=include_time_series,
            base_seed=base_seed,
        )

    bias_g1 = signal_bias_g1
    if bias_g1 is None:
        bias_g1 = float(scenarios["policy_bias"].workers.signal_bias_g1)

    return {
        "metadata": {
            "horizon": ref.horizon,
            "burn_in": burn_in,
            "n_periods_measured": ref.horizon - burn_in,
            "wage_which": wage_which,
            "n_groups": ref.ng,
            "base_seed": base_seed,
            "signal_bias_g1": bias_g1,
            "group_shares_target": ref.workers.group_shares.tolist(),
            "pairs": {
                pair_id: {"baseline": base_key, "policy": pol_key}
                for pair_id, (base_key, pol_key) in SCENARIO_PAIRS
            },
        },
        "scenarios": scenario_out,
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


def _fmt_tex(x, nd: int = 2) -> str:
    if x is None:
        return "---"
    return f"{x:.{nd}f}"


def _fmt_tex_int(x, nd: int = 0) -> str:
    if x is None:
        return "---"
    return f"{float(x):,.{nd}f}"


def _fmt_tex_pct(x, nd: int = 1) -> str:
    if x is None:
        return "---"
    return f"{100.0 * float(x):.{nd}f}\\%"


def _ks_pvalue_str(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else _fmt_tex(p)


def _employment_gap_final(run: dict) -> float | None:
    """Group 0 minus group 1 final-period employment rate."""
    er0 = run["employment"]["by_group"]["0"]["employment_rate"]["final"]
    er1 = run["employment"]["by_group"]["1"]["employment_rate"]["final"]
    if er0 is None or er1 is None:
        return None
    return float(er0) - float(er1)


def write_scenario_comparison_tex(out: dict, path: str | Path) -> None:
    """Write paired market tables: unbiased then biased, Baseline | Policy each."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = out["metadata"]
    scenarios = out["scenarios"]
    fmt = _fmt_tex
    pair_section_titles = {
        "unbiased": "Unbiased signals",
        "biased": "Downward-biased signals",
    }

    lines = [
        "% Auto-generated market outcomes (four scenarios, paired); "
        "\\input{results_910/market_outcomes.tex}",
        f"% horizon={meta['horizon']}, burn_in={meta['burn_in']}, "
        f"wage_which={meta['wage_which']}",
        "",
        "\\subsection*{Market outcomes (post burn-in)}",
        f"Measurements use periods $t \\geq {meta['burn_in']}$ "
        f"({meta['n_periods_measured']} periods). "
        f"Wages pooled with \\texttt{{{meta['wage_which']}}}.",
        "",
    ]

    for pair_id, (base_key, pol_key) in SCENARIO_PAIRS:
        section = pair_section_titles[pair_id]
        base_run = scenarios[base_key]
        pol_run = scenarios[pol_key]
        minority_g = base_run["equity"]["minority_group"]

        lines += [
            f"\\subsubsection*{{{section}}}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Employment-rate gap ($e_0 - e_1$, final period) --- {section.lower()}}}",
            "\\begin{tabular}{|l|r|r|}",
            "\\hline",
            " & Baseline & Policy \\\\",
            "\\hline",
            f"$e_0 - e_1$ & {fmt(_employment_gap_final(base_run))} "
            f"& {fmt(_employment_gap_final(pol_run))} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Wage distribution moments by group --- {section.lower()}}}",
            "\\begin{tabular}{|l|l|r|r|r|r|r|}",
            "\\hline",
            "Regime & Group & $n$ & Mean & Variance & Skewness & Kurtosis \\\\",
            "\\hline",
        ]

        for label, run in (("Baseline", base_run), ("Policy", pol_run)):
            for i in range(meta["n_groups"]):
                w = run["wages"]["by_group"][str(i)]
                regime_cell = label if i == 0 else ""
                lines.append(
                    f"{regime_cell} & {i} & {w['n']} & {fmt(w['mean'])} "
                    f"& {fmt(w['variance'])} & {fmt(w['skewness'])} "
                    f"& {fmt(w['kurtosis'])} \\\\"
                )
            lines.append("\\hline")
        if lines[-1] == "\\hline":
            lines.pop()

        lines += [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Pairwise Kolmogorov--Smirnov tests on wages --- {section.lower()}}}",
            "\\begin{tabular}{|l|r|r|r|r|}",
            "\\hline",
            "Pair & $D$ (baseline) & $D$ (policy) "
            "& $p$ (baseline) & $p$ (policy) \\\\",
            "\\hline",
        ]

        base_ks = base_run["wages"]["ks_pairwise"]
        pol_ks = pol_run["wages"]["ks_pairwise"]
        for key in base_ks:
            pair_label = key.replace("_", " ")

            def _d(ks: dict) -> str:
                if not ks or ks.get("insufficient_data"):
                    return "---"
                return fmt(ks["statistic"])

            def _p(ks: dict) -> str:
                if not ks or ks.get("insufficient_data"):
                    return "---"
                return _ks_pvalue_str(ks["pvalue"])

            lines.append(
                f"{pair_label} & {_d(base_ks.get(key, {}))} & {_d(pol_ks.get(key, {}))} "
                f"& {_p(base_ks.get(key, {}))} & {_p(pol_ks.get(key, {}))} \\\\"
            )

        lines += [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Minority choice share --- {section.lower()}}}",
            "\\begin{tabular}{|l|r|r|}",
            "\\hline",
            "Metric & Baseline & Policy \\\\",
            "\\hline",
            f"Minority choice share (group {minority_g}) "
            f"& {_fmt_tex_pct(base_run['equity']['minority_choice_share'])} "
            f"& {_fmt_tex_pct(pol_run['equity']['minority_choice_share'])} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Policy effect on group welfare ($\\Delta$ = policy $-$ baseline, "
            f"{section.lower()})}}",
            "\\begin{tabular}{|l|r|r|r|}",
            "\\hline",
            "Group & $\\Delta$ wages & $\\Delta$ profit & $\\Delta$ welfare \\\\",
            "\\hline",
        ]

        deltas = compute_equity_deltas(base_run["equity"], pol_run["equity"])
        for g in range(meta["n_groups"]):
            delta_g = deltas["by_group"][str(g)]
            label = f"Group {g}"
            if g == minority_g:
                label += " (minority)"
            lines.append(
                f"{label} & "
                f"{_fmt_tex_int(delta_g['total_wages_received'])} & "
                f"{_fmt_tex_int(delta_g['total_profit_received'])} & "
                f"{_fmt_tex_int(delta_g['total_welfare'])} \\\\"
            )

        lines += [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_market_comparison_tex(out: dict, path: str | Path) -> None:
    """Write paired baseline-vs-policy market tables (employment, wages, KS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = out["metadata"]
    ng = meta["n_groups"]
    base = out["baseline"]
    policy = out["policy"]
    fmt = _fmt_tex

    lines = [
        "% Auto-generated market outcomes (baseline vs policy); "
        "\\input{results/market_outcomes.tex}",
        f"% horizon={meta['horizon']}, burn_in={meta['burn_in']}, "
        f"wage_which={meta['wage_which']}",
        "",
        "\\subsection*{Market outcomes (post burn-in)}",
        f"Measurements use periods $t \\geq {meta['burn_in']}$ "
        f"({meta['n_periods_measured']} periods). "
        f"Wages pooled with \\texttt{{{meta['wage_which']}}}. "
        "Each metric is shown for the baseline (myopic) and policy (BayesUCB) runs.",
        "",
    ]

    # Final-period employment: difference (policy minus baseline) by group.
    emp_diff_cells = []
    for i in range(ng):
        pol_f = policy["employment"]["by_group"][str(i)]["employment_rate"]["final"]
        base_f = base["employment"]["by_group"][str(i)]["employment_rate"]["final"]
        diff = None if pol_f is None or base_f is None else pol_f - base_f
        emp_diff_cells.append(fmt(diff))
    emp_group_header = " & ".join(f"Group {i}" for i in range(ng))

    lines += [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Final-period employment rate by group (policy $-$ baseline)}",
        "\\begin{tabular}{|l|" + "c|" * ng + "}",
        "\\hline",
        f" & {emp_group_header} \\\\",
        "\\hline",
        f"$\\Delta e_g$ (policy $-$ baseline) & {' & '.join(emp_diff_cells)} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wage distribution moments by group}",
        "\\begin{tabular}{|l|l|r|r|r|r|r|}",
        "\\hline",
        "Regime & Group & $n$ & Mean & Variance & Skewness & Kurtosis \\\\",
        "\\hline",
    ]

    for label, run in (("Policy", policy), ("Baseline", base)):
        for i in range(ng):
            w = run["wages"]["by_group"][str(i)]
            regime_cell = label if i == 0 else ""
            lines.append(
                f"{regime_cell} & {i} & {w['n']} & {fmt(w['mean'])} "
                f"& {fmt(w['variance'])} & {fmt(w['skewness'])} "
                f"& {fmt(w['kurtosis'])} \\\\"
            )
        lines.append("\\hline")
    if lines[-1] == "\\hline":
        lines.pop()

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Pairwise Kolmogorov--Smirnov tests on wages}",
        "\\begin{tabular}{|l|r|r|r|r|}",
        "\\hline",
        "Pair & $D$ (baseline) & $D$ (policy) " "& $p$ (baseline) & $p$ (policy) \\\\",
        "\\hline",
    ]

    base_ks = base["wages"]["ks_pairwise"]
    policy_ks = policy["wages"]["ks_pairwise"]
    for key in base_ks:
        label = key.replace("_", " ")
        bks = base_ks.get(key, {})
        pks = policy_ks.get(key, {})

        def _d(ks: dict) -> str:
            if not ks or ks.get("insufficient_data"):
                return "---"
            return fmt(ks["statistic"])

        def _p(ks: dict) -> str:
            if not ks or ks.get("insufficient_data"):
                return "---"
            return _ks_pvalue_str(ks["pvalue"])

        lines.append(f"{label} & {_d(bks)} & {_d(pks)} & {_p(bks)} & {_p(pks)} \\\\")

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]

    base_eq = base["equity"]
    pol_eq = policy["equity"]
    deltas = out.get("equity_deltas") or compute_equity_deltas(base_eq, pol_eq)
    minority_g = base_eq["minority_group"]

    lines += [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Minority choice share}",
        "\\begin{tabular}{|l|r|r|}",
        "\\hline",
        "Metric & Baseline & Policy \\\\",
        "\\hline",
        f"Minority choice share (group {minority_g}) "
        f"& {_fmt_tex_pct(base_eq['minority_choice_share'])} "
        f"& {_fmt_tex_pct(pol_eq['minority_choice_share'])} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Policy effect on group welfare ($\\Delta$ = policy $-$ baseline)}",
        "\\begin{tabular}{|l|r|r|r|}",
        "\\hline",
        "Group & $\\Delta$ wages & $\\Delta$ profit & $\\Delta$ welfare \\\\",
        "\\hline",
    ]

    for g in range(ng):
        delta_g = deltas["by_group"][str(g)]
        label = f"Group {g}"
        if g == minority_g:
            label += " (minority)"
        lines.append(
            f"{label} & "
            f"{_fmt_tex_int(delta_g['total_wages_received'])} & "
            f"{_fmt_tex_int(delta_g['total_profit_received'])} & "
            f"{_fmt_tex_int(delta_g['total_welfare'])} \\\\"
        )

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_market_outcomes_tex(out: dict, path: str | Path) -> None:
    if "scenarios" in out:
        write_scenario_comparison_tex(out, path)
        return
    if "baseline" in out and "policy" in out:
        write_market_comparison_tex(out, path)
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = out["metadata"]
    ng = meta["n_groups"]

    def fmt(x, nd: int = 2) -> str:
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
        "\\begin{tabular}{|l|c|c|c|}",
        "\\hline",
        "Group & $\\bar{e}_g$ & $\\bar{u}_g$ & $e_g$ (final) \\\\",
        "\\hline",
    ]

    for g in range(ng):
        row = out["employment"]["by_group"][str(g)]
        er = row["employment_rate"]
        ur = row["unemployment_rate"]
        lines.append(
            f"{g} & {fmt(er['mean'])} & {fmt(ur['mean'])} & {fmt(er['final'])} \\\\"
        )

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wage distribution moments by group}",
        "\\begin{tabular}{|l|r|r|r|r|r|}",
        "\\hline",
        "Group & $n$ & Mean & Variance & Skewness & Kurtosis \\\\",
        "\\hline",
    ]

    for g in range(ng):
        w = out["wages"]["by_group"][str(g)]
        lines.append(
            f"{g} & {w['n']} & {fmt(w['mean'])} & {fmt(w['variance'])} "
            f"& {fmt(w['skewness'])} & {fmt(w['kurtosis'])} \\\\"
        )

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Pairwise Kolmogorov--Smirnov tests on wages}",
        "\\begin{tabular}{|l|l|r|}",
        "\\hline",
        "Pair & $D$ & $p$-value \\\\",
        "\\hline",
    ]

    for key, ks in out["wages"]["ks_pairwise"].items():
        label = key.replace("_", " ")
        if ks.get("insufficient_data"):
            lines.append(f"{label} & --- & --- \\\\")
        else:
            p = ks["pvalue"]
            p_str = f"{p:.2e}" if p < 0.001 else fmt(p)
            lines.append(f"{label} & {fmt(ks['statistic'])} & {p_str} \\\\")

    lines += [
        "\\hline",
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


def write_market_comparison(
    sim_baseline: Simulation,
    sim_policy: Simulation,
    json_path: str | Path,
    tex_path: str | Path | None = None,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    wage_which: str = "chosen",
    include_time_series: bool = True,
    base_seed: int | None = None,
) -> dict:
    out = compute_market_comparison(
        sim_baseline,
        sim_policy,
        burn_in=burn_in,
        wage_which=wage_which,
        include_time_series=include_time_series,
        base_seed=base_seed,
    )
    write_market_outcomes_json(out, json_path)
    if tex_path is not None:
        write_market_comparison_tex(out, tex_path)
    return out


def write_scenario_comparison(
    scenarios: dict[str, Simulation],
    json_path: str | Path,
    tex_path: str | Path | None = None,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    wage_which: str = "chosen",
    include_time_series: bool = True,
    base_seed: int | None = None,
    signal_bias_g1: float | None = None,
) -> dict:
    out = compute_scenario_comparison(
        scenarios,
        burn_in=burn_in,
        wage_which=wage_which,
        include_time_series=include_time_series,
        base_seed=base_seed,
        signal_bias_g1=signal_bias_g1,
    )
    write_market_outcomes_json(out, json_path)
    if tex_path is not None:
        write_scenario_comparison_tex(out, tex_path)
    return out


def plot_wage_cdf_scenarios(
    scenarios: dict[str, Simulation],
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    which: str = "chosen",
    survival: bool = False,
) -> None:
    """Stacked wage CDFs: unbiased pair (top), biased pair (bottom)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wages = {
        sid: collect_wages_from_sim(scenarios[sid], burn_in=burn_in, which=which)
        for sid in SCENARIO_IDS
    }
    cdfs = {sid: build_cdf_by_group(w) for sid, w in wages.items()}

    if survival:
        ylab = r"$P(w > \mathrm{wage})$"
    else:
        ylab = r"$P(w \leq \mathrm{wage})$"

    def _plot_pair(ax, base_key: str, pol_key: str) -> None:
        for sid in (base_key, pol_key):
            label = SCENARIO_LABELS[sid]
            for g, (x, y) in cdfs[sid].items():
                if x.size == 0:
                    continue
                y_plot = 1.0 - y if survival else y
                n = wages[sid][g].size
                ax.step(
                    x,
                    y_plot,
                    where="post",
                    color=scenario_group_color(sid, g),
                    linewidth=1.5,
                    label=rf"{label} $g={g}$ ($n={n}$)",
                )

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_CDF[0], 2.0 * FIGSIZE_CDF[1]),
            layout="constrained",
            sharex=True,
        )
        for ax, (pair_id, (base_key, pol_key)) in zip(axes, SCENARIO_PAIRS):
            _plot_pair(ax, base_key, pol_key)
            ax.set_ylabel(ylab)
            dissertation_title(ax, PANEL_TITLES[pair_id])
            ax.set_ylim(0, 1)
            style_axis(ax)
            legend_dissertation(ax, fontsize=7)

        axes[-1].set_xlabel(r"wage offer")
        save_dissertation_figure(fig, path)


def plot_wage_cdf_scenarios_faceted(
    scenarios: dict[str, Simulation],
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    which: str = "chosen",
    survival: bool = False,
) -> None:
    """Four-panel wage CDFs: one subplot per scenario (groups within each)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [sid for sid in SCENARIO_IDS if sid not in scenarios]
    if missing:
        raise ValueError(f"missing scenario keys: {missing}")

    wages = {
        sid: collect_wages_from_sim(scenarios[sid], burn_in=burn_in, which=which)
        for sid in SCENARIO_IDS
    }
    cdfs = {sid: build_cdf_by_group(w) for sid, w in wages.items()}

    if survival:
        ylab = r"$P(w > \mathrm{wage})$"
    else:
        ylab = r"$P(w \leq \mathrm{wage})$"

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(FIGSIZE_CDF[0], 2.05 * FIGSIZE_CDF[1]),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        for ax, sid in zip(axes.ravel(), SCENARIO_IDS):
            for g, (x, y) in cdfs[sid].items():
                if x.size == 0:
                    continue
                y_plot = 1.0 - y if survival else y
                n = wages[sid][g].size
                ax.step(
                    x,
                    y_plot,
                    where="post",
                    color=scenario_group_color(sid, g),
                    linewidth=1.5,
                    label=rf"$g={g}$ ($n={n}$)",
                )
            ax.set_title(SCENARIO_FACET_TITLES[sid], loc="left", pad=2, fontsize=9)
            ax.set_ylim(0, 1)
            style_axis(ax)
            legend_dissertation(ax, fontsize=7)

        for ax in axes[:, 0]:
            ax.set_ylabel(ylab)
        for ax in axes[1, :]:
            ax.set_xlabel(r"wage offer")

        save_dissertation_figure(fig, path)


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

    period_note = rf"$t \geq {burn_in}$"
    with plot_rc_context():
        fig, ax = plt.subplots(figsize=FIGSIZE_CDF, layout="constrained")
        for g, (x, y) in cdf_by_group.items():
            if x.size == 0:
                continue
            if survival:
                y_plot = 1.0 - y
                ylab = r"$P(w > \mathrm{wage})$"
                title = rf"Wage exceedance by group ({which}, {period_note})"
            else:
                y_plot = y
                ylab = r"$P(w \leq \mathrm{wage})$"
                title = rf"Wage CDF by group ({which}, {period_note})"
            n = wages_by_group[g].size
            ax.step(
                x,
                y_plot,
                where="post",
                color=group_line_color(g),
                linewidth=1.5,
                label=rf"group {g} ($n={n}$)",
            )

        ax.set_xlabel(r"wage offer")
        ax.set_ylabel(ylab)
        dissertation_title(ax, title)
        ax.set_ylim(0, 1)
        style_axis(ax)
        legend_dissertation(ax)
        save_dissertation_figure(fig, path)


def plot_wage_cdf_pair(
    sim_baseline: Simulation,
    sim_policy: Simulation,
    path: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    which: str = "chosen",
    survival: bool = False,
) -> None:
    """Stacked wage CDFs: policy on the upper panel, baseline on the lower panel.

    Both panels share the x-axis so the baseline-vs-policy gap is directly
    comparable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    runs = [
        ("Policy (BayesUCB)", sim_policy),
        ("Baseline (myopic)", sim_baseline),
    ]
    wages = {
        label: collect_wages_from_sim(sim, burn_in=burn_in, which=which)
        for label, sim in runs
    }
    cdfs = {label: build_cdf_by_group(w) for label, w in wages.items()}

    if survival:
        ylab = r"$P(w > \mathrm{wage})$"
        suptitle = rf"Wage exceedance ({which}, $t \geq {burn_in}$)"
    else:
        ylab = r"$P(w \leq \mathrm{wage})$"
        suptitle = rf"Wage CDF ({which}, $t \geq {burn_in}$)"

    with plot_rc_context():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGSIZE_CDF[0], 2.0 * FIGSIZE_CDF[1]),
            layout="constrained",
            sharex=True,
        )
        for ax, (label, _) in zip(axes, runs):
            cdf_by_group = cdfs[label]
            wages_by_group = wages[label]
            for g, (x, y) in cdf_by_group.items():
                if x.size == 0:
                    continue
                y_plot = 1.0 - y if survival else y
                n = wages_by_group[g].size
                ax.step(
                    x,
                    y_plot,
                    where="post",
                    color=group_line_color(g),
                    linewidth=1.5,
                    label=rf"group {g} ($n={n}$)",
                )
            ax.set_ylabel(ylab)
            dissertation_title(ax, label)
            ax.set_ylim(0, 1)
            style_axis(ax)
            legend_dissertation(ax)

        axes[-1].set_xlabel(r"wage offer")
        fig.suptitle(suptitle)
        save_dissertation_figure(fig, path)
