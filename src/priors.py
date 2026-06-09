"""Productivity prior visuals: marginal Student-t and Normal–Inverse-Gamma joint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma, norm, normal_inverse_gamma, t

TRUE_MU = 5.0
TRUE_VAR = 1.0
TRUE_SIG_VAR = 0.25  # Workers.sigma_signal**2 default (0.5**2)

# model_timeline.pdf: box fill #e1f5fe, border/arrows #01579b / blue-grey
COLOR_TIMELINE_BLUE = "#01579b"
COLOR_TIMELINE_MID = "#4a90a4"
COLOR_TRUE = "#607d8b"
COLOR_MEAN = "#c62828"
COLOR_VAR = "#ef6c00"
COLOR_CONTOUR = "#01579b"

# Sized for LaTeX \\includegraphics[width=\\textwidth] (~6.5in); do not oversize figsize
PAGE_WIDTH_IN = 6.5
PAGE_HEIGHT_IN = 9.0

PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}

# Pedagogical prior sets for dissertation figures (label, mu, nu, alpha, beta)
DISSERTATION_EXAMPLES: list[tuple[str, float, float, float, float]] = [
    ("Symmetric initial prior", 5.0, 1.0, 2.0, 2.0),
    ("Pessimistic mean ($\\mu_0=1$)", 1.0, 1.0, 2.0, 4.0),
    ("Confident majority ($\\nu_0$ large)", 5.0, 500.0, 100.0, 2.0),
    ("Uncertain variance ($\\beta_0$ large)", 5.0, 1.0, 2.0, 100.0),
    ("Posterior after learning", 4.2, 36.0, 19.5, 14.32),
]

# Defaults from src/ebd.py (Firms.mu_0, nu_0, alpha_0, beta_0)
DEFAULT_PRODUCTIVITY_GROUP_PRIORS: list[tuple[str, float, float, float, float]] = [
    ("group 0 default prior", 5.0, 1.0, 2.0, 2.0),
    ("group 1 default prior", 5.0, 1.0, 2.0, 4.0),
]

# Signal-noise variance: Inv-Gamma(delta, kappa); E[sigma_s^2] = kappa / (delta - 1)
DISSERTATION_SIGNAL_EXAMPLES: list[tuple[str, float, float]] = [
    ("Symmetric initial prior", 2.0, 0.5),
    ("Confident signal ($\\delta_0$ large)", 50.0, 12.25),
    ("Noisy signal ($\\kappa_0$ large)", 2.0, 10.0),
    ("Minority-style ($\\delta$ small, $\\kappa$ large)", 2.0, 10.0),
]

# Defaults from src/ebd.py (Firms.delta_0, kappa_0)
DEFAULT_SIGNAL_GROUP_PRIORS: list[tuple[str, float, float]] = [
    ("group 0 default signal prior", 2.0, 2.0),
    ("group 1 default signal prior", 2.0, 2.0),
]


def signal_var_mean(delta: float, kappa: float) -> float:
    """Posterior mean of signal variance; matches ebd sig_var = kappa / (delta - 1)."""
    return kappa / (delta - 1.0)


def x_grid_signal_var(
    true_sig_var: float = TRUE_SIG_VAR,
    var_hi: float | None = None,
    n_grid: int = 120,
) -> np.ndarray:
    hi = 3.0 * true_sig_var if var_hi is None else var_hi
    return np.linspace(0.02, max(hi, 0.5), n_grid)


def x_grid_signal_error(
    true_sig_var: float = TRUE_SIG_VAR,
    t_scale: float = 1.0,
    n_grid: int = 120,
) -> np.ndarray:
    half = max(3.0 * np.sqrt(true_sig_var), 4.0 * t_scale)
    return np.linspace(-half, half, n_grid)


def productivity_t_params(
    mu: float, eta: float, alpha: float, beta: float
) -> tuple[float, float, float]:
    """Degrees of freedom, location, and scale for the marginal Student-t."""
    scale2 = beta * (eta + 1.0) / (alpha * eta)
    return 2.0 * alpha, mu, float(np.sqrt(scale2))


def signal_t_params(delta: float, kappa: float) -> tuple[float, float, float]:
    """
    Marginal Student-t for signal error ε = p − s with ε | σ_s² ~ N(0, σ_s²),
    σ_s² ~ Inv-Gamma(δ, κ) — same form as productivity with μ=0, ν=1.
    """
    return productivity_t_params(0.0, 1.0, delta, kappa)


def x_grid_productivity(true_mu: float = TRUE_MU, n_grid: int = 120) -> np.ndarray:
    return np.linspace(min(2.5, true_mu - 2.5), max(7.5, true_mu + 2.5), n_grid)


def _plot_nig_joint(
    ax,
    dist_nig,
    mu: float,
    eta: float,
    alpha: float,
    beta: float,
    *,
    true_mu: float,
    true_var: float,
    mu_axis: np.ndarray,
    var_axis: np.ndarray,
    n_contour: int,
    show_ref_labels: bool,
) -> None:
    """NIG joint contour (line contours only)."""
    m_mesh, v_mesh = np.meshgrid(mu_axis, var_axis)
    log_pdf = np.log(np.maximum(dist_nig.pdf(m_mesh, v_mesh), 1e-300))
    levels = np.linspace(
        np.percentile(log_pdf, 8), np.percentile(log_pdf, 92), n_contour
    )
    ax.contour(
        m_mesh,
        v_mesh,
        log_pdf,
        levels=levels,
        colors=COLOR_CONTOUR,
        linewidths=0.55,
        alpha=0.85,
    )
    ax.axvline(true_mu, color=COLOR_MEAN, ls=":", lw=0.8, alpha=0.75)
    ax.axhline(true_var, color=COLOR_VAR, ls=":", lw=0.8, alpha=0.75)
    ax.text(
        0.02,
        0.97,
        rf"$\mathcal{{NIG}}$: $\mu={mu:g}$, $\nu={eta:g}$, $\alpha={alpha:g}$, $\beta={beta:g}$",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
    )
    if show_ref_labels:
        var_lo, var_hi = var_axis[0], var_axis[-1]
        ax.text(
            true_mu,
            var_lo + 0.08 * (var_hi - var_lo),
            rf"$\mu={true_mu:g}$",
            color=COLOR_MEAN,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
        )
        ax.text(
            mu_axis[-1],
            true_var,
            rf"$\sigma^2={true_var:g}$",
            color=COLOR_VAR,
            va="bottom",
            ha="right",
            fontsize=8,
        )
    ax.grid(True)


def _plot_prior_pair(
    ax_t,
    ax_nig,
    mu: float,
    eta: float,
    alpha: float,
    beta: float,
    *,
    label: str,
    true_mu: float,
    true_var: float,
    x_grid: np.ndarray,
    n_grid: int,
    n_contour: int,
    show_legend: bool,
    show_ylabel: bool,
    show_ref_labels: bool,
) -> None:
    true_sd = np.sqrt(true_var)
    nu_t, _, t_scale = productivity_t_params(mu, eta, alpha, beta)
    dist_nig = normal_inverse_gamma(mu, eta, alpha, beta)
    _, mean_s2 = dist_nig.mean()
    var_mid = np.nan_to_num(mean_s2, nan=beta / (alpha + 1))

    mu_axis = np.linspace(min(mu, true_mu) - 1.0, max(mu, true_mu) + 1.0, n_grid)
    var_lo = max(0.05, 0.4 * min(true_var, var_mid))
    var_hi = max(1.6 * true_var, 1.6 * var_mid, var_lo + 0.4)
    var_axis = np.linspace(var_lo, var_hi, n_grid)

    pdf_t = t.pdf(x_grid, nu_t, loc=mu, scale=t_scale)
    pdf_true = norm.pdf(x_grid, loc=true_mu, scale=true_sd)

    ax_t.plot(x_grid, pdf_t, color=COLOR_TIMELINE_BLUE, lw=1.5, label="marginal $t$")
    ax_t.plot(
        x_grid,
        pdf_true,
        color=COLOR_TRUE,
        lw=1.2,
        ls="--",
        label=rf"true $\mathcal{{N}}({true_mu},{true_sd:g})$",
    )
    ax_t.axvline(true_mu, color=COLOR_MEAN, ls=":", lw=0.8, alpha=0.75)
    ax_t.set_title(label, loc="left", pad=2)
    ax_t.grid(True)
    if show_ylabel:
        ax_t.set_ylabel("density")
    if show_legend:
        ax_t.legend(loc="upper right", framealpha=0.9, borderpad=0.4)

    _plot_nig_joint(
        ax_nig,
        dist_nig,
        mu,
        eta,
        alpha,
        beta,
        true_mu=true_mu,
        true_var=true_var,
        mu_axis=mu_axis,
        var_axis=var_axis,
        n_contour=n_contour,
        show_ref_labels=show_ref_labels,
    )


def prod(
    mu: float | None = None,
    eta: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    path: str | Path = "figs/productivity_priors.pdf",
    *,
    n_examples: int = 4,
    examples: list[tuple[str, float, float, float, float]] | None = None,
    true_mu: float = TRUE_MU,
    true_var: float = TRUE_VAR,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    n_grid: int = 120,
    n_contour: int = 16,
) -> Path:
    """
    Plot marginal Student-t and NIG joint for one or more prior specifications.

    Single prior (backward compatible):
        prod(5.0, 1.0, 2.0, 2.0, "figs/one.pdf")

    Dissertation panel (default four pedagogical examples):
        prod(path="figs/productivity_priors_panel.pdf", n_examples=4)

    Custom list (label, mu, nu, alpha, beta):
        prod(examples=[("Mine", 5, 1, 3, 2)], path="figs/custom.pdf")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mu is not None:
        rows = [("Prior", mu, eta, alpha, beta)]
    else:
        pool = examples if examples is not None else DISSERTATION_EXAMPLES
        rows = pool[: max(1, n_examples)]

    n = len(rows)
    if figsize is None:
        figsize = (PAGE_WIDTH_IN, PAGE_HEIGHT_IN) if n > 1 else (PAGE_WIDTH_IN, 2.6)

    x_grid = x_grid_productivity(true_mu, n_grid)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            n,
            2,
            figsize=figsize,
            squeeze=False,
            layout="constrained",
        )

        for i, (label, m, e, a, b) in enumerate(rows):
            _plot_prior_pair(
                axes[i, 0],
                axes[i, 1],
                m,
                e,
                a,
                b,
                label=label,
                true_mu=true_mu,
                true_var=true_var,
                x_grid=x_grid,
                n_grid=n_grid,
                n_contour=n_contour,
                show_legend=(i == 0),
                show_ylabel=(i == n // 2),
                show_ref_labels=(i == n - 1),
            )

        axes[-1, 0].set_xlabel(r"productivity $y$")
        axes[-1, 1].set_xlabel(r"mean $\mu$")
        for j in range(n - 1):
            axes[j, 0].tick_params(labelbottom=False)
            axes[j, 1].tick_params(labelbottom=False)
        axes[n // 2, 0].set_ylabel("density")
        axes[n // 2, 1].set_ylabel(r"variance $\sigma^2$")

        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    return path


def prod_default_groups(
    path: str | Path = "figs/productivity_priors_defaults_by_group.png",
    *,
    true_mu: float = TRUE_MU,
    true_var: float = TRUE_VAR,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    n_grid: int = 120,
    n_contour: int = 16,
) -> Path:
    """
    Plot the initial productivity priors for group 0 and group 1 from ebd.py defaults.
    """
    return prod(
        path=path,
        examples=DEFAULT_PRODUCTIVITY_GROUP_PRIORS,
        n_examples=len(DEFAULT_PRODUCTIVITY_GROUP_PRIORS),
        true_mu=true_mu,
        true_var=true_var,
        figsize=figsize,
        dpi=dpi,
        n_grid=n_grid,
        n_contour=n_contour,
    )


def priors_quadrants_groups_ij(
    path: str | Path = "figs/priors_quadrants_groups_ij.png",
    *,
    true_mu: float = TRUE_MU,
    true_var: float = TRUE_VAR,
    true_sig_var: float = TRUE_SIG_VAR,
    figsize: tuple[float, float] = (PAGE_WIDTH_IN, 10.5),
    dpi: int = 200,
    n_grid: int = 120,
    n_contour: int = 16,
) -> Path:
    """
    Build a stacked 2x2 + 2x2 layout:
      group i (top):    [prod t, prod NIG]
                        [signal t, signal Inv-Gamma]
      group j (bottom): [prod t, prod NIG]
                        [signal t, signal Inv-Gamma]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    prod_rows = DEFAULT_PRODUCTIVITY_GROUP_PRIORS
    sig_rows = DEFAULT_SIGNAL_GROUP_PRIORS
    group_labels = ["group i", "group j"]

    x_grid_prod = x_grid_productivity(true_mu, n_grid)
    max_sig_scale = max(signal_t_params(d, k)[2] for _, d, k in sig_rows)
    x_grid_sig = x_grid_signal_error(true_sig_var, max_sig_scale, n_grid)
    true_prod_sd = np.sqrt(true_var)
    true_sig_sd = np.sqrt(true_sig_var)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            4, 2, figsize=figsize, squeeze=False, layout="constrained"
        )

        for g in range(2):
            block_top = 2 * g
            grp = group_labels[g]
            _, mu, eta, alpha, beta = prod_rows[g]
            _, delta, kappa = sig_rows[g]

            # Productivity: Student-t + NIG
            _plot_prior_pair(
                axes[block_top, 0],
                axes[block_top, 1],
                mu,
                eta,
                alpha,
                beta,
                label=f"{grp}: productivity marginal $t$",
                true_mu=true_mu,
                true_var=true_var,
                x_grid=x_grid_prod,
                n_grid=n_grid,
                n_contour=n_contour,
                show_legend=(g == 0),
                show_ylabel=True,
                show_ref_labels=(g == 1),
            )
            axes[block_top, 1].set_title(
                f"{grp}: productivity prior $\\mathcal{{NIG}}$", loc="left", pad=2
            )

            # Signal: Student-t + Inv-Gamma
            nu_sig, _, scale_sig = signal_t_params(delta, kappa)
            pdf_sig_t = t.pdf(x_grid_sig, nu_sig, loc=0.0, scale=scale_sig)
            pdf_sig_true = norm.pdf(x_grid_sig, loc=0.0, scale=true_sig_sd)

            ax_st = axes[block_top + 1, 0]
            ax_st.plot(
                x_grid_sig,
                pdf_sig_t,
                color=COLOR_TIMELINE_BLUE,
                lw=1.5,
                label="marginal $t$",
            )
            ax_st.plot(
                x_grid_sig,
                pdf_sig_true,
                color=COLOR_TRUE,
                lw=1.2,
                ls="--",
                label=rf"true $\mathcal{{N}}(0,{true_sig_sd:g})$",
            )
            ax_st.axvline(0.0, color=COLOR_MEAN, ls=":", lw=0.8, alpha=0.75)
            ax_st.set_title(f"{grp}: signal-error marginal $t$", loc="left", pad=2)
            ax_st.set_ylabel("density")
            ax_st.grid(True)
            if g == 0:
                ax_st.legend(loc="upper right", framealpha=0.9, borderpad=0.4)

            sig_mean = signal_var_mean(delta, kappa)
            sig_v_grid = x_grid_signal_var(
                true_sig_var,
                var_hi=max(1.4 * sig_mean, 1.4 * true_sig_var),
                n_grid=n_grid,
            )
            ax_sv = axes[block_top + 1, 1]
            ax_sv.plot(
                sig_v_grid,
                invgamma(delta, scale=kappa).pdf(sig_v_grid),
                color=COLOR_TIMELINE_BLUE,
                lw=1.5,
            )
            ax_sv.axvline(true_sig_var, color=COLOR_VAR, ls="--", lw=1.0, alpha=0.85)
            ax_sv.axvline(
                sig_mean, color=COLOR_TIMELINE_MID, ls=":", lw=0.9, alpha=0.85
            )
            ax_sv.text(
                0.02,
                0.97,
                rf"$\delta={delta:g}$, $\kappa={kappa:g}$",
                transform=ax_sv.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )
            ax_sv.set_title(
                f"{grp}: signal-variance prior Inv-Gamma", loc="left", pad=2
            )
            ax_sv.set_ylabel("density")
            ax_sv.grid(True)

        for r in range(3):
            axes[r, 0].tick_params(labelbottom=False)
            axes[r, 1].tick_params(labelbottom=False)
        axes[-1, 0].set_xlabel(r"signal error $\varepsilon = p - s$")
        axes[-1, 1].set_xlabel(r"signal variance $\sigma_s^2$")

        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    return path


def _plot_signal_var_row(
    ax,
    delta: float,
    kappa: float,
    *,
    label: str,
    true_sig_var: float,
    v_grid: np.ndarray,
    show_legend: bool,
    show_ylabel: bool,
    show_ref_label: bool,
) -> None:
    dist = invgamma(delta, scale=kappa)
    pdf_ig = dist.pdf(v_grid)
    mean_v = signal_var_mean(delta, kappa)
    ax.plot(v_grid, pdf_ig, color=COLOR_TIMELINE_BLUE, lw=1.5, label=r"Inv-Gamma prior")
    ax.axvline(
        true_sig_var,
        color=COLOR_VAR,
        ls="--",
        lw=1.0,
        alpha=0.85,
        label="true signal variance",
    )
    ax.axvline(
        mean_v,
        color=COLOR_TIMELINE_MID,
        ls=":",
        lw=0.9,
        alpha=0.85,
        label=r"mean $\kappa/(\delta-1)$",
    )
    ax.set_title(label, loc="left", pad=2)
    ax.text(
        0.98,
        0.97,
        rf"$\delta={delta:g}$, $\kappa={kappa:g}$",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
    )
    if show_ref_label:
        ax.text(
            true_sig_var,
            0.92 * ax.get_ylim()[1],
            rf"$\sigma_s^2={true_sig_var:g}$",
            color=COLOR_VAR,
            ha="center",
            fontsize=8,
        )
    ax.grid(True)
    if show_ylabel:
        ax.set_ylabel("density")

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9, borderpad=0.4)


def signal_var(
    delta: float | None = None,
    kappa: float | None = None,
    path: str | Path = "figs/signal_variance_priors.pdf",
    *,
    n_examples: int = 4,
    examples: list[tuple[str, float, float]] | None = None,
    true_sig_var: float = TRUE_SIG_VAR,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    n_grid: int = 120,
) -> Path:
    """
    Plot inverse-gamma priors on signal variance (ebd: delta, kappa).

        signal_var(2.0, 2.0, "figs/one.pdf")
        signal_var(path="figs/signal_variance_panel.pdf", n_examples=4)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if delta is not None:
        rows = [("Prior", delta, kappa)]
    else:
        pool = examples if examples is not None else DISSERTATION_SIGNAL_EXAMPLES
        rows = pool[: max(1, n_examples)]

    n = len(rows)
    row_h = 2.0
    if figsize is None:
        figsize = (PAGE_WIDTH_IN, row_h * n) if n > 1 else (PAGE_WIDTH_IN, 2.4)

    means = [signal_var_mean(d, k) for _, d, k in rows]
    v_grid = x_grid_signal_var(true_sig_var, var_hi=1.4 * max(means + [true_sig_var]))

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            n, 1, figsize=figsize, squeeze=False, layout="constrained"
        )
        axes = axes[:, 0]

        for i, (label, d, k) in enumerate(rows):
            _plot_signal_var_row(
                axes[i],
                d,
                k,
                label=label,
                true_sig_var=true_sig_var,
                v_grid=v_grid,
                show_legend=(i == 0),
                show_ylabel=(i == n // 2),
                show_ref_label=(i == n - 1),
            )

        axes[-1].set_xlabel(r"signal variance $\sigma_s^2$")
        for j in range(n - 1):
            axes[j].tick_params(labelbottom=False)

        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    return path


# Outcome / simulation figures (firm_outcomes, market_outcomes, ebd)
DISSERTATION_DPI = 200
FIGSIZE_TIMESERIES = (PAGE_WIDTH_IN, 3.4)
FIGSIZE_CDF = (PAGE_WIDTH_IN, 3.2)

GROUP_LINE_COLORS = (
    COLOR_TIMELINE_BLUE,
    COLOR_TIMELINE_MID,
    COLOR_VAR,
    COLOR_MEAN,
    COLOR_TRUE,
)


def plot_rc_context():
    """Matplotlib rc context matching dissertation prior figures."""
    return plt.rc_context(PLOT_RC)


def save_dissertation_figure(
    fig: plt.Figure, path: str | Path, *, dpi: int = DISSERTATION_DPI
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True)


def dissertation_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=2)


def group_line_color(group_index: int) -> str:
    return GROUP_LINE_COLORS[group_index % len(GROUP_LINE_COLORS)]


# Four-scenario pipeline (paired baseline vs policy, unbiased and biased)
SCENARIO_IDS = ("baseline", "policy", "baseline_bias", "policy_bias")
SCENARIO_PAIRS = (
    ("unbiased", ("baseline", "policy")),
    ("biased", ("baseline_bias", "policy_bias")),
)
PANEL_TITLES = {
    "unbiased": "Unbiased signals: baseline vs policy",
    "biased": "Downward-biased signals: baseline vs policy",
}
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "policy": "Policy",
    "baseline_bias": "Baseline",
    "policy_bias": "Policy",
}
SCENARIO_TEX_LABELS = {
    "baseline": "Baseline (myopic)",
    "policy": "Policy (BayesUCB)",
    "baseline_bias": "Baseline (myopic)",
    "policy_bias": "Policy (BayesUCB)",
}
# Light (g0) / dark (g1) per scenario — blue, green, red, purple families
SCENARIO_GROUP_COLORS = {
    "baseline": ("#90caf9", "#1565c0"),
    "policy": ("#a5d6a7", "#2e7d32"),
    "baseline_bias": ("#ef9a9a", "#c62828"),
    "policy_bias": ("#ce93d8", "#6a1b9a"),
}
SCENARIO_LINE_COLORS = {
    "baseline": COLOR_TRUE,
    "policy": "#2e7d32",
    "baseline_bias": "#c62828",
    "policy_bias": "#6a1b9a",
}
PAIR_BASELINE_KEYS = {
    "unbiased": "baseline",
    "biased": "baseline_bias",
}
PAIR_POLICY_KEYS = {
    "unbiased": "policy",
    "biased": "policy_bias",
}
SCENARIO_FACET_TITLES = {
    "baseline": "Baseline (myopic, unbiased)",
    "policy": "Policy (BayesUCB, unbiased)",
    "baseline_bias": "Baseline (myopic, biased)",
    "policy_bias": "Policy (BayesUCB, biased)",
}


def scenario_group_color(scenario_id: str, group_index: int) -> str:
    colors = SCENARIO_GROUP_COLORS[scenario_id]
    return colors[group_index % len(colors)]


def legend_dissertation(ax: plt.Axes, **kwargs) -> None:
    defaults = {"loc": "upper right", "framealpha": 0.9, "borderpad": 0.4}
    defaults.update(kwargs)
    ax.legend(**defaults)


if __name__ == "__main__":
    prod(path="figs/productivity_priors_panel.png", n_examples=4)
    prod_default_groups(
        figsize=(6.5, 5.0), path="figs/productivity_priors_defaults_by_group.png"
    )
    priors_quadrants_groups_ij(path="figs/priors_quadrants_groups_ij.png")
    signal_var(path="figs/signal_variance_priors_panel.png", n_examples=4)
