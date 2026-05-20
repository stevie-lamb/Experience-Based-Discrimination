import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy import stats

rng = np.random.default_rng(123)

@dataclass
class Workers:
    n: int = 2
    n_g: int = 2
    mu_p: float = 5.0
    sigma_p: float = 1.0 # Std, not variance
    mu_res: float = 1.0
    sigma_res: float = 0.5 # Std, not variance
    rho: float = 0.25 # Correlation between Productivity and Reservation wage
    sigma_signal: float = 0.5 # True standard deviation of signal around realised productivity
    group_1_share: float = 0.1  # fraction of workers in group 1 (minority); group 0 is majority

    def __post_init__(self):
        if not 0.0 <= self.group_1_share <= 1.0:
            raise ValueError("group_1_share must be in [0, 1]")

        n1 = int(round(self.n * self.group_1_share))
        n1 = min(max(n1, 0), self.n)
        n0 = self.n - n1
        self.groups = np.concatenate(
            [np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)]
        )
        rng.shuffle(self.groups)

        cov_pr = np.array(
            [
                [self.sigma_p**2, self.rho * self.sigma_p * self.sigma_res],
                [self.rho * self.sigma_p * self.sigma_res, self.sigma_res**2],
            ]
        )
        pr = rng.multivariate_normal(
            mean=[self.mu_p, self.mu_res],
            cov=cov_pr,
            size=self.n,
        )
        self.prod = pr[:, 0]
        self.res = pr[:, 1]
        noise = rng.normal(0.0, self.sigma_signal, size=self.n)
        self.signal = self.prod + noise
        self.wid = np.arange(self.n)

        self.workers_info = {
            i: [self.prod[i], self.res[i], self.signal[i], int(self.groups[i])]
            for i in range(self.n)
        }

    def wage_acceptance(self, chosen_worker, chosen_wage, nf) -> tuple[np.ndarray, np.ndarray]:
        """
        From firm-level offers (chosen_worker, chosen_wage), pick winning firm per worker.
        Returns:
            firm_hired: (F,) bool — firm actually employs its target worker
            worker_employer: (W,) int — employer id or -1 if unemployed
        """
        firm_hired = np.zeros(nf, dtype=bool)
        worker_employer = np.full(self.n, -1, dtype=np.int64)
        target = chosen_worker          # (F,)
        wage = chosen_wage              # (F,)
        res = self.res               # (W,)
        for w in range(self.n):
            firms = np.flatnonzero(target == w)
            if firms.size == 0:
                continue
            feas_mask = wage[firms] > res[w]
            if not np.any(feas_mask):
                continue
            feas_firms = firms[feas_mask]
            f_win = feas_firms[np.argmax(wage[feas_firms])]
            firm_hired[f_win] = True
            worker_employer[w] = f_win
        return firm_hired, worker_employer

I_MU, I_NU, I_ALPHA, I_BETA, I_DELTA, I_KAPPA = 0, 1, 2, 3, 4, 5
N_FIELDS = 6

@dataclass
class Firms:
    n: int = 1
    n_g: int = 2 # Num Groups

    hierarchical: bool = False # If firms use model expectations using hierarchical bayes
    global_mean: float = 2.0 # Only used if hierarchical == True

    #Posteriors - uncertainty over true distribution is the same for all groups
    mu_0: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0]))
    nu_0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    alpha_0: np.ndarray = field(default_factory=lambda: np.array([25.0, 2.5]))
    beta_0: np.ndarray = field(default_factory=lambda: np.array([2.5, 25.0]))

    # Posteriors - uncertainty over quality of signal is different
    # Correctly believe tha signal is unbiased - true mean of signal variation is zero
    delta_0: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0]))
    kappa_0: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0]))

    def __post_init__(self):
        self.beliefs = self._init_firm_posteriors(self.n, self.n_g, self.mu_0,
                                                 self.nu_0, self.alpha_0,
                                                 self.beta_0, self.delta_0,
                                                 self.kappa_0)

    def _init_firm_posteriors(self,
        n_firms: int,
        n_g: int,
        mu_0: np.ndarray,
        nu_0: np.ndarray,
        alpha_0: np.ndarray,
        beta_0: np.ndarray,
        delta_0: np.ndarray,   # shape (n_g,) or (n_firms, n_g)
        kappa_0: np.ndarray, # shape (n_g,) or (n_firms, n_g)
    ) -> np.ndarray:

        """
        posterior[f, g, :] = [mu, nu, alpha, beta, delta, kappa]
        """
        posterior = np.zeros((n_firms, n_g, N_FIELDS), dtype=np.float64)
        posterior[:, :, I_MU] = mu_0
        posterior[:, :, I_NU] = nu_0
        posterior[:, :, I_ALPHA] = alpha_0
        posterior[:, :, I_BETA] = beta_0

        d = np.asarray(delta_0, dtype=np.float64)
        k = np.asarray(kappa_0, dtype=np.float64)
        if d.ndim == 1:
            d = np.broadcast_to(d, (n_firms, n_g))
        if k.ndim == 1:
            k = np.broadcast_to(k, (n_firms, n_g))
        posterior[:, :, I_DELTA] = d
        posterior[:, :, I_KAPPA] = k
        return posterior

    def batch_update_posteriors(self,
    posterior: np.ndarray,
    firm_idx: np.ndarray,
    group_idx: np.ndarray,
    obs_prod: np.ndarray,
    obs_signal: np.ndarray,
    ) -> None:
        """In-place update for each (firm, group) pair in the batch."""
        mu = posterior[firm_idx, group_idx, I_MU]
        nu = posterior[firm_idx, group_idx, I_NU]
        alpha = posterior[firm_idx, group_idx, I_ALPHA]
        beta = posterior[firm_idx, group_idx, I_BETA]
        delta = posterior[firm_idx, group_idx, I_DELTA]
        kappa = posterior[firm_idx, group_idx, I_KAPPA]

        obs_signal_error = obs_prod - obs_signal

        nu_post = nu + 1.0
        mu_post = (mu * nu + obs_prod) / nu_post
        alpha_post = alpha + 0.5
        beta_post = beta + (nu * (obs_prod - mu) ** 2) / (2.0 * nu_post)
        delta_post = delta + 0.5
        kappa_post = kappa + (obs_signal_error**2) / 2.0

        posterior[firm_idx, group_idx, I_MU] = mu_post
        posterior[firm_idx, group_idx, I_NU] = nu_post
        posterior[firm_idx, group_idx, I_ALPHA] = alpha_post
        posterior[firm_idx, group_idx, I_BETA] = beta_post
        posterior[firm_idx, group_idx, I_DELTA] = delta_post
        posterior[firm_idx, group_idx, I_KAPPA] = kappa_post

    def update_priors(self,         # (F,G,6), in-place
        group_mask: np.ndarray,       # (F,G) bool
        accepted: np.ndarray,         # (F,) bool
        obs_prod: np.ndarray,         # (F,)
        obs_signal: np.ndarray,       # (F,)
    ):
        # keep only firms that accepted and have one selected group
        active = accepted & group_mask.any(axis=1)
        firm_idx = np.flatnonzero(active)
        if firm_idx.size == 0:
            return

        # Get groups of accepted matches
        group_idx = np.argmax(group_mask[firm_idx], axis=1)

        mu = self.beliefs[firm_idx, group_idx, I_MU]
        nu = self.beliefs[firm_idx, group_idx, I_NU]
        alpha = self.beliefs[firm_idx, group_idx, I_ALPHA]
        beta = self.beliefs[firm_idx, group_idx, I_BETA]
        delta = self.beliefs[firm_idx, group_idx, I_DELTA]
        kappa = self.beliefs[firm_idx, group_idx, I_KAPPA]

        p = obs_prod[firm_idx]
        s = obs_signal[firm_idx]
        e = p - s

        nu_post = nu + 1.0
        mu_post = (mu * nu + p) / nu_post
        alpha_post = alpha + 0.5
        beta_post = beta + (nu * (p - mu) ** 2) / (2.0 * nu_post)
        delta_post = delta + 0.5
        kappa_post = kappa + 0.5 * (e ** 2)

        self.beliefs[firm_idx, group_idx, I_MU] = mu_post
        self.beliefs[firm_idx, group_idx, I_NU] = nu_post
        self.beliefs[firm_idx, group_idx, I_ALPHA] = alpha_post
        self.beliefs[firm_idx, group_idx, I_BETA] = beta_post
        self.beliefs[firm_idx, group_idx, I_DELTA] = delta_post
        self.beliefs[firm_idx, group_idx, I_KAPPA] = kappa_post

    def get_priors(self, workers, pairs) -> tuple[np.ndarray, np.ndarray]:
        """sig_var and prod_var for all (firm, group). Shapes (n_firms, n_g)."""
        _cand_groups = workers.groups[pairs] 
        # gather posterior params for each candidate's group
        firm_idx_2d = np.arange(self.n)[:, None]     # (F,1)
        _cand_post = self.beliefs[firm_idx_2d, _cand_groups, :]   # (F,2,6)
        mu = _cand_post[:, :, I_MU]
        nu = _cand_post[:, :, I_NU]
        alpha = _cand_post[:, :, I_ALPHA]
        beta = _cand_post[:, :, I_BETA]
        delta = _cand_post[:, :, I_DELTA]
        kappa = _cand_post[:, :, I_KAPPA]

        sig_var = kappa / (delta - 1.0)
        prod_var = beta / (alpha - 1.0)
        return sig_var, prod_var, mu, nu, alpha, beta, delta, kappa

    def choice(self, options):
        """Function for choosing which bandit is best; currently myopic, need to implement UCB here too;
        COuld also potentially do q-Learning? But not really applicable here!"""
        pass

N_G = 2 # Num groups same in both worker and firm classes

default_firm_kwargs = {
    "n": 10,
    "n_g": N_G
}

default_worker_kwargs = {
    "n": 20,
    "n_g": N_G,
    "group_1_share": 0.1,
}

class Simulation:

    def __init__(self,
                firm_kwargs: dict = default_firm_kwargs, 
                worker_kwarg: dict = default_worker_kwargs,
                horizon: int = 10,
                wage_dist_which: str = "all_offers",
                wage_dist_scope: str = "all",
                ucb: bool = False,
                replace_firms: bool = True,
                no_hire_cost: float = -2.5
                                ):

                self.horizon: int = horizon
                self.wage_dist_which = wage_dist_which
                if wage_dist_scope not in ("all", "final"):
                    raise ValueError('wage_dist_scope must be "all" or "final"')
                self.wage_dist_scope = wage_dist_scope
                self._wage_dist_scope_built: str | None = None
                self.wages: np.ndarray | None = None
                self.regret: np.ndarray | None = None
                self.fkwargs = firm_kwargs
                self.wkwargs = worker_kwarg

                self.firms = Firms(**self.fkwargs)
                self.workers = Workers(**self.wkwargs)
                self.nw = self.workers.n
                self.nf = self.firms.n
                self.ng = self.workers.n_g

                self.replace_firms = replace_firms

                self.no_hire_cost = no_hire_cost

                self.ucb = ucb

                self._init_log_buffers()

    def _init_log_buffers(self) -> None:
        """Preallocate Monte Carlo logs. Call from __init__ and reset()."""
        T, F, G = self.horizon, self.nf, self.ng

        self.wage_offers = np.full((T, F, 2), np.nan, dtype=np.float64)
        self.chosen_wage_log = np.full((T, F), np.nan, dtype=np.float64)
        self.accepted_log = np.zeros((T, F), dtype=bool)

        self.profit = np.zeros((T, F), dtype=np.float64)
        self.regret = np.zeros((T, F), dtype=np.float64)

        # Mean chosen wage by group (for plots like simulation.py)
        self.wages = np.full((T, G), np.nan, dtype=np.float64)

        self.all_wages = self.wage_offers
        self.accepted_wages = np.full((T, F), np.nan, dtype=np.float64)

        # Step counters set by simulate()
        self._t = 0

        # Scratch filled in firm_step, read in record()
        self._wage_offer = None
        self._cand_prod = None
        self._cand_res = None

        self.chosen_group_log = np.zeros((T, F), dtype=np.int64)
        self.cand_groups_log = np.zeros((T, F, 2), dtype=np.int64)

        self.cand_surplus_log = np.full((T, F, 2), np.nan, dtype=np.float64)
        self.surplus_obtained_log = np.zeros((T, F), dtype=np.float64)

        self.emp_record = np.full((T, 2), np.nan, dtype=np.float64)

        # Cumulative statistics
        self.cum_profit = np.zeros((F, ), dtype=np.float64)
        self.cum_regret = np.zeros((F, ), dtype=np.float64)


        self.wage_by_group: dict[int, np.ndarray] | None = None
        self.cdf_by_group: dict[int, tuple[np.ndarray, np.ndarray]] | None = None

    def match_two_per_firm(self, rng: np.random.Generator) -> np.ndarray:
        """
        Per-firm bandit arms: up to 2 distinct workers drawn independently across firms.
        Both slots may be the same group. Requires at least one worker in the pool.
        """
        if self.nw < 1:
            raise ValueError("Need at least one worker")

        pairs = np.empty((self.nf, 2), dtype=np.int64)
        for f in range(self.nf):
            if self.nw >= 2:
                pairs[f] = rng.choice(self.nw, size=2, replace=False)
            else:
                pairs[f] = 0
        return pairs

    def build_group_mask(self):
        mask = np.zeros((self.nf, self.ng), dtype=bool)
        mask[np.arange(self.nf), self.chosen_group] = True
        return mask

    def firm_step(self, pairs):
        """
        1. Get information on candidates; arrays indexed with pair indices
        2. Calculate Wage offer from firms for matched individuals
        3. Choose bandit with highest expected profit (signal - wage_offer)
        4. Update posteriors
        """
        cand_groups = self.workers.groups[pairs]   # (F, 2)
        cand_signal = self.workers.signal[pairs]   # (self.n,2)
        cand_prod = self.workers.prod[pairs]       # (self.n,2)
        cand_res = self.workers.res[pairs]         # (self.n,2)
  
        sig_var, prod_var, *priors = self.firms.get_priors(self.workers, pairs)
        mu = priors[0]
        w_sig = prod_var / (prod_var + sig_var)

        # Weight signal and prior mean by their variances
        # (w_sig) -> 0 as sig_var -> inf; w_sig -> 1 as sig_var -> 0
        # TODO! Wage offer function? ALthough I think it will always be the same!
        wage_offer = w_sig * cand_signal + (1.0 - w_sig) * mu   # (F,2)


        if self.ucb:
            c = 25
            p_t = 1 - (1 / (self._t + 1)) ** c
            exp_profit_ucb = stats.norm.ppf(p_t, loc=mu, scale=prod_var)
            choice = np.argmax(exp_profit_ucb, axis=1)  # (F,)
            print(exp_profit_ucb)
        else:
            # Expected productivity of current matched workers is their signal since they believe it is unbiased!
            exp_profit = cand_signal - wage_offer
            choice = np.argmax(exp_profit, axis=1)  # (F,)
            print(exp_profit)

        
        f = np.arange(self.nf)

        self.chosen_worker = pairs[f, choice]     # (F,)
        self.chosen_group = cand_groups[f, choice]   # (F,)
        self.chosen_wage = wage_offer[f, choice]     # (F,)
        self.chosen_prod = cand_prod[f, choice]      # (F,)
        self.chosen_signal = cand_signal[f, choice]  # (F,)
        self.chosen_res = cand_res[f, choice]        # (F,)

        self.firm_hired, self.worker_employer = self.workers.wage_acceptance(self.chosen_worker, self.chosen_wage, self.nf)
        self.accepted = self.firm_hired                          # (F,) bool

        self.mask = self.build_group_mask()

        self.firms.update_priors(
            self.mask, 
            self.accepted,
            self.chosen_prod,
            self.chosen_signal)

        self._wage_offer = wage_offer          # (F, 2)
        self._cand_prod = cand_prod            # (F, 2)
        self._cand_res = cand_res              # (F, 2)
        self._cand_groups = cand_groups        # (F, 2)

        # Tracking Firm behaviour
        self._chosen_exp_prod = mu[f, choice]        # (F, ) ; expected productivity (prior on candidates productivity)
        self._exp_diff = self.chosen_prod - self._chosen_exp_prod # (F, ) ; difference between observed and expected productivity

        ## Note
        # This is driving the changes in expectations - IF there is a trly bad initial experience, ie an individual from gr 1 
        # is hired, the observed productivity is much worse, and they tend not to hire from thm again, while positive experiences
        # tend to reutrn to proper expectations, as they hire from the same group again, and consequently gather more data on them

        self.record()
        if self.replace_firms:
            self.check_profits()

    def check_profits(self):

        # If firm cumulative profits fall below 2.5, then they are replaced by a firm with refreshed priors and zero profit

        profit = self.cum_profit

        f_idx = profit < -2.5 

        self.firms.beliefs[f_idx, :, I_MU] = self.firms.mu_0
        self.firms.beliefs[f_idx, :, I_NU] = self.firms.nu_0
        self.firms.beliefs[f_idx, :, I_ALPHA] = self.firms.alpha_0
        self.firms.beliefs[f_idx, :, I_BETA] = self.firms.beta_0
        self.firms.beliefs[f_idx, :, I_KAPPA] = self.firms.kappa_0
        self.firms.beliefs[f_idx, :, I_DELTA] = self.firms.delta_0
        self.cum_profit[f_idx] = 0

        


        
    def record(self) -> None:
        """Write one period into preallocated logs."""
        t = self._t
        w = self._wage_offer
        p = self._cand_prod
        r = self._cand_res
        d = self._exp_diff

        # See firm_step
        self.surplus_obtained_log[t] = d

        self.chosen_group_log[t] = self.chosen_group
        self.cand_groups_log[t] = self._cand_groups

        # Offers and choices
        self.wage_offers[t] = w
        self.chosen_wage_log[t] = self.chosen_wage
        self.accepted_log[t] = self.accepted

        # Profit: surplus only if hire accepted
        actual_profit = np.where(
            self.accepted,
            self.chosen_prod - self.chosen_wage,
            self.no_hire_cost,
        )

        self.profit[t] = actual_profit
        self.cum_profit += actual_profit

        # Oracle: best feasible surplus among the two candidates
        feasible = w > r
        surplus = p - w
        surplus = np.where(feasible, surplus, np.nan)
        oracle = np.nanmax(surplus, axis=1)
        oracle = np.nan_to_num(oracle, nan=0.0)

        self.regret[t] = oracle - actual_profit
        self.cum_regret += oracle - actual_profit


        #TODO! Make plot for this, where it shows pc of 1s, pc of 0s employed, then % 1's unemp, pc 0s unemp
        # On a graph that obviously always adds up to 1, then shaded in with each colour.
        emp_record = self.workers.groups[self.worker_employer > -0.5]
        ones = np.sum(emp_record == 1) / self.nw
        zeros = np.sum(emp_record == 0) / self.nw
        self.emp_record[t] = [zeros, ones]
        

        # Accepted wage (NaN if rejected)
        self.accepted_wages[t] = np.where(
            self.accepted,
            self.chosen_wage,
            np.nan,
        )

        # Mean chosen wage by group (among firms that chose that group)
        for g in range(self.ng):
            mask = self.chosen_group == g
            if np.any(mask):
                self.wages[t, g] = float(np.mean(self.chosen_wage[mask]))


    def simulate(self, base_seed: int = 1):
        for t in range(self.horizon):
            self._t = t
            rng = np.random.default_rng(base_seed + t)
            self.pairs = self.match_two_per_firm(rng)
            self.firm_step(self.pairs)
            
        self._build_wage_distributions()

    def trace_firm_choice(self):
        choice = self.chosen_group_log
        diff = self.surplus_obtained_log
        print(choice.shape)
        for firm_log in range(self.nf):
            if choice[:, firm_log].sum()/ self.horizon < 0.05:
                print(diff[:, firm_log][choice[:, firm_log] == 1])

    def model_wide_stats(self):
        ## MLE and LR test for mean and variance of wage dist.
        choice = self.chosen_group_log[self._t]

        g0_data = self.chosen_wage_log[self._t][choice == 0]
        g1_data = self.chosen_wage_log[self._t][choice == 1]

        #m0, v0, s0, k0 = stats.norm.fit(g0_data, moments='mvsk')
        #m1, v1, s1, k1= stats.norm.fit(g1_data, moments='mvsk')

        #print(m0, v0, s0, k0, m1, v1, s1, k1)

        ## Employment
        
        
        

    def _period_range(self, scope: str | None) -> range | list[int]:
        scope = scope or self.wage_dist_scope
        if scope == "all":
            return range(self.horizon)
        if scope == "final":
            return [self.horizon - 1]
        raise ValueError(f'unknown scope={scope!r}; use "all" or "final"')

    def collect_wages_by_group(
        self,
        scope: str | None = None,
        which: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Pool wages from period logs. scope: 'all' | 'final'. which: wage_dist_which."""
        scope = scope or self.wage_dist_scope
        which = which or self.wage_dist_which
        buckets: dict[int, list[float]] = {g: [] for g in range(self.ng)}

        for t in self._period_range(scope):
            if which == "chosen":
                for f in range(self.nf):
                    g = int(self.chosen_group_log[t, f])
                    buckets[g].append(self.chosen_wage_log[t, f])
            elif which == "all_offers":
                for f in range(self.nf):
                    for j in range(2):
                        g = int(self.cand_groups_log[t, f, j])
                        buckets[g].append(self.wage_offers[t, f, j])
            elif which == "accepted":
                for f in range(self.nf):
                    if self.accepted_log[t, f]:
                        g = int(self.chosen_group_log[t, f])
                        buckets[g].append(self.accepted_wages[t, f])
            else:
                raise ValueError(f"unknown which={which!r}")

        return {g: np.asarray(buckets[g], dtype=np.float64) for g in buckets}

    def _build_wage_distributions(self, scope: str | None = None) -> None:
        scope = scope or self.wage_dist_scope
        self._wage_dist_scope_built = scope
        self.wage_by_group = self.collect_wages_by_group(scope=scope)
        self.cdf_by_group = {}
        for g, w in self.wage_by_group.items():
            w = w[np.isfinite(w)]
            if w.size == 0:
                self.cdf_by_group[g] = (np.array([]), np.array([]))
                continue
            x = np.sort(w)
            y = np.arange(1, w.size + 1, dtype=np.float64) / w.size
            self.cdf_by_group[g] = (x, y)

    def fraction_above(
        self,
        group: int,
        threshold: float,
        scope: str | None = None,
    ) -> float:
        scope = scope or self.wage_dist_scope
        if self.wage_by_group is None or self._wage_dist_scope_built != scope:
            self._build_wage_distributions(scope)
        w = self.wage_by_group[group]
        w = w[np.isfinite(w)]
        if w.size == 0:
            return float("nan")
        return float(np.mean(w > threshold))

    def plot_wage_cdf(
        self,
        path: str = "figs/wage_cdf.png",
        survival: bool = False,
        scope: str | None = None,
    ) -> None:
        scope = scope or self.wage_dist_scope
        if self.cdf_by_group is None or self._wage_dist_scope_built != scope:
            self._build_wage_distributions(scope)

        period_label = "final period" if scope == "final" else "all periods"
        fig, ax = plt.subplots(figsize=(8, 5))
        for g, (x, y) in self.cdf_by_group.items():
            if x.size == 0:
                continue
            if survival:
                y_plot = 1.0 - y
                ylab = r"P(wage > $w$)"
                title = (
                    f"Wage exceedance by group ({self.wage_dist_which}, {period_label})"
                )
            else:
                y_plot = y
                ylab = r"P(wage $\leq$ $w$)"
                title = f"Wage CDF by group ({self.wage_dist_which}, {period_label})"
            n = self.wage_by_group[g].size
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

    def plot_regret_over_time(
        self,
        path: str = "figs/regret_over_time.png",
        *,
        cumulative: bool = False,
        show_mean: bool = True,
        firm_alpha: float = 0.03,
        max_firms_plot: int | None = None,
    ) -> None:
        """
        Plot per-period regret for each firm. Uses self.regret with shape (horizon, n_firms).
        """
        regret = np.cumsum(self.regret, axis=0) if cumulative else self.regret
        time = np.arange(self.horizon)
        n_plot = self.nf if max_firms_plot is None else min(self.nf, max_firms_plot)

        fig, ax = plt.subplots(figsize=(10, 6))
  
        ax.plot(
                time,
                regret[:, :n_plot].mean(axis=1),
                color="C1",
                linewidth=2.0,
                label="mean across firms",
            )

        ax.set_xlabel("Period")
        ax.set_ylabel("Cumulative regret" if cumulative else "Regret")
        title = f"Firm regret over time ({n_plot} firms"
        if max_firms_plot is not None and max_firms_plot < self.nf:
            title += f" of {self.nf}"
        title += ")"
        ax.set_title(title)
        if show_mean:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def reset(self) -> None:
        self.firms = Firms(**self.fkwargs)
        self.workers = Workers(**self.wkwargs)
        self.nw = self.workers.n
        self.nf = self.firms.n
        self.ng = self.workers.n_g
        self._init_log_buffers()


if __name__ == "__main__":
    pass



