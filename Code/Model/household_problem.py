"""Household problem solver using DC-EGM with discrete housing choice.

Implements Algorithm 1 (backward iteration over time and housing states) and 
Algorithm 2 (EGM-step with productivity shocks) from Iskhakov et al. (2017).
"""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace


class HousingModel:


    def __init__(self, model=None):
        """Initialize from GE model or standalone for demonstrating partial equilibrium.
        """
        if model is not None:
            self.par = model.par
            self.ss = model.ss
            self.sol = model.sol
            self.model_ge = model
        else:
            self.par = SimpleNamespace()
            self.ss = SimpleNamespace()
            self.sol = SimpleNamespace()
            self.model_ge = None



    # =========================================================================
    # 1. Main hh problem solver methods
    # =========================================================================

    def solve_dcegm(self, prices_path=None):
        """Solve household problem backwards using DC-EGM.
        
        Directly stores solutions in model.sol arrays.
        
        """
        par = self.par
        sol = self.sol
        
        # a. Ensure grids exist and are valid
        if (not hasattr(par, 'a_grid') or par.a_grid is None or len(par.a_grid) < 2 or
            not np.any(np.asarray(par.a_grid) > 0)):
            par.a_grid = np.linspace(0.0, par.mmax, par.Na)
        if (not hasattr(par, 'm_grid') or par.m_grid is None or len(par.m_grid) < 2 or
            not np.any(np.asarray(par.m_grid) > 0)):
            par.m_grid = par.a_grid.copy()

        # b. Extract dimensions
        Na = par.a_grid.size
        T = par.T
        Nfix = par.Nfix if hasattr(par, 'Nfix') else 1
        Nz = par.Nz
        
        # c. Allocate solution arrays if not already done
        Nh = 4  # rural_renter=0, urban_renter=1, rural_owner=2, urban_owner=3
        sol.c = np.zeros((Nfix, Nz, Nh, Na))
        sol.a = np.zeros((Nfix, Nz, Nh, Na))
        sol.pr_choices = np.zeros((Nfix, Nz, Nh, 4, Na))  # prob of each of 4 choices
        sol.pr_urban = np.zeros((Nfix, Nz, Nh, Na))  # marginal urban probability (for compatibility)
        sol.pr_choices_asav = np.zeros((Nfix, Nz, Nh, 4, Na))  # z_next-conditional probs on savings grid
        
        # d. Setup prices - either time-varying or steady state (depending on PE or GE)
        _kappa_orig = par.kappa  # save to restore after solve
        if prices_path is None:
            # Steady state: same prices at all periods
            def get_prices(t):
                f_u = float(getattr(self.ss, 'f_u', self.ss.q_u * ((1.0 + self.ss.r) - ((1- par.theta) * (1.0 - par.delta_H)))))
                f_r = float(getattr(self.ss, 'f_r', self.ss.q_r * ((1.0 + self.ss.r) - ((1- par.theta) * (1.0 - par.delta_H)))))
                return self.ss.r, self.ss.w, self.ss.q_u, self.ss.q_r, f_u, f_r
            def get_kappa(t):
                return _kappa_orig
        else:
            # Transition path: time-varying prices
            def get_prices(t):
                r_t = prices_path['r'][t] if prices_path['r'].ndim == 1 else prices_path['r'][t, 0]
                w_t = prices_path['w'][t] if prices_path['w'].ndim == 1 else prices_path['w'][t, 0]
                q_u_t = prices_path['q_u'][t] if prices_path['q_u'].ndim == 1 else prices_path['q_u'][t, 0]
                q_r_t = prices_path['q_r'][t] if prices_path['q_r'].ndim == 1 else prices_path['q_r'][t, 0]
                if 'f_u' in prices_path and 'f_r' in prices_path:
                    f_u_t = prices_path['f_u'][t] if prices_path['f_u'].ndim == 1 else prices_path['f_u'][t, 0]
                    f_r_t = prices_path['f_r'][t] if prices_path['f_r'].ndim == 1 else prices_path['f_r'][t, 0]
                else:
                    # Fallback: compute from rental sector FOC at steady-state q_{t+1}=q_t
                    f_u_t = float(q_u_t * ((1.0 + r_t) - ((1- par.theta) * (1.0 - par.delta_H))))
                    f_r_t = float(q_r_t * ((1.0 + r_t) - ((1- par.theta) * (1.0 - par.delta_H))))
                return r_t, w_t, q_u_t, q_r_t, f_u_t, f_r_t
            def get_kappa(t):
                if 'kappa' not in prices_path:
                    return _kappa_orig
                kappa_arr = prices_path['kappa']
                return float(kappa_arr[t] if np.asarray(kappa_arr).ndim == 1 else kappa_arr[t, 0])
        
        # e. Setup discrete productivity states
        savingsgrid = par.a_grid
        m_grid_regular = par.m_grid
        z_grid = np.asarray(par.z_grid)
        z_trans_fix = np.asarray(par.z_trans[0] if np.asarray(par.z_trans).ndim == 3 else par.z_trans)
        
        # e. Endogenous grid storage (4 housing states)
        # 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner
        policy_endog = {0: {}, 1: {}, 2: {}, 3: {}}
        value_endog = {0: {}, 1: {}, 2: {}, 3: {}}
        

        # =====================================================================
        # ALGORITHM 1: Outer loop. Backward iteration over time.
        # =====================================================================
        
        J_life = int(getattr(par, 'J', T))  # lifecycle length; T if not set
        terminal_it = J_life - 1           # last age an agent is alive

        for it in range(terminal_it, -1, -1):
            
            # a. Get prices for current period
            r, w, q_u, q_r, f_u, f_r = get_prices(it)
            par.kappa = get_kappa(it)
            
            # b. Terminal period: consume everything (no bequest motive)
            if it == terminal_it:
                self._solve_terminal_period(
                    it, policy_endog, value_endog, m_grid_regular,
                    sol, Nfix, Nz, Na
                )
                continue
            
            # c. Non-terminal periods: solve EGM for each current housing state
            #    States: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner
            self._solve_non_terminal_period(
                it, policy_endog, value_endog, savingsgrid, m_grid_regular,
                z_grid, z_trans_fix, r, w, q_u, q_r, f_u, f_r,
                sol, Nfix, Nz, Na
            )
        
        # Store endogenous grid solutions for reference
        self.policy_endog = policy_endog
        self.value_endog = value_endog

        par.kappa = _kappa_orig  # restore after time-varying solve

        # Sanity check: warn if sol arrays are degenerate, e.g. below zero
        if (not np.any(np.isfinite(sol.c)) or np.nanmax(sol.c) <= 0.0 or
            not np.any(np.isfinite(sol.a))):
            if prices_path is None:
                # Steady-state solve: prices should never be bad — raise here.
                raise RuntimeError(
                    "Household solver produced empty solution arrays. "
                    "Check that grids are initialized and solve_hh_backwards() is called."
                )
            else:
                import warnings
                warnings.warn(
                    "solve_dcegm: NaN/zero sol arrays detected. "
                    "Filling with cfloor fallback.",
                    RuntimeWarning, stacklevel=2
                ) #This shoudl never trigger
                sol.c[:] = par.cfloor
                sol.a[:] = 0.0
                sol.pr_choices[:] = 0.25
                sol.pr_urban[:] = 0.5

        return policy_endog, value_endog






    def solve_dcegm_calendar_age_grid(self, prices_dict):
        """Backward sweep on the full (T_calendar, J_age) grid in a single pass,

        as per Appendix B.1.

        Replaces T per-cohort solve_dcegm calls when the user needs policies at
        every (calendar_t, age) pair (cohort = calendar_t - age). Eliminates the
        ~T*J/2 redundant EGM steps that occur in the SS tail when each cohort
        re-solves an entire lifetime, plus T copies of solve_dcegm setup.

        Args:
            prices_dict: dict with keys 'r', 'w', 'q_u', 'q_r', 'kappa' and 'f_u', 'f_r'. Each is a length-T calendar-time array.

        Continuation at (calendar_t+1, age+1) is clamped at calendar_t+1 = T-1
        to match the SS-tail behaviour of the original per-cohort solver
        (`_build_shifted_prices` clamps at min(birth+age, T-1)).

        Returns dense numpy arrays:
            a_arr   : (T, J, Nz, 4, Na)         savings policy on par.m_grid
            c_arr   : (T, J, Nz, 4, Na)         consumption policy on par.m_grid
            pr_arr  : (T, J, 4, Nz, 4, Na)      next-state choice probs
                       (axis 2 = next h_state, axis 3 = i_z, axis 4 = i_h)
        """
        par = self.par
        sol = self.sol

        # a. Ensure grids exist
        if (not hasattr(par, 'a_grid') or par.a_grid is None or len(par.a_grid) < 2 or
            not np.any(np.asarray(par.a_grid) > 0)):
            par.a_grid = np.linspace(0.0, par.mmax, par.Na)
        if (not hasattr(par, 'm_grid') or par.m_grid is None or len(par.m_grid) < 2 or
            not np.any(np.asarray(par.m_grid) > 0)):
            par.m_grid = par.a_grid.copy()

        Na = par.a_grid.size
        T = par.T
        J = int(getattr(par, 'J', T))
        J = max(1, min(J, T))
        Nfix = par.Nfix if hasattr(par, 'Nfix') else 1
        Nz = par.Nz
        Nh = 4
        savingsgrid = par.a_grid
        m_grid_reg = par.m_grid

        # b. Ensure all sol arrays the inner solvers write into exist 
        for _name, _shape in (
            ('c',                 (Nfix, Nz, Nh, Na)),
            ('a',                 (Nfix, Nz, Nh, Na)),
            ('pr_choices',        (Nfix, Nz, Nh, 4, Na)),
            ('pr_urban',          (Nfix, Nz, Nh, Na)),
            ('pr_choices_asav',   (Nfix, Nz, Nh, 4, Na)),
        ):
            if (not hasattr(sol, _name) or
                np.asarray(getattr(sol, _name)).shape != _shape):
                setattr(sol, _name, np.zeros(_shape))

        z_grid = np.asarray(par.z_grid)
        z_trans_fix = np.asarray(par.z_trans[0]
                                 if np.asarray(par.z_trans).ndim == 3 else par.z_trans)

        _kappa_orig = par.kappa

        # c. Pre-extract scalar price paths
        def _arr(name):
            x = np.asarray(prices_dict[name])
            if x.ndim == 2:
                x = x[:, 0]
            return x.ravel()[:T].astype(np.float64)

        r_path = _arr('r'); w_path = _arr('w')
        q_u_path = _arr('q_u'); q_r_path = _arr('q_r')
        if 'f_u' in prices_dict and 'f_r' in prices_dict:
            f_u_path = _arr('f_u'); f_r_path = _arr('f_r')
        else:
            f_u_path = q_u_path * ((1.0 + r_path) - (1.0 - par.theta) * (1.0 - par.delta_H))
            f_r_path = q_r_path * ((1.0 + r_path) - (1.0 - par.theta) * (1.0 - par.delta_H))
        if 'kappa' in prices_dict:
            kappa_path = _arr('kappa')
        else:
            kappa_path = np.full(T, _kappa_orig)

        # d. Output dense arrays
        a_arr  = np.zeros((T, J, Nz, Nh, Na))
        c_arr  = np.zeros((T, J, Nz, Nh, Na))
        pr_arr = np.zeros((T, J, 4, Nz, Nh, Na))

        # e. Per-calendar-time "rolling container" dicts: pol_next[ct][i_h][i_z] holds
        # the policy at (ct, age+1). Rolled forward as we sweep age = J-1 .. 0.
        pol_next = [None] * T
        val_next = [None] * T

        # f. Backward sweep over both T and J
        for age in range(J - 1, -1, -1):
            pol_curr = [None] * T
            val_curr = [None] * T
            prices_prev = None   # prices at ct+1, for flat-tail deduplication

            for ct in range(T - 1, -1, -1):
                # Prices at calendar time ct
                r = float(r_path[ct]);    w = float(w_path[ct])
                q_u = float(q_u_path[ct]); q_r = float(q_r_path[ct])
                f_u = float(f_u_path[ct]); f_r = float(f_r_path[ct])
                kap = float(kappa_path[ct])
                prices_ct = (r, w, q_u, q_r, f_u, f_r, kap)

                # Deduplicate for speed gains: the (ct, age) solve is identical to (ct+1, age)
                # when the prices match AND the continuation policy is the same
                # object (true throughout the flat post-2070 price tail).
                if (ct < T - 1 and prices_ct == prices_prev
                        and (age == J - 1
                             or pol_next[ct + 1] is pol_next[min(ct + 2, T - 1)])):
                    pol_curr[ct] = pol_curr[ct + 1]
                    val_curr[ct] = val_curr[ct + 1]
                    a_arr[ct, age]  = a_arr[ct + 1, age]
                    c_arr[ct, age]  = c_arr[ct + 1, age]
                    pr_arr[ct, age] = pr_arr[ct + 1, age]
                    prices_prev = prices_ct
                    continue
                prices_prev = prices_ct
                par.kappa = kap

                # Build a temporary (single-period) policy_endog around this cell.
                fake_pe = {0: {}, 1: {}, 2: {}, 3: {}}
                fake_ve = {0: {}, 1: {}, 2: {}, 3: {}}

                if age == J - 1:
                    self._solve_terminal_period(
                        age, fake_pe, fake_ve, m_grid_reg, sol, Nfix, Nz, Na,
                    )
                else:
                    cont_t = ct + 1 if ct + 1 < T else T - 1
                    pol_n = pol_next[cont_t]
                    val_n = val_next[cont_t]
                    for i_h in range(4):
                        fake_pe[i_h][age + 1] = pol_n[i_h]
                        fake_ve[i_h][age + 1] = val_n[i_h]

                    self._solve_non_terminal_period(
                        age, fake_pe, fake_ve, savingsgrid, m_grid_reg,
                        z_grid, z_trans_fix, r, w, q_u, q_r, f_u, f_r,
                        sol, Nfix, Nz, Na,
                    )

                # Stash the (ct, age) policy for use as continuation at age-1
                pol_curr_ct = {i_h: fake_pe[i_h][age] for i_h in range(4)}
                val_curr_ct = {i_h: fake_ve[i_h][age] for i_h in range(4)}
                pol_curr[ct] = pol_curr_ct
                val_curr[ct] = val_curr_ct

                # Interpolate to regular grid → write into dense (T, J, ...) arrays
                for i_h in range(4):
                    pe_ih = pol_curr_ct[i_h]
                    if isinstance(pe_ih, dict) and len(pe_ih) > 0 and isinstance(next(iter(pe_ih)), int):
                        # z-conditional structure {i_z: {'m','c','pr_choices'}}
                        for i_z in range(Nz):
                            pol = pe_ih.get(i_z, pe_ih.get(0))
                            c_line = np.interp(m_grid_reg, pol['m'], pol['c'])
                            a_line = m_grid_reg - c_line
                            a_arr[ct, age, i_z, i_h, :] = a_line
                            c_arr[ct, age, i_z, i_h, :] = c_line
                            if 'pr_choices' in pol:
                                for k in range(4):
                                    pr_arr[ct, age, k, i_z, i_h, :] = np.interp(
                                        m_grid_reg, pol['m'], pol['pr_choices'][k]
                                    )
                            else:
                                pr_arr[ct, age, :, i_z, i_h, :] = 0.25
                    else:
                        # Terminal/flat structure {'m','c','pr_choices'}
                        pol = pe_ih
                        c_line = np.interp(m_grid_reg, pol['m'], pol['c'])
                        a_line = m_grid_reg - c_line
                        for i_z in range(Nz):
                            a_arr[ct, age, i_z, i_h, :] = a_line
                            c_arr[ct, age, i_z, i_h, :] = c_line
                        if 'pr_choices' in pol:
                            pc = np.asarray(pol['pr_choices'])
                            if pc.ndim == 2 and pc.shape[1] == m_grid_reg.size:
                                # Already on regular grid? Use directly via interp on shared m
                                for k in range(4):
                                    pr_arr[ct, age, k, :, i_h, :] = np.interp(
                                        m_grid_reg, pol['m'], pc[k]
                                    )[None, :]
                            else:
                                for k in range(4):
                                    pr_arr[ct, age, k, :, i_h, :] = np.interp(
                                        m_grid_reg, pol['m'], pc[k]
                                    )[None, :]
                        else:
                            pr_arr[ct, age, :, :, i_h, :] = 0.25

            # Roll
            pol_next = pol_curr
            val_next = val_curr

        par.kappa = _kappa_orig
        return a_arr, c_arr, pr_arr
    


    def solve_dcegm_one_step(self, it, policy_endog, value_endog, prices_path=None,
                              force_non_terminal=False):
        """Solve a single backward step of DC-EGM for period it - needed for computing Jacobian of hh block.
        """
        par = self.par
        sol = self.sol

        # a. Ensure grids exist
        if (not hasattr(par, 'a_grid') or par.a_grid is None or len(par.a_grid) < 2 or
            not np.any(np.asarray(par.a_grid) > 0)):
            par.a_grid = np.linspace(0.0, par.mmax, par.Na)
        if (not hasattr(par, 'm_grid') or par.m_grid is None or len(par.m_grid) < 2 or
            not np.any(np.asarray(par.m_grid) > 0)):
            par.m_grid = par.a_grid.copy()

        Na = par.a_grid.size
        Nfix = par.Nfix if hasattr(par, 'Nfix') else 1
        Nz = par.Nz
        Nh = 4  # rural_renter=0, urban_renter=1, rural_owner=2, urban_owner=3

        if not hasattr(sol, 'a'):
            sol.c = np.zeros((Nfix, Nz, Nh, Na))
            sol.a = np.zeros((Nfix, Nz, Nh, Na))
            sol.pr_choices = np.zeros((Nfix, Nz, Nh, 4, Na))
            sol.pr_urban = np.zeros((Nfix, Nz, Nh, Na))
            sol.pr_choices_asav = np.zeros((Nfix, Nz, Nh, 4, Na))  # z_next-conditional probs on savings grid

        # b. Price getter frpom path or steady state
        _kappa_orig = par.kappa
        if prices_path is None:
            def get_prices(t):
                f_u = float(getattr(self.ss, 'f_u', self.ss.q_u * ((1.0 + self.ss.r) - ((1- par.theta) * (1.0 - par.delta_H)))))
                f_r = float(getattr(self.ss, 'f_r', self.ss.q_r * ((1.0 + self.ss.r) - ((1- par.theta) * (1.0 - par.delta_H)))))
                return self.ss.r, self.ss.w, self.ss.q_u, self.ss.q_r, f_u, f_r
            get_kappa = lambda t: _kappa_orig
        else:
            def get_prices(t):
                r_t = prices_path['r'][t] if prices_path['r'].ndim == 1 else prices_path['r'][t, 0]
                w_t = prices_path['w'][t] if prices_path['w'].ndim == 1 else prices_path['w'][t, 0]
                q_u_t = prices_path['q_u'][t] if prices_path['q_u'].ndim == 1 else prices_path['q_u'][t, 0]
                q_r_t = prices_path['q_r'][t] if prices_path['q_r'].ndim == 1 else prices_path['q_r'][t, 0]
                if 'f_u' in prices_path and 'f_r' in prices_path:
                    f_u_t = prices_path['f_u'][t] if prices_path['f_u'].ndim == 1 else prices_path['f_u'][t, 0]
                    f_r_t = prices_path['f_r'][t] if prices_path['f_r'].ndim == 1 else prices_path['f_r'][t, 0]
                else:
                    f_u_t = float(q_u_t * ((1.0 + r_t) - ((1- par.theta) * (1.0 - par.delta_H))))
                    f_r_t = float(q_r_t * ((1.0 + r_t) - ((1- par.theta) * (1.0 - par.delta_H))))
                return r_t, w_t, q_u_t, q_r_t, f_u_t, f_r_t
            def get_kappa(t):
                if 'kappa' not in prices_path:
                    return _kappa_orig
                kappa_arr = prices_path['kappa']
                return float(kappa_arr[t] if np.asarray(kappa_arr).ndim == 1 else kappa_arr[t, 0])

        z_grid = np.asarray(par.z_grid)
        z_trans_fix = np.asarray(par.z_trans[0] if np.asarray(par.z_trans).ndim == 3 else par.z_trans)

        r, w, q_u, q_r, f_u, f_r = get_prices(it)
        par.kappa = get_kappa(it)
        savingsgrid = par.a_grid
        m_grid_regular = par.m_grid
        J_life = int(getattr(par, 'J', par.T))
        terminal_it = J_life - 1

        # c. calling either terminal or interior period solver
        if it == terminal_it and not force_non_terminal:
            self._solve_terminal_period(
                it, policy_endog, value_endog, m_grid_regular,
                sol, Nfix, Nz, Na
            )
        else:
            if not all(k in policy_endog for k in range(4)):
                raise RuntimeError('policy_endog must include housing keys 0, 1, 2, 3')
            if any((it + 1) not in policy_endog[k] for k in range(4)):
                raise RuntimeError(
                    f'policy_endog missing it+1={it+1} for one-step solve '
                    f'(it={it}, terminal_it={terminal_it}, J={J_life}, T={par.T})'
                )

            self._solve_non_terminal_period(
                it, policy_endog, value_endog, savingsgrid, m_grid_regular,
                z_grid, z_trans_fix, r, w, q_u, q_r, f_u, f_r,
                sol, Nfix, Nz, Na
            )

        par.kappa = _kappa_orig  # restore after one-step solve
        return policy_endog, value_endog


    def _solve_terminal_period(self, it, policy_endog, value_endog, m_grid_regular,
                               sol, Nfix, Nz, Na):
        """Terminal-period policy: consume everything.

        Housing states: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner.
        In the terminal period every agent consumes all cash-on-hand and the uniform
        prior over the 4 choices is used (no future value to differentiate).
        """
        par = self.par
        Nh = 4

        # a. loop over current housing states
        for i_h in range(Nh):
            # Equal probability across all 4 choices in terminal period
            pr_choices_term = np.full((4, 2), 0.25)
            policy_endog[i_h][it] = {}
            value_endog[i_h][it] = {}
            for i_z in range(Nz):
                policy_endog[i_h][it][i_z] = {
                    "m": np.array([0.0, par.mmax]),
                    "c": np.array([0.0, par.mmax]),
                    "pr_choices": pr_choices_term,
                }
                value_endog[i_h][it][i_z] = {
                    "m": np.array([0.0, par.mmax]),
                    "v": np.array([0.0, self._utility(par.mmax, i_h)]),
                }

            # b. populate solution arrays — vectorised over (i_fix, i_z, i_m)
            # Terminal policy is z-independent, so the same (m,c) applies to all i_z.
            pol0 = policy_endog[i_h][it][0]   # any i_z gives the same result
            c_reg = np.interp(m_grid_regular, pol0["m"], pol0["c"])   # (Na,)
            a_reg = m_grid_regular - c_reg
            sol.c[:, :, i_h, :]            = c_reg        # broadcasts over Nfix, Nz
            sol.a[:, :, i_h, :]            = a_reg
            sol.pr_choices[:, :, i_h, :, :] = 0.25        # uniform over 4 choices
            sol.pr_urban[:, :, i_h, :]     = 0.5          # states 1+3 = 50%


    # ==========================================================================
    # ALGORITHM 2 from the appending: EGM step
    # ==========================================================================

    def _solve_non_terminal_period(self, it, policy_endog, value_endog, savingsgrid,
                                   m_grid_regular, z_grid, z_trans_fix,
                                   r, w, q_u, q_r, f_u, f_r, sol, Nfix, Nz, Na):
        """ The "clean" Interior DC-EGM step for a single period.

        Housing states: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner.
        Agents draw taste shocks over all four (region x tenure) choices.

        Budget constraints are in cash_on_hand functions 

        z_trans_fix: (Nz, Nz) matrix of transition probabilities; row i gives P(z'|z=i).
        Policies are stored conditionally on current z-state, giving genuine
        heterogeneity in savings and housing choice across productivity levels.
        """
        par = self.par
        tol_mono = 1e-10
        Nh = 4  # rural_renter=0, urban_renter=1, rural_owner=2, urban_owner=3
        chi_j_next = float(par.chi[it + 1]) if hasattr(par, 'chi') else 0.0

        Na_sav    = savingsgrid.size
        Nz_states = z_grid.size

        # base_az[a, z'] = (1+r)*a + w*exp(chi+z')  — next-period resources before housing costs
        base_az = (
            (1.0 + r) * savingsgrid[:, None]
            + w * np.exp(chi_j_next + z_grid[None, :])
        )  # (Na_sav, Nz_states)

        # a. loop over current housing states
        for i_h_current in range(Nh):

            policy_endog[i_h_current][it] = {}
            value_endog[i_h_current][it] = {}

            # ── a. Pre-compute next-period values (independent of z_current) ──────────
            c_next_all = np.empty((4, Na_sav, Nz_states))
            v_next_all = np.empty((4, Na_sav, Nz_states))

            for i_h_next in range(4):
                if (it + 1) not in policy_endog[i_h_next]:
                    raise RuntimeError(
                        f"policy_endog[{i_h_next}][{it+1}] missing. it={it}"
                    )
                housing_cost, adj = self._cash_on_hand_offsets(
                    i_h_current, i_h_next, r, w, q_u, q_r, f_u, f_r
                )
                m_az = np.maximum(par.cfloor, base_az + adj - housing_cost)  # (Na, Nz)

                # Use z'-conditional policy for each column (z') of m_az
                for i_z_next in range(Nz_states):
                    pol_n = policy_endog[i_h_next][it + 1][i_z_next]
                    val_n = value_endog[i_h_next][it + 1][i_z_next]
                    c_next_all[i_h_next, :, i_z_next] = np.interp(
                        m_az[:, i_z_next], pol_n["m"], pol_n["c"]
                    )
                    v_next_all[i_h_next, :, i_z_next] = np.interp(
                        m_az[:, i_z_next], val_n["m"], val_n["v"]
                    )

            # Logit choice probabilities (softmax over choices, axis=0): (4, Na, Nz)
            if par.sigma <= np.finfo(float).eps:
                best   = np.argmax(v_next_all, axis=0)        # (Na, Nz)
                pr_all = np.zeros((4, Na_sav, Nz_states))
                for k in range(4):
                    pr_all[k] = (best == k).astype(float)
                logsum_az = np.max(v_next_all, axis=0)         # (Na, Nz)
            else:
                v_max     = np.max(v_next_all, axis=0, keepdims=True)       # (1, Na, Nz)
                exp_v     = np.exp((v_next_all - v_max) / par.sigma)        # (4, Na, Nz)
                sum_exp   = np.sum(exp_v, axis=0, keepdims=True)            # (1, Na, Nz)
                pr_all    = exp_v / sum_exp                                  # (4, Na, Nz)
                logsum_az = v_max[0] + par.sigma * np.log(sum_exp[0])       # (Na, Nz)

            # Store z_next-conditional choice probabilities on the savings grid.
            # pr_all[k, i_a, i_z_next] = P(h_next=k | savings=a[i_a], realized z'=i_z_next).
            # Shape (4, Na_sav, Nz) → stored as sol.pr_choices_asav[:, Nz_next, Nh_curr, 4, Na_sav].
            sol.pr_choices_asav[:, :, i_h_current, :, :] = pr_all.transpose(2, 0, 1)[None, ...]

            # Marginal utilities of next-period consumption: (4, Na, Nz)
            mu_c_all = np.power(np.maximum(c_next_all, par.cfloor), -par.rho)

            # ── b. Loop over current z-states: one EGM per (i_h_current, i_z_current) ──
            for i_z_current in range(Nz_states):

                # Transition weights P(z'|z_current) — row of the transition matrix
                z_weights_iz = z_trans_fix[i_z_current, :]  # shape (Nz,)

                # Conditional z-weighted aggregation
                # Expected marginal utility (sum over choices and z'): (Na,)
                rhs_euler_a = (
                    np.einsum('z,haz->a', z_weights_iz, pr_all * mu_c_all) * (1.0 + r)
                )
                # Expected value (sum over z'): (Na,)
                ev_next_a = logsum_az @ z_weights_iz
                # Expected choice probabilities (sum over z'): (4, Na)
                pr_choices_avg_a = np.einsum('z,haz->ha', z_weights_iz, pr_all)

                # Endogenous grid
                c_grid = self._inverse_marginal_utility(par.beta * rhs_euler_a)  # (Na,)
                m_grid = savingsgrid + c_grid                                      # (Na,)
                v_grid = self._utility(c_grid, i_h_current) + par.beta * ev_next_a

                # Borrowing-constraint values (from the lowest savings grid point)
                ev_next_at_bc    = float(ev_next_a[0])
                pr_choices_at_bc = pr_choices_avg_a[:, 0].copy()  # (4,)

                # pr_choices_grid: shape (Na, 4) to match downstream UE code
                pr_choices_grid = pr_choices_avg_a.T   # (Na, 4)

                if np.any(np.diff(m_grid) <= tol_mono):
                    m_grid, c_grid, v_grid, pr_choices_grid = self._apply_upper_envelope(
                        m_grid, c_grid, v_grid, pr_choices_grid, tol=tol_mono
                    )
                    row_sums = pr_choices_grid.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums > 0, row_sums, 1.0)
                    pr_choices_grid = pr_choices_grid / row_sums

                pr_choices_grid_final = pr_choices_grid

                # c. enforce borrowing constraint
                m0  = 0.0
                c0  = 0.0
                v0  = self._utility(c0, i_h_current) + par.beta * ev_next_at_bc
                pr0 = pr_choices_at_bc.copy()
                s0  = pr0.sum()
                pr0 = pr0 / s0 if s0 > 0.0 else np.full(4, 0.25)

                if len(m_grid) > 0 and m_grid[0] <= tol_mono:
                    m_grid[0] = m0
                    c_grid[0] = c0
                    v_grid[0] = v0
                    pr_choices_grid_final[0] = pr0
                else:
                    m_grid = np.concatenate(([m0], m_grid))
                    c_grid = np.concatenate(([c0], c_grid))
                    v_grid = np.concatenate(([v0], v_grid))
                    pr_choices_grid_final = np.vstack([pr0[None, :], pr_choices_grid_final])

                pr_ch_T = pr_choices_grid_final.T  # (4, N_grid)

                # d. store endogenous grid solution for this (i_h_current, i_z_current)
                policy_endog[i_h_current][it][i_z_current] = {
                    "m": m_grid,
                    "c": c_grid,
                    "pr_choices": pr_ch_T,  # shape (4, N_endog_grid)
                }
                value_endog[i_h_current][it][i_z_current] = {
                    "m": m_grid,
                    "v": v_grid,
                }

                # e. populate sol arrays for this (i_z_current, i_h_current)
                c_reg = np.interp(m_grid_regular, m_grid, c_grid)
                a_reg = m_grid_regular - c_reg
                sol.c[:, i_z_current, i_h_current, :]              = c_reg
                sol.a[:, i_z_current, i_h_current, :]              = a_reg

                # f. Interpolate the DC-EGM probabilities from endogenous to regular grid.
                pr_ch_reg = np.empty((4, Na))
                for k in range(4):
                    pr_ch_reg[k, :] = np.interp(m_grid_regular, m_grid, pr_ch_T[k])

                pr_ch_reg = np.clip(pr_ch_reg, 0.0, 1.0)
                col_sums = np.sum(pr_ch_reg, axis=0, keepdims=True)
                col_sums = np.where(col_sums > 0.0, col_sums, 1.0)
                pr_ch_reg = pr_ch_reg / col_sums

                sol.pr_choices[:, i_z_current, i_h_current, :, :] = pr_ch_reg
                sol.pr_urban[:, i_z_current, i_h_current, :]       = np.minimum(
                    pr_ch_reg[1] + pr_ch_reg[3], 1.0
                )

      


    def _apply_upper_envelope(self, m_grid, c_grid, v_grid, pr_choices_grid, tol=1e-10):
        """Apply upper-envelope correction for non-monotone DC-EGM grids.

        Removes suboptimal choices by constructing the upper envelope of the
        piecewise-linear value function and returning associated policy points.

        pr_choices_grid : (N, 4) array of choice probabilities.  Each choice is
            carried as a passenger through the same segment-selection logic used
            for the value function — no separate interpolation from the (possibly
            non-monotone) pre-UE grid is needed.
        """

        m  = np.asarray(m_grid,       dtype=float)
        c  = np.asarray(c_grid,       dtype=float)
        v  = np.asarray(v_grid,       dtype=float)
        pr = np.asarray(pr_choices_grid, dtype=float)   # (N, 4)

        finite_mask = (
            np.isfinite(m) & np.isfinite(c) & np.isfinite(v)
            & np.all(np.isfinite(pr), axis=1)
        )
        m  = m[finite_mask]
        c  = c[finite_mask]
        v  = v[finite_mask]
        pr = pr[finite_mask]   # (N_finite, 4)

        if m.size <= 1:
            return m, c, v, pr

        segments   = []
        candidates = []

        for i in range(m.size - 1):
            m0 = float(m[i])
            m1 = float(m[i + 1])

            if m1 <= m0 + tol:
                continue

            dm  = m1 - m0
            s_c  = (float(c[i + 1]) - float(c[i])) / dm
            s_v  = (float(v[i + 1]) - float(v[i])) / dm
            s_pr = (pr[i + 1] - pr[i]) / dm   # shape (4,)

            i_c  = float(c[i])  - s_c  * m0
            i_v  = float(v[i])  - s_v  * m0
            i_pr = pr[i]        - s_pr * m0   # shape (4,)

            segments.append((m0, m1, s_c, i_c, s_v, i_v, s_pr.copy(), i_pr.copy()))
            candidates.extend([m0, m1])

        if len(segments) == 0:
            order = np.argsort(m)
            return m[order], c[order], v[order], pr[order]

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                m0_i, m1_i, _, _, s_v_i, i_v_i, _, _ = segments[i]
                m0_j, m1_j, _, _, s_v_j, i_v_j, _, _ = segments[j]

                lo = max(m0_i, m0_j)
                hi = min(m1_i, m1_j)
                if hi <= lo + tol:
                    continue

                denom = s_v_i - s_v_j
                if np.abs(denom) <= tol:
                    continue

                x_star = (i_v_j - i_v_i) / denom
                if (x_star > lo + tol) and (x_star < hi - tol):
                    candidates.append(float(x_star))

        candidates = np.array(sorted(candidates), dtype=float)
        if candidates.size == 0:
            order = np.argsort(m)
            return m[order], c[order], v[order], pr[order]

        m_nodes = [candidates[0]]
        for x in candidates[1:]:
            if x - m_nodes[-1] > tol:
                m_nodes.append(float(x))

        m_env  = []
        c_env  = []
        v_env  = []
        pr_env = []

        for x in m_nodes:
            best_val = -np.inf
            best_c   = np.nan
            best_pr  = np.full(4, np.nan)

            for m0, m1, s_c, i_c, s_v, i_v, s_pr, i_pr in segments:
                if (x < m0 - tol) or (x > m1 + tol):
                    continue

                v_x = s_v * x + i_v
                if v_x > best_val:
                    best_val = v_x
                    best_c   = s_c  * x + i_c
                    best_pr  = s_pr * x + i_pr   # shape (4,)

            if np.isfinite(best_val):
                m_env.append(x)
                c_env.append(best_c)
                v_env.append(best_val)
                pr_env.append(np.clip(best_pr, 0.0, 1.0))   # (4,)

        m_env  = np.asarray(m_env,  dtype=float)
        c_env  = np.asarray(c_env,  dtype=float)
        v_env  = np.asarray(v_env,  dtype=float)
        pr_env = np.asarray(pr_env, dtype=float)   # (M, 4)

        if m_env.size <= 1:
            order = np.argsort(m)
            return m[order], c[order], v[order], pr[order]

        keep = np.ones(m_env.size, dtype=bool)
        for i in range(1, m_env.size):
            if m_env[i] - m_env[i - 1] <= tol:
                if v_env[i] > v_env[i - 1]:
                    keep[i - 1] = False
                else:
                    keep[i] = False

        return m_env[keep], c_env[keep], v_env[keep], pr_env[keep]



    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    def _utility(self, consumption, i_h):
        """Utility from consumption and housing service flow.

        Housing states: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner.
        Urban states (1, 3) receive the kappa preference shifter.
        Renters (states 0, 1) get housing service h_l; owners get h_r (rural) or h_u (urban).
        """
        par = self.par

        # a. Housing service flow
        is_urban  = (i_h % 2 == 1)   # states 1 and 3 are urban
        is_owner  = (i_h >= 2)        # states 2 and 3 are owners
        if is_owner:
            h = par.h_u if is_urban else par.h_r
        else:
            h = float(getattr(par, 'h_l', par.h_r))  # rental unit size

        # b. Consumption utility (CRRA)
        if par.rho == 1:
            u_c = np.log(np.maximum(consumption, par.cfloor))
        else:
            u_c = (np.power(np.maximum(consumption, par.cfloor), 1 - par.rho) - 1) / (1 - par.rho)

        # c. Housing utility (CRRA)
        if par.gamma == 1:
            u_h = np.log(h)
        else:
            u_h = (np.power(h, 1 - par.gamma) - 1) / (1 - par.gamma)

        # d. Urban preference shifter (applies to urban states 1 and 3)
        u = u_c + u_h
        if is_urban:
            u += par.kappa

        return u

    def _marginal_utility(self, consumption, i_h):
        """Marginal utility of consumption.

        Note: i_h is the housing state (0-3); currently marginal utility
        depends only on consumption (separable preferences).
        """
        par = self.par

        if par.rho == 1:
            mu = 1.0 / np.maximum(consumption, par.cfloor)
        else:
            mu = np.power(np.maximum(consumption, par.cfloor), -par.rho)

        return mu

    def _inverse_marginal_utility(self, mu):
        """Inverse of marginal utility of consumption.
        """
        par = self.par
        
        if par.rho == 1:
            return 1.0 / mu
        else:
            return np.power(mu, -1.0 / par.rho)

    def _choice_prob(self, v_array):
        """Softmax choice probabilities using logsum trick.

        v_array: shape (N_choices, 1) where N_choices can be 2 or 4.
        Returns 1-D array of N_choices probabilities.
        """
        par = self.par

        if par.sigma <= np.finfo(float).eps:
            # Deterministic choice: argmax
            idx_max = np.argmax(v_array[:, 0])
            pr = np.zeros(v_array.shape[0])
            pr[idx_max] = 1.0
            return pr

        # Numerically stable softmax
        v_max = np.max(v_array)
        exp_v = np.exp((v_array - v_max) / par.sigma)
        pr = exp_v[:, 0] / np.sum(exp_v[:, 0])

        return pr

    def _logsum(self, v_array):
        """Logsum-exp for computing expected value.
        """
        par = self.par
        
        if par.sigma <= np.finfo(float).eps:
            return np.max(v_array, axis=0)
        
        v_max = np.max(v_array, axis=0)
        logsum = v_max + par.sigma * np.log(
            np.sum(np.exp((v_array - v_max) / par.sigma), axis=0)
        )
        
        return logsum

    def _cash_on_hand_offsets(self, i_h_current, i_h_next, r, w, q_u, q_r, f_u, f_r):
        """Return scalar (housing_cost, adj) for use in vectorised EGM.

        housing_cost: ongoing period cost (rent or mortgage payment).
        adj: one-time surplus from selling/buying, following Schang (2025):
          Selling (owner exits):    +(1 - delta_H - zeta - lambda) * q_curr * h_curr
          Buying (enter ownership): -(1 + zeta - lambda) * q_next * h_next
        Plus a renter migration cost zeta_renter * w when a renter changes
        region (rural_renter <-> urban_renter).

        These scalars are independent of (a, z) so they can be lifted out of the
        inner loops and applied by broadcasting over the (Na, Nz) base array.
        """
        par = self.par
        is_urban_next = (i_h_next % 2 == 1)
        is_owner_next = (i_h_next >= 2)
        q_next = q_u if is_urban_next else q_r
        h_next = par.h_u if is_urban_next else par.h_r
        f_next = f_u if is_urban_next else f_r

        h_l = float(getattr(par, 'h_l', par.h_r))
        housing_cost = (
            par.lambda_ltv * q_next * h_next * (r + 1.0 / par.T_mort) + par.tau_wealth * q_next * h_next
            if is_owner_next else f_next * h_l
        )

        adj = 0.0
        if i_h_next != i_h_current:
            # Selling proceeds: only if currently an owner
            if i_h_current >= 2:
                is_urban_curr = (i_h_current == 3)
                q_curr = q_u if is_urban_curr else q_r
                h_curr = par.h_u if is_urban_curr else par.h_r
                adj += (1-par.tau_profits) * par.zeta * q_curr * h_curr

            # Buying cost: only if becoming an owner
            if is_owner_next:
                adj -= (1-par.tau_profits) * par.zeta * q_next * h_next

            # Renter migration cost: both states are renters (0,1) and region differs.
            if i_h_current < 2 and i_h_next < 2:
                adj -= float(getattr(par, 'zeta_renter', 0.0)) * w

        return housing_cost, adj

    def _cash_on_hand(self, a_today, z_shock, i_h_current, i_h_next, r, w, q_u, q_r, f_u, f_r, h_l, chi_j=0.0):
        """Compute next-period cash-on-hand.

        Housing states: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner.
        
        MUST match cash_on_hand_numba in steady_state.py exactly.
        """
        par = self.par

        base = (1.0 + r) * a_today + w * np.exp(chi_j + z_shock)

        # --- Ongoing housing cost for next-period state ---
        is_urban_next = (i_h_next % 2 == 1)  # states 1 (urban_renter) and 3 (urban_owner)
        is_owner_next = (i_h_next >= 2)       # states 2 (rural_owner) and 3 (urban_owner)
        q_next = q_u if is_urban_next else q_r
        h_next = par.h_u if is_urban_next else par.h_r
        f_next = f_u if is_urban_next else f_r

        if is_owner_next:
            # Mortgage interest + amortisation on new (or continuing) mortgage at LTV
            housing_cost = par.lambda_ltv * q_next * h_next * (r + 1.0 / par.T_mort) + par.tau_wealth * q_next * h_next
        else:
            # Renter: pay rent to rental sector
            housing_cost = f_next * h_l

        # --- One-time adjustment from buying/selling ---
        adj = 0.0
        if i_h_next != i_h_current:
            # Selling proceeds: only if currently an owner
            if i_h_current >= 2:
                is_urban_curr = (i_h_current == 3)
                q_curr = q_u if is_urban_curr else q_r
                h_curr = par.h_u if is_urban_curr else par.h_r
                adj += (1-par.tau_profits)*par.zeta * q_curr * h_curr

            # Buying cost: only if becoming an owner
            if is_owner_next:
                adj -= (1-par.tau_profits) * par.zeta * q_next * h_next

            # Renter migration cost: both states are renters (0,1) and region differs.
            if i_h_current < 2 and i_h_next < 2:
                adj -= float(getattr(par, 'zeta_renter', 0.0)) * w

        return base - housing_cost + adj


    # =========================================================================
    # Calling the solver and storing results
    # =========================================================================

def solve_hh_backwards(model):
    """Entry point for backward iteration called by steady_state.py.
    
    Solves household problem and directly stores all policies and values
    in model.sol arrays.
    """
    hh = HousingModel(model)
    policy_endog, value_endog = hh.solve_dcegm()
    
    # Store endogenous grid solutions for diagnostics and plotting
    model.hh_policy = policy_endog
    model.hh_value = value_endog




