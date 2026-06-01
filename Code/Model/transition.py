"""
Transition path computation for HANCHousingModel using direct Jacobian method
with accumulation along the DAG/block structure.

Functions:
1. evaluate_path_blocks() - Evaluate all blocks sequentially for transition path
2. compute_jacobian_direct() - Compute household Jacobian using finite differences
3. compute_jacobian_accumulated() - Accumulate Jacobians block-by-block along DAG
"""

import numpy as np
import importlib
import inspect
import warnings
import steady_state
import household_problem
from scipy.sparse import coo_matrix
from numba import njit
from steady_state import build_transition_coo as _build_transition_coo_njit
from steady_state import scatter_dcegm_forward as _scatter_dcegm_forward



def _normalize_cohort_weights(raw_weights, J):
    """Return non-negative cohort weights of length J summing to one."""
    w = np.asarray(raw_weights, dtype=float).ravel()
    if w.size != J:
        w_new = np.zeros(J)
        n = min(J, w.size)
        if n > 0:
            w_new[:n] = w[:n]
        w = w_new
    w = np.maximum(w, 0.0)
    s = np.sum(w)
    if s <= 0.0:
        return np.ones(J) / J
    return w / s


def _get_path_cohort_weights(model, J):
    """Get exogenous cohort weights ω_{j,t}."""
    par = model.par
    path = model.path
    ss = model.ss

    if hasattr(path, 'cohort_weights'):
        raw = np.asarray(path.cohort_weights)
        if raw.ndim == 1:
            w0 = _normalize_cohort_weights(raw, J)
            return np.tile(w0[None, :], (par.T, 1))
        if raw.ndim == 2 and raw.shape[0] >= par.T:
            out = np.zeros((par.T, J))
            for t in range(par.T):
                out[t] = _normalize_cohort_weights(raw[t], J)
            return out

    if hasattr(ss, 'cohort_weights'):
        w0 = _normalize_cohort_weights(ss.cohort_weights, J)
    elif hasattr(par, 'cohort_weights_ss'):
        w0 = _normalize_cohort_weights(par.cohort_weights_ss, J)
    else:
        w0 = np.ones(J) / J

    return np.tile(w0[None, :], (par.T, 1))


def _get_newborn_distribution(par, ss, Nz, Nh, Nm):
    """Return normalized newborn distribution over (z,h,m).

    With 4 housing states (0=rural_renter,1=urban_renter,2=rural_owner,3=urban_owner),
    newborns enter as renters (states 0 and 1).
    """
    if hasattr(ss, 'D_birth') and isinstance(ss.D_birth, np.ndarray) and ss.D_birth.shape == (Nz, Nh, Nm):
        D_birth = ss.D_birth.copy()
        s = np.sum(D_birth)
        if s > 0.0 and np.isfinite(s):
            return D_birth / s

    D_birth = np.zeros((Nz, Nh, Nm))
    z_erg = par.z_ergodic[0] if par.z_ergodic.ndim == 2 else par.z_ergodic
    newborn_urban_share = float(np.clip(getattr(par, 'newborn_urban_share', 0.0), 0.0, 1.0))
    # Newborns are renters (no prior ownership)
    D_birth[:, 0, 0] = z_erg * (1.0 - newborn_urban_share)  # rural_renter
    D_birth[:, 1, 0] = z_erg * newborn_urban_share            # urban_renter
    s = np.sum(D_birth)
    if s > 0.0:
        D_birth /= s
    return D_birth


def _initialize_cohort_distribution(model, J, Nz, Nh, Nm):
    """Initialize age-cohort distributions at date 0.

    Permanent-transition override: if ``model.ini.D_cohort`` (or
    ``model.ini.D``) is supplied, use it instead of the terminal-SS
    distribution.  This is what makes the transition actually start from
    the initial-SS cross-section instead of the terminal one.
    """
    ss = model.ss
    ini = getattr(model, 'ini', None)
    D_birth = _get_newborn_distribution(model.par, ss, Nz, Nh, Nm)

    src = None
    if ini is not None and hasattr(ini, 'D_cohort') \
            and isinstance(ini.D_cohort, np.ndarray) and ini.D_cohort.shape == (J, Nz, Nh, Nm):
        src = ini.D_cohort
    elif hasattr(ss, 'D_cohort') and isinstance(ss.D_cohort, np.ndarray) and ss.D_cohort.shape == (J, Nz, Nh, Nm):
        src = ss.D_cohort

    if src is not None:
        D0 = src.copy()
        for j in range(J):
            s_j = np.sum(D0[j])
            if s_j > 0.0 and np.isfinite(s_j):
                D0[j] /= s_j
            else:
                D0[j] = D_birth
        return D0, D_birth

    D0 = np.zeros((J, Nz, Nh, Nm))
    D0[0] = D_birth
    D_src = None
    if ini is not None and hasattr(ini, 'D') and isinstance(ini.D, np.ndarray) and ini.D.shape == (Nz, Nh, Nm):
        D_src = ini.D
    elif hasattr(ss, 'D') and isinstance(ss.D, np.ndarray) and ss.D.shape == (Nz, Nh, Nm):
        D_src = ss.D
    if D_src is not None:
        D_ss = D_src.copy()
        s = np.sum(D_ss)
        if s > 0.0 and np.isfinite(s):
            D_ss /= s
            for j in range(1, J):
                D0[j] = D_ss
        else:
            for j in range(1, J):
                D0[j] = D_birth
    else:
        for j in range(1, J):
            D0[j] = D_birth

    return D0, D_birth


def _policy_arrays_from_endog(policy_endog, age_idx, par, Nh):
    """Interpolate policy_endog at one age to regular (Nz,Nh,Na) arrays.

    Returns (a_pol, c_pol, pr_choices_pol) where pr_choices_pol has shape (4, Nz, Nh, Na).
    """
    a_pol = np.zeros((par.Nz, Nh, par.Na))
    c_pol = np.zeros((par.Nz, Nh, par.Na))
    pr_choices_pol = np.zeros((4, par.Nz, Nh, par.Na))  # 4-way choice probs

    for i_h in range(Nh):
        pol_i_h = policy_endog.get(i_h, {})
        if age_idx in pol_i_h:
            pol_age = pol_i_h[age_idx]
        elif 0 in pol_i_h:
            pol_age = pol_i_h[0]
        else:
            continue

        # Detect z-conditional structure: pol_age = {0: {'m':..., 'c':...}, 1: {...}, ...}
        # vs flat structure: pol_age = {'m':..., 'c':..., ...}
        is_z_conditional = (isinstance(pol_age, dict) and len(pol_age) > 0
                            and isinstance(next(iter(pol_age)), int))

        if is_z_conditional:
            for i_z in range(par.Nz):
                pol = pol_age.get(i_z, pol_age.get(0))
                if pol is None:
                    continue
                c_line = np.interp(par.m_grid, pol['m'], pol['c'])
                a_line = par.m_grid - c_line
                a_pol[i_z, i_h, :] = a_line
                c_pol[i_z, i_h, :] = c_line
                if 'pr_choices' in pol:
                    for k in range(4):
                        pr_choices_pol[k, i_z, i_h, :] = np.interp(
                            par.m_grid, pol['m'], pol['pr_choices'][k]
                        )
        else:
            # Flat (legacy) structure: same policy for all z
            pol = pol_age
            c_line = np.interp(par.m_grid, pol['m'], pol['c'])
            a_line = par.m_grid - c_line
            if 'pr_choices' in pol:
                for k in range(4):
                    pr_choices_pol[k, :, i_h, :] = np.interp(
                        par.m_grid, pol['m'], pol['pr_choices'][k]
                    )
            else:
                # Legacy fallback: pr_rural is a 1-D array
                pr_r_line = np.interp(par.m_grid, pol['m'], pol.get('pr_rural', np.full(len(pol['m']), 0.5)))
                pr_u_line = np.clip(1.0 - pr_r_line, 0.0, 1.0)
                pr_choices_pol[0, :, i_h, :] = pr_r_line  # rural_renter (proxy)
                pr_choices_pol[1, :, i_h, :] = pr_u_line  # urban_renter (proxy)
            for i_z in range(par.Nz):
                a_pol[i_z, i_h, :] = a_line
                c_pol[i_z, i_h, :] = c_line

    return a_pol, c_pol, pr_choices_pol


def _get_path_scalar(arr, t):
    """Read scalar from path array with shape (T,) or (T,1)."""
    return arr[t] if arr.ndim == 1 else arr[t, 0]




def _get_hh_level_scale_path(model):
    """Return multiplicative HH level scale N_t as a length-T array.

    Returns the *level* of hh_scale at each calendar time.  Used by the
    non-linear demographic forward simulation to scale cohort aggregates.

    If model.path.hh_scale is populated (i.e. hh_scale is in shocks and the
    shock path has been written), returns SS_level + shock_deviation.
    Otherwise returns the SS level (par.hh_scale or ss.hh_scale) at all t.
    """
    ss_level = float(getattr(model.ss, 'hh_scale',
                             getattr(model.par, 'hh_scale', 1.0)))
    if hasattr(model.path, 'hh_scale'):
        arr = np.asarray(model.path.hh_scale)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return np.maximum(arr.ravel()[:model.par.T], 1e-12)
    return np.full(model.par.T, max(ss_level, 1e-12))


# ==========================================================================
# Forward pass for evaluate_path_blocks
# ==========================================================================

def _simulate_olg_path(model):
    """Forward pass for evaluate_path_blocks (non-linear branch).

    OLG:
      1. Solves DC-EGM ONCE on the (T, J) calendar-age grid via
         solve_dcegm_calendar_age_grid — a single backward sweep yields
         policies a_arr[ct, age], c_arr[ct, age], pr_arr[ct, age].
      2. Forward-simulates each cohort (pre-transition with initial age j0 > 0
         and post-transition newborns born at calendar time b >= 0) with the
         njit scatter kernel _simulate_cohort_njit — no sparse matrices are
         materialised, so this is ~5-20x faster than a per-(cal_t, age)
         scipy COO build + matvec.
      3. Aggregates with cohort_weights to produce path.{A_hh, C_hh, H_u_hh,
         H_r_hh} as proper OLG aggregates over alive cohorts.
      4. Populates path.D as the cohort-weighted aggregate distribution.
    """
    par = model.par
    path = model.path
    Nh = par.Nh if hasattr(par, 'Nh') else 2

    J = int(getattr(par, 'J', par.T))
    J = max(1, min(J, par.T))
    T = int(par.T)

    omega_path = np.ascontiguousarray(
        _get_path_cohort_weights(model, J), dtype=np.float64)
    hh_scale = np.ascontiguousarray(
        _get_hh_level_scale_path(model), dtype=np.float64)
    D0_cohort, D_birth = _initialize_cohort_distribution(model, J, par.Nz, Nh, par.Na)
    D_birth = np.ascontiguousarray(D_birth, dtype=np.float64)

    # --- Single backward sweep over the (T, J) calendar-age grid -------------
    prices_dict = {}
    for name in ('r', 'w', 'q_u', 'q_r', 'f_u', 'f_r', 'kappa'):
        arr = path.__dict__.get(name)
        if isinstance(arr, np.ndarray) and arr.size > 0:
            prices_dict[name] = np.asarray(arr)

    hh = household_problem.HousingModel(model)
    a_arr, c_arr, pr_arr = hh.solve_dcegm_calendar_age_grid(prices_dict)
    # a_arr, c_arr : (T, J, Nz, 4, Na);  pr_arr : (T, J, 4, Nz, 4, Na)
    a_arr = np.ascontiguousarray(a_arr, dtype=np.float64)
    c_arr = np.ascontiguousarray(c_arr, dtype=np.float64)
    pr_arr = np.ascontiguousarray(pr_arr, dtype=np.float64)

    # --- Grid / scalar constants for the njit kernel -------------------------
    z_grid = np.ascontiguousarray(par.z_grid, dtype=np.float64)
    z_trans = np.ascontiguousarray(
        par.z_trans[0] if par.z_trans.ndim == 3 else par.z_trans, dtype=np.float64)
    m_grid = np.ascontiguousarray(par.m_grid, dtype=np.float64)

    cfloor = float(par.cfloor)
    zeta = float(par.zeta)
    zeta_renter = float(getattr(par, 'zeta_renter', 0.0))
    h_u = float(par.h_u)
    h_r = float(par.h_r)
    h_l = float(getattr(par, 'h_l', par.h_r))
    lambda_ltv = float(par.lambda_ltv)
    T_mort = float(par.T_mort)
    delta_H = float(par.delta_H)
    tau_wealth = float(getattr(par, 'tau_wealth', 0.0))
    tau_profits = float(getattr(par, 'tau_profits', 0.0))

    # --- Full-path price arrays (length T) -----------------------------------
    def _full(name):
        arr = path.__dict__.get(name)
        if isinstance(arr, np.ndarray) and arr.size > 0:
            vec = arr[:, 0] if arr.ndim == 2 else arr.ravel()[:T]
            return np.ascontiguousarray(vec, dtype=np.float64)
        return None

    r_path = _full('r')
    w_path = _full('w')
    q_u_path = _full('q_u')
    q_r_path = _full('q_r')
    f_u_path = _full('f_u')
    f_r_path = _full('f_r')
    if f_u_path is None:
        f_u_path = q_u_path * ((1.0 + r_path) - (1.0 - par.theta) * (1.0 - delta_H))
    if f_r_path is None:
        f_r_path = q_r_path * ((1.0 + r_path) - (1.0 - par.theta) * (1.0 - delta_H))

    chi_full = (np.ascontiguousarray(par.chi, dtype=np.float64)
                if hasattr(par, 'chi') else np.zeros(T))

    # --- Accumulators shared across every cohort kernel call -----------------
    A_hh = np.zeros(T)
    C_hh = np.zeros(T)
    H_u_hh = np.zeros(T)
    H_r_hh = np.zeros(T)
    path_D_4d = np.zeros((T, par.Nz, Nh, par.Na))

    def _run_cohort(D_init, cal_t_list, age_list):
        """Forward-simulate one cohort through its (cal_t, age) schedule."""
        cal_t_arr = np.asarray(cal_t_list, dtype=np.int64)
        age_arr = np.asarray(age_list, dtype=np.int64)

        a_pol_ages = np.ascontiguousarray(a_arr[cal_t_arr, age_arr])
        c_pol_ages = np.ascontiguousarray(c_arr[cal_t_arr, age_arr])
        pr_ch_ages = np.ascontiguousarray(pr_arr[cal_t_arr, age_arr])

        omega_arr = np.ascontiguousarray(omega_path[cal_t_arr, age_arr])
        scale_arr = np.ascontiguousarray(hh_scale[cal_t_arr])
        chi_arr = np.ascontiguousarray(chi_full[np.minimum(age_arr + 1, T - 1)])

        r_a = np.ascontiguousarray(r_path[cal_t_arr])
        w_a = np.ascontiguousarray(w_path[cal_t_arr])
        q_u_a = np.ascontiguousarray(q_u_path[cal_t_arr])
        q_r_a = np.ascontiguousarray(q_r_path[cal_t_arr])
        f_u_a = np.ascontiguousarray(f_u_path[cal_t_arr])
        f_r_a = np.ascontiguousarray(f_r_path[cal_t_arr])

        D_age = np.ascontiguousarray(D_init, dtype=np.float64).copy()
        _simulate_cohort_njit(
            D_age, D_birth,
            a_pol_ages, c_pol_ages, pr_ch_ages,
            cal_t_arr, omega_arr, scale_arr, chi_arr,
            r_a, w_a, q_u_a, q_r_a, f_u_a, f_r_a,
            z_grid, z_trans, m_grid,
            cfloor, zeta, zeta_renter, h_u, h_r, h_l, lambda_ltv, T_mort, delta_H,
            tau_wealth, tau_profits,
            A_hh, C_hh, H_u_hh, H_r_hh, path_D_4d,
        )

    # --- (1) Pre-transition cohorts: at cal_t=0 they are age j0 > 0 ----------
    for j0 in range(1, J):
        max_cal_t = min(J - 1 - j0, T - 1)
        cal_t_list = list(range(max_cal_t + 1))
        age_list = [j0 + ct for ct in cal_t_list]
        _run_cohort(D0_cohort[j0], cal_t_list, age_list)

    # --- (2) Post-transition cohorts: born at cal_t b in [0, T-1] ------------
    for birth in range(T):
        max_age = min(J - 1, T - 1 - birth)
        age_list = list(range(max_age + 1))
        cal_t_list = [birth + a for a in age_list]
        _run_cohort(D_birth, cal_t_list, age_list)

    # --- Write into path -----------------------------------------------------
    totals = {'A_hh': A_hh, 'C_hh': C_hh, 'H_u_hh': H_u_hh, 'H_r_hh': H_r_hh}
    for varname, vals in totals.items():
        arr = path.__dict__.get(varname)
        if arr is None:
            continue
        if arr.ndim == 2:
            arr[:, 0] = vals
        else:
            arr[:] = vals

    # Aggregate distribution per calendar time, broadcast across Nfix if needed
    if (not hasattr(path, 'D') or not isinstance(path.D, np.ndarray)
            or path.D.size == 0):
        path.D = np.zeros((T, par.Nz, Nh, par.Na))
    if path.D.ndim == 5:
        path.D[:, 0] = path_D_4d
        if par.Nfix > 1:
            for fix in range(1, par.Nfix):
                path.D[:, fix] = path_D_4d
    else:
        path.D[:] = path_D_4d


# ==========================================
# 1. Solving and simulating household problem along transition path
# ==========================================

# ──────────────────────────────────────────────────────────────────────────────
# Numba kernel: simulate one cohort forward through (cal_t, age) pairs.
# ──────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def _simulate_cohort_njit(
    D_age,                # (Nz, Nh, Nm)  modified in place across ages
    D_birth,              # (Nz, Nh, Nm)  fallback if mass collapses
    a_pol_ages,           # (n_ages, Nz, Nh, Nm)
    c_pol_ages,           # (n_ages, Nz, Nh, Nm)
    pr_ch_ages,           # (n_ages, 4, Nz, Nh, Nm)
    cal_t_arr,            # (n_ages,) int64
    omega_arr,            # (n_ages,)
    scale_arr,            # (n_ages,)
    chi_arr,              # (n_ages,)  par.chi[min(age+1, T-1)]
    r_arr, w_arr, q_u_arr, q_r_arr, f_u_arr, f_r_arr,   # (n_ages,)
    z_grid, z_trans, m_grid,
    cfloor, zeta, zeta_renter, h_u, h_r, h_l, lambda_ltv, T_mort, delta_H,
    tau_wealth, tau_profits,
    A_hh, C_hh, H_u_hh, H_r_hh,         # (T,) accumulators
    path_D_4d,                          # (T, Nz, Nh, Nm) accumulator
):
    Nz, Nh, Nm = D_age.shape
    n_ages = a_pol_ages.shape[0]

    D_next = np.empty((Nz, Nh, Nm))

    for age in range(n_ages):
        cal_t = cal_t_arr[age]
        w_age = omega_arr[age]
        s_t = scale_arr[age]

        if w_age > 0.0:
            sum_a = 0.0
            sum_c = 0.0
            sum_urb_renter = 0.0
            sum_urb_owner  = 0.0
            sum_rur_renter = 0.0
            sum_rur_owner  = 0.0
            for iz in range(Nz):
                for ih in range(Nh):
                    for im in range(Nm):
                        d = D_age[iz, ih, im]
                        sum_a += d * a_pol_ages[age, iz, ih, im]
                        sum_c += d * c_pol_ages[age, iz, ih, im]
                        if ih == 1:
                            sum_urb_renter += d
                        elif ih == 3:
                            sum_urb_owner += d
                        elif ih == 0:
                            sum_rur_renter += d
                        else:
                            sum_rur_owner += d

            sw = s_t * w_age
            A_hh[cal_t] += sw * sum_a
            C_hh[cal_t] += sw * sum_c
            H_u_hh[cal_t] += sw * (sum_urb_renter * h_l + sum_urb_owner * h_u)
            H_r_hh[cal_t] += sw * (sum_rur_renter * h_l + sum_rur_owner * h_r)

            for iz in range(Nz):
                for ih in range(Nh):
                    for im in range(Nm):
                        path_D_4d[cal_t, iz, ih, im] += w_age * D_age[iz, ih, im]

        if age == n_ages - 1:
            break

        # Direct scatter forward; chi shifted by 1 as in original.
        _scatter_dcegm_forward(
            D_age, D_next,
            a_pol_ages[age], pr_ch_ages[age], z_trans, z_grid, m_grid,
            r_arr[age], w_arr[age] * np.exp(chi_arr[age]),
            q_u_arr[age], q_r_arr[age], cfloor,
            zeta, zeta_renter, h_u, h_r, h_l,
            f_u_arr[age], f_r_arr[age],
            lambda_ltv, T_mort, delta_H,
            tau_wealth, tau_profits,
        )

        s_age = 0.0
        for iz in range(Nz):
            for ih in range(Nh):
                for im in range(Nm):
                    s_age += D_next[iz, ih, im]

        if s_age > 0.0 and np.isfinite(s_age):
            inv_s = 1.0 / s_age
            for iz in range(Nz):
                for ih in range(Nh):
                    for im in range(Nm):
                        D_age[iz, ih, im] = D_next[iz, ih, im] * inv_s
        else:
            for iz in range(Nz):
                for ih in range(Nh):
                    for im in range(Nm):
                        D_age[iz, ih, im] = D_birth[iz, ih, im]


def simulate_hh_path(model, do_print=False):
    """Simulate distribution forward and compute aggregates along transition path.
    
    Uses time-varying policy functions from path.sol_* arrays and forward
    simulates the distribution, computing aggregates at each date.
    
    Args:
        model: HANCHousingModel with:
            - path.sol_*[t]: policy functions at each date (input)
            - ss.D: initial distribution
            - path.A_hh[t], path.C_hh[t], etc.: aggregates (output)
    """
    par = model.par
    path = model.path
    ss = model.ss
    Nh = par.Nh if hasattr(par, 'Nh') else 2

    # Allocate distribution arrays if needed
    if not hasattr(path, 'D'):
        path.D = np.zeros((par.T, par.Nz, Nh, par.Na))
    
    # Initialize distribution
    if path.D.ndim == 5:
        path.D[0, 0] = ss.D.copy()
    else:
        path.D[0] = ss.D.copy()

    # Discretized productivity states
    z_grid = np.asarray(par.z_grid)

    # Unpack transition matrix for productivity
    z_trans = par.z_trans[0] if par.z_trans.ndim == 3 else par.z_trans

    # Simulate forward using transition operator
    Nstate = par.Nz * Nh * par.Na
    max_transitions = Nstate * par.Nz * 4 * 2 + 1000  # 4 choices

    for t in range(1, par.T):
        # Policies at t-1 (use first fixed type if present)
        a_policy = path.sol_a[t - 1, 0]
        # Build (4, Nz, Nh, Nm) pr_choices from marginal pr_urban (equal renter/owner split)
        _pr_u = path.sol_pr_urban[t - 1, 0]
        pr_choices_t = np.zeros((4, par.Nz, Nh, par.Na))


        # use actual DC-EGM choice probabilities (shape: Nfix, Nz, Nh, 4, Na)
        pr_choices_t = path.sol_pr_choices[t - 1, 0]   # (Nz, Nh, 4, Na)
        # build_transition_coo expects (4, Nz, Nh, Na) — transpose:
        pr_choices_t = np.moveaxis(pr_choices_t, 2, 0)  # (4, Nz, Nh, Na)

        # Prices at t-1
        r_t = path.r[t - 1] if path.r.ndim == 1 else path.r[t - 1, 0]
        w_t = path.w[t - 1] if path.w.ndim == 1 else path.w[t - 1, 0]
        q_u_t = path.q_u[t - 1] if path.q_u.ndim == 1 else path.q_u[t - 1, 0]
        q_r_t = path.q_r[t - 1] if path.q_r.ndim == 1 else path.q_r[t - 1, 0]

        rows, cols, data, _ = steady_state.build_transition_coo(
            a_policy, pr_choices_t, z_trans, z_grid, par.m_grid,
            r_t, w_t * (np.exp(float(par.chi[t])) if hasattr(par, 'chi') else 1.0), q_u_t, q_r_t, par.cfloor,
            par.zeta, float(getattr(par, 'zeta_renter', 0.0)), par.h_u, par.h_r, float(getattr(par, 'h_l', par.h_r)),
            float(path.f_u[t - 1] if path.f_u.ndim == 1 else path.f_u[t - 1, 0]) if hasattr(path, 'f_u') else q_u_t * ((1.0 + r_t) - (1- par.theta) * (1.0 - par.delta_H)),
            float(path.f_r[t - 1] if path.f_r.ndim == 1 else path.f_r[t - 1, 0]) if hasattr(path, 'f_r') else q_r_t * ((1.0 + r_t) - (1- par.theta) * (1.0 - par.delta_H)),
            par.lambda_ltv, float(par.T_mort), float(par.delta_H),
            float(getattr(par, 'tau_wealth', 0.0)), float(getattr(par, 'tau_profits', 0.0)),
            max_transitions
        )
        P = coo_matrix((data, (rows, cols)), shape=(Nstate, Nstate)).tocsr()

        if path.D.ndim == 5:
            D_prev = path.D[t - 1, 0].reshape(-1)
            D_next = P @ D_prev
            D_next = D_next.reshape((par.Nz, Nh, par.Na))
            path.D[t, 0] = D_next
            if par.Nfix > 1:
                path.D[t, 1:] = D_next
        else:
            D_prev = path.D[t - 1].reshape(-1)
            D_next = P @ D_prev
            path.D[t] = D_next.reshape((par.Nz, Nh, par.Na))
    
    # Compute aggregates at each date
    if not hasattr(path, 'A_hh'):
        path.A_hh = np.zeros(par.T)
        path.C_hh = np.zeros(par.T)
        path.H_u_hh = np.zeros(par.T)
        path.H_r_hh = np.zeros(par.T)
    
    for t in range(par.T):
        # Use policy functions and distribution at date t
        if path.D.ndim == 5:
            D_t = path.D[t]
        else:
            D_t = path.D[t][None, :, :, :]

        path.A_hh[t] = np.sum(D_t * path.sol_a[t])
        path.C_hh[t] = np.sum(D_t * path.sol_c[t])
        # Urban housing: urban_renter (state 1) AND urban_owner (state 3)
        _h_l = float(getattr(par, 'h_l', par.h_r))
        path.H_u_hh[t] = np.sum(D_t[:, :, 1, :]) * _h_l + np.sum(D_t[:, :, 3, :]) * par.h_u
        # Rural housing: rural_renter (state 0) AND rural_owner (state 2)
        path.H_r_hh[t] = np.sum(D_t[:, :, 0, :]) * _h_l + np.sum(D_t[:, :, 2, :]) * par.h_r








def compute_wealth_inequality(model, T_plot=None):
    """Compute wealth inequality statistics from the simulated path distributions.

    Reads path.D[t] (shape T×Nz×Nh×Na), marginalises over productivity and
    housing-type dimensions to obtain a 1-D wealth distribution over par.m_grid,
    and writes the resulting time series back onto the path object so that
    plot_transition_paths() can display them.

    Must be called after simulate_hh_path() has populated path.D.

    Writes to path:
        top1_share  – top-1%  share of cash-on-hand wealth
        top10_share – top-10% share of cash-on-hand wealth
        bot50_share – bottom-50% share of cash-on-hand wealth
        gini        – Gini coefficient of cash-on-hand wealth
        mean_wealth – mean cash-on-hand wealth

    Args:
        model:  HANCHousingModelClass instance with path.D populated.
        T_plot: number of periods to compute (default: par.T).
    """
    par  = model.par
    path = model.path

    if not hasattr(path, 'D'):
        raise RuntimeError("path.D not found — run simulate_hh_path() first.")

    T = int(par.T) if T_plot is None else int(min(T_plot, par.T))
    m_grid = np.asarray(par.m_grid, dtype=float)
    Na = len(m_grid)

    top1  = np.full(T, np.nan)
    top10 = np.full(T, np.nan)
    bot50 = np.full(T, np.nan)
    gini  = np.full(T, np.nan)
    meanw = np.full(T, np.nan)

    D = path.D  # (T, Nz, Nh, Na)

    for t in range(T):
        # marginalise over Nz and Nh axes → wealth PMF (Na,)
        D_t = np.asarray(D[t])
        if D_t.ndim == 4:          # (Nfix, Nz, Nh, Na)
            D_t = D_t.sum(axis=(0, 1, 2))
        elif D_t.ndim == 3:        # (Nz, Nh, Na)
            D_t = D_t.sum(axis=(0, 1))
        else:
            continue

        if D_t.size != Na:
            continue
        total = D_t.sum()
        if total <= 0 or not np.isfinite(total):
            continue
        p = D_t / total

        # sort by wealth level
        idx   = np.argsort(m_grid)
        g, pw = m_grid[idx], p[idx]
        mu    = float(np.dot(pw, g))
        if mu <= 0 or not np.isfinite(mu):
            continue

        cum_p = np.cumsum(pw)                 # cumulative population share
        cum_w = np.cumsum(pw * g) / mu        # cumulative wealth share

        top1[t]  = float(np.dot(pw[cum_p >= 0.99], g[cum_p >= 0.99]) / mu)
        top10[t] = float(np.dot(pw[cum_p >= 0.90], g[cum_p >= 0.90]) / mu)
        bot50[t] = float(np.dot(pw[cum_p <= 0.50], g[cum_p <= 0.50]) / mu)
        meanw[t] = mu

        # Gini via trapezoidal rule on Lorenz curve
        cum_p_ext = np.concatenate([[0.0], cum_p])
        cum_w_ext = np.concatenate([[0.0], cum_w])
        gini[t]   = float(1.0 - 2.0 * np.trapz(cum_w_ext, cum_p_ext))

    path.top1_share  = top1
    path.top10_share = top10
    path.bot50_share = bot50
    path.gini        = gini
    path.mean_wealth = meanw


# ==========================================
# 2. Brute force system Jacobian and linearized IRF machinery
# ==========================================

def compute_H_U_full(model, dx=1e-4, save_path='H_U_full.npy', do_print=True):
    """Exact SS-linearised H_U by full column FD, around the steady state.
    Anchored unknown-periods get an identity row/column (their FD column is
    identically zero because _set_unknowns clobbers them). Checkpointed/resumable."""
    import numpy as np, time, os
    from transition import _evaluate_H_nonlinear
    par, ss, path = model.par, model.ss, model.path
    T = par.T
    nu, nt   = len(model.unknowns), len(model.targets)
    n_anchor = int(getattr(par, 'n_terminal_anchor', 0))

    for sh in model.shocks:                       # flat baseline -> dedup -> fast
        path.__dict__[sh][:, 0] = ss.__dict__[sh]
    model._ini_frozen = False
    model.ini.D = ss.D.copy()
    if hasattr(ss, 'D_cohort'):
        model.ini.D_cohort = ss.D_cohort.copy()
    if hasattr(path, 'cohort_weights'):
        path.cohort_weights = np.asarray(ss.cohort_weights, float).ravel().copy()


    x0   = np.concatenate([np.full(T, float(ss.__dict__[u])) for u in model.unknowns])
    dxv  = [max(dx, 1e-3*max(abs(float(ss.__dict__[u])), 1.0)) for u in model.unknowns]
    ncol = nu*T
    anchored = lambda s: n_anchor > 0 and s >= T - n_anchor

    H_U = None
    if os.path.exists(save_path):
        H_U = np.load(save_path)
        if H_U.shape != (nt*T, nu*T):
            print(f'  stale save {H_U.shape} != {(nt*T, nu*T)} — starting fresh')
            H_U = None
    if H_U is None:
        H_U = np.zeros((nt*T, nu*T))

    y0 = _evaluate_H_nonlinear(model, x0)
    n_rem = sum(1 for k in range(ncol)
                if not anchored(k % T) and not np.any(H_U[:, k]))
    t0, did = time.time(), 0
    for k in range(ncol):
        j, s = k // T, k % T
        if anchored(s):                                   # pin equation -> identity
            H_U[k, :] = 0.0; H_U[:, k] = 0.0; H_U[k, k] = 1.0
            continue
        if np.any(H_U[:, k]):                             # already done (resume)
            continue
        xp = x0.copy(); xp[k] += dxv[j]
        H_U[:, k] = (_evaluate_H_nonlinear(model, xp) - y0) / dxv[j]
        did += 1
        if did % 5 == 0:
            np.save(save_path, H_U)
            el = time.time() - t0
            print(f'  {did}/{n_rem} this run  {el/60:.0f}min  '
                  f'ETA {el/did*(n_rem-did)/60:.0f}min', flush=True)
    np.save(save_path, H_U)
    _evaluate_H_nonlinear(model, x0)
    model.H_U = H_U
    print(f'DONE. cond(H_U) = {np.linalg.cond(H_U):.2e}')
    return H_U




def make_demographic_shock(m, eps=0.02, rho=0.95, young_age_max=36):
    """Shift `eps` of mass from older cohorts to ages 16..young_age_max,
    AR(1)-decaying with persistence `rho`. Row-sums of cohort_weights stay 1."""
    omega_ss = np.asarray(m.ss.cohort_weights, float).ravel()
    J, T, age_min = omega_ss.size, m.par.T, 16
    young = np.arange(J) < (young_age_max - age_min + 1)        # ages 16..36
    old   = ~young
    delta = np.zeros(J)
    delta[young] = +eps * omega_ss[young] / omega_ss[young].sum()
    delta[old]   = -eps * omega_ss[old]   / omega_ss[old].sum()
    return (rho ** np.arange(T))[:, None] * delta[None, :]      # (T, J)


def linear_irf(model, shock_name, shock_dev, do_print=True, tikhonov=0.0):
    """Linearised GE IRF + non-linear PE IRF for the HH demand panels.

    GE: dU = −H_U⁻¹ · dH where dH ≈ H_Z · dZ is found by re-evaluating the DAG
    at (U_ss, Z_ss + dZ).  HH block is linearised via `use_hh_jac=True` for the
    propagation step.  The IRF for every (T, 1) path variable is read off
    `path` after linear propagation.

    PE: non-linear OLG household simulation via `simulation.simulate_olg_pe`.

    Shock → simulate_olg_pe scenario mapping:
        r_world         → scenario['r']         = ss.r + shock_dev
        kappa           → scenario['kappa']     = ss.kappa + shock_dev
        hh_scale        → scenario['hh_scale']  = ss.hh_scale + shock_dev
        cohort_weights  → scenario['cohort_weights'] = ω_ss + shock_dev (T, J)
    Other shocks are leftovers from earlier developmet (e.g. Gamma, L_supply) have no direct HH channel and the
    PE line is left out.

    Returns a dict with one (T,) array per `(T, 1)` path variable, plus
    `irf['_base']`

    tikhonov : if > 0, regularise the H_U inverse by replacing each singular
               value sigma with sigma / (sigma**2 + tikhonov**2). Note used.
    """
    import numpy as np
    par, ss, path = model.par, model.ss, model.path
    T = par.T
    H_U = np.asarray(model.H_U)

    # Tikhonov-regularised solve: du = -(V * sigma/(sigma^2+lam^2)) U^T dH.
    # Reduces to np.linalg.solve(H_U, ...) when tikhonov == 0.
    if tikhonov > 0:
        _U_svd, _sv, _Vt = np.linalg.svd(H_U, full_matrices=False)
        _sv_inv = _sv / (_sv**2 + tikhonov**2)
        def _solve(rhs):
            return -(_Vt.T * _sv_inv) @ (_U_svd.T @ rhs)
    else:
        def _solve(rhs):
            return -np.linalg.solve(H_U, rhs)

    def _reset_to_ss():
        for s in model.shocks:
            path.__dict__[s][:, 0] = ss.__dict__[s]
        if hasattr(path, 'cohort_weights'):
            path.cohort_weights = np.asarray(ss.cohort_weights, float).ravel().copy()

    def _apply_shock():
        # cohort_weights is special: (T, J) shock added to the SS cross-section
        # tiled across T.  Everything else is a (T, 1) additive shock.
        if shock_name == 'cohort_weights':
            omega_ss = np.asarray(ss.cohort_weights, float).ravel()
            path.cohort_weights = omega_ss[None, :] + np.asarray(shock_dev, float)
        else:
            path.__dict__[shock_name][:, 0] += shock_dev

    def _eval():
        # Construct x from whatever is currently on path for each unknown,
        # then re-evaluate the DAG.  Uses _evaluate_H_nonlinear (full HH solve);
        # see the module-level definition for the linear-HH alternative.
        x = np.concatenate([np.asarray(path.__dict__[u])[:, 0]
                            for u in model.unknowns])
        return _evaluate_H_nonlinear(model, x)

    # ---- 1. baseline (no shock, U at SS) -----------------------------------
    _reset_to_ss()
    for u in model.unknowns:
        path.__dict__[u][:, 0] = ss.__dict__[u]
    H_base = _eval()
    base = {nm: np.asarray(path.__dict__[nm])[:, 0].copy()
            for nm in path.__dict__
            if isinstance(path.__dict__[nm], np.ndarray)
            and path.__dict__[nm].ndim == 2
            and path.__dict__[nm].shape == (T, 1)}

    # ---- 2. PE via simulate_olg_pe (full non-linear HH, no GE feedback) ----
    irf_pe_hh = None
    try:
        from simulation import simulate_olg_pe
    except Exception:
        simulate_olg_pe = None

    _pe_direct_map = {
        'r_world':  'r',       # interest_rate block: r = r_world
        'kappa':    'kappa',
        'hh_scale': 'hh_scale',
    }
    pe_scenario = None
    if simulate_olg_pe is not None:
        if shock_name == 'cohort_weights':
            _omega_ss = np.asarray(ss.cohort_weights, float).ravel()
            _shock2d  = np.asarray(shock_dev, float)
            # Accept (T, J) directly, or broadcast a length-J vector across T.
            if _shock2d.ndim == 1:
                _shock2d = np.tile(_shock2d[None, :], (T, 1))
            pe_scenario = {'cohort_weights': _omega_ss[None, :] + _shock2d}
        elif shock_name in _pe_direct_map:
            _hh_input = _pe_direct_map[shock_name]
            _ss_level = float(getattr(ss, _hh_input))
            _dvec     = np.asarray(shock_dev, float).ravel()
            pe_scenario = {_hh_input: _ss_level + _dvec}

    if pe_scenario is not None:
        # Snapshot the path arrays simulate_olg_pe might mutate, so the GE
        # block below sees the same baseline path as before this call.
        _snap_names = ('r', 'w', 'q_u', 'q_r', 'f_u', 'f_r',
                       'kappa', 'hh_scale', 'cohort_weights')
        _snap = {}
        for _nm in _snap_names:
            _arr = path.__dict__.get(_nm)
            if isinstance(_arr, np.ndarray):
                _snap[_nm] = _arr.copy()
        try:
            _pe = simulate_olg_pe(model, scenario=pe_scenario)
            irf_pe_hh = {
                'H_u_hh': np.asarray(_pe['H_u_hh'], float) - float(ss.H_u_hh),
                'H_r_hh': np.asarray(_pe['H_r_hh'], float) - float(ss.H_r_hh),
            }
        finally:
            for _nm, _arr in _snap.items():
                _cur = path.__dict__.get(_nm)
                if isinstance(_cur, np.ndarray) and _cur.shape == _arr.shape:
                    _cur[:] = _arr
                else:
                    path.__dict__[_nm] = _arr.copy()

    # ---- 3. GE shock: solve linear system ---------------------------------
    _reset_to_ss()
    _apply_shock()
    H_shock = _eval()
    dH = H_shock - H_base                       # shock-induced residual
    du = _solve(dH)                              # linear unknowns update (Tikhonov-regularised if tikhonov>0)

    for i, u in enumerate(model.unknowns):
        path.__dict__[u][:, 0] = ss.__dict__[u] + du[i*T:(i+1)*T]
    _ = _eval()                                  # linear propagation through DAG

    # ---- 4. assemble IRF dict (GE deviations + simulate_olg_pe PE) ---------
    irf = {nm: np.asarray(path.__dict__[nm])[:, 0] - base[nm] for nm in base}

    if irf_pe_hh is not None:
        irf['H_u_hh_pe'] = irf_pe_hh['H_u_hh']
        irf['H_r_hh_pe'] = irf_pe_hh['H_r_hh']

    # Populate _base so plot_irf's pct=True actually scales to % of SS instead
    # of silently falling back to raw levels. Keys are series names; values
    # are the SS scalar baselines (read off model.ss).
    irf['_base'] = {}
    for nm in base:
        if not hasattr(ss, nm):
            continue
        try:
            _v = float(getattr(ss, nm))
        except (TypeError, ValueError):
            continue
        if np.isfinite(_v):
            irf['_base'][nm] = _v

    if do_print:
        print(f"||baseline H|| = {np.linalg.norm(H_base):.3e}")
        print(f"||shock H||    = {np.linalg.norm(H_shock):.3e}")
        print(f"||dH||         = {np.linalg.norm(dH):.3e}")
        if irf_pe_hh is None:
            print(f"(no PE line: shock '{shock_name}' has no direct HH-input "
                  f"mapping or simulate_olg_pe is unavailable)")
    return irf


def cumulative_multipliers(irfs, rho=0.95, eps=0.01, T_sum=None, t_start=0,
                            label='Shock', do_print=True):
    """Cumulative IRF multipliers for AR(1) shocks (Table 5.4 of the thesis).

    Operates on IRFs you've already computed via `linear_irf`, so nothing is
    re-solved here.

    Args:
        irfs   : either
            (a) a single IRF dict returned by `linear_irf`, or
            (b) a dict  {row_label: irf_dict}  for tabling many shocks at once.
        rho    : AR(1) persistence. Scalar (used for every shock) or a dict
                 {row_label: rho} for per-shock overrides.
        eps    : impact magnitude. Scalar or dict, same convention as `rho`.
        T_sum  : exclusive upper bound of the cumulative window. Default =
                 length of the first available series in the IRF.
        t_start: inclusive lower bound of the cumulative window (default 0).
                 Skips the first `t_start` periods of BOTH the numerator (IRF
                 response) and the denominator (AR(1) shock). Same convention
                 as `rho`/`eps` — scalar or per-row dict.
        label  : only used when `irfs` is a single dict — the row label.
        do_print: print the PE / GE / Wedge table.

    Returns:
        dict {row_label: {column_key: multiplier, ...}}. Columns:
            'PE_H_u_hh', 'PE_H_r_hh',
            'GE_q_u',   'GE_q_r',   'GE_H_u_hh',  'GE_H_r_hh',
            'Wedge_W_u', 'Wedge_W_r'   ← wealth GE wedge (Definition 5.1)
        Missing entries are np.nan (e.g. PE for q_u/q_r is undefined).

    The wealth wedge per region j is:
        M_W^j = q^j_ss · (M_GE^{H_j} − M_PE^{H_j})
              + H^j_ss · (M_GE^{q_j} − M_PE^{q_j})
    with M_PE^{q_j} = 0 by default (SOE PE pins equilibrium prices at SS).
    Reads q^j_ss and H^j_ss from irf['_base'] populated by linear_irf.
    """
    # Detect single IRF vs. dict-of-IRFs.
    if not isinstance(irfs, dict) or not irfs:
        raise TypeError('irfs must be a non-empty dict.')
    _sample = next(iter(irfs.values()))
    is_dict_of_irfs = isinstance(_sample, dict) and any(
        k in _sample for k in ('q_u', 'q_r', 'H_u_hh', 'H_r_hh', '_base')
    )
    all_irfs = irfs if is_dict_of_irfs else {label: irfs}

    def _resolve(arg, key, default):
        if isinstance(arg, dict):
            return float(arg.get(key, default))
        return float(arg)

    pe_responses = [('H_u_hh', 'H^u'), ('H_r_hh', 'H^r')]
    ge_responses = [('q_u',    'q^u'), ('q_r',    'q^r'),
                    ('H_u_hh', 'H^u'), ('H_r_hh', 'H^r')]

    results = {}
    for lbl, irf in all_irfs.items():
        rho_i     = _resolve(rho,     lbl, 0.95)
        eps_i     = _resolve(eps,     lbl, 0.01)
        t_start_i = int(_resolve(t_start, lbl, 0))

        # Resolve horizon for the cumulative sum.
        if T_sum is None:
            T_eff = None
            for key, _ in ge_responses:
                if key in irf:
                    T_eff = int(np.asarray(irf[key]).size)
                    break
            if T_eff is None:
                T_eff = 0
        else:
            T_eff = int(T_sum)

        # Clamp t_start to a sensible range.
        if t_start_i < 0:
            t_start_i = 0
        if t_start_i >= T_eff:
            t_start_i = max(0, T_eff - 1)

        # Cumulative shock denominator over [t_start, T_eff).
        s_arr = np.arange(t_start_i, T_eff, dtype=float)
        cum_shock = float(np.sum(eps_i * (rho_i ** s_arr)))
        if not np.isfinite(cum_shock) or cum_shock == 0.0:
            # Fallback: closed form ε·ρ^{t_start} / (1−ρ).
            cum_shock = eps_i * (rho_i ** t_start_i) / max(1.0 - rho_i, 1e-12)

        row = {}
        # PE columns
        for key, _ in pe_responses:
            pe_key = key + '_pe'
            if pe_key in irf:
                arr = np.asarray(irf[pe_key])[t_start_i:T_eff]
                row['PE_' + key] = float(np.sum(arr)) / cum_shock
            else:
                row['PE_' + key] = np.nan
        # GE columns
        for key, _ in ge_responses:
            if key in irf:
                arr = np.asarray(irf[key])[t_start_i:T_eff]
                row['GE_' + key] = float(np.sum(arr)) / cum_shock
            else:
                row['GE_' + key] = np.nan

        ss_base = irf.get('_base', {})

        def _arr(key):
            """Return the IRF deviation series for `key`, sliced to the
            cumulative window.  None if the IRF dict doesn't carry it."""
            v = irf.get(key)
            return None if v is None else np.asarray(v, dtype=float)[t_start_i:T_eff]

        # Length of the cumulative window (consistent across all panels).
        N_window = max(0, T_eff - t_start_i)

        for region_suffix, q_key, H_key in (('u', 'q_u', 'H_u_hh'),
                                              ('r', 'q_r', 'H_r_hh')):
            q_ss = ss_base.get(q_key, np.nan)
            H_ss = ss_base.get(H_key, np.nan)

            dH_GE = _arr(H_key)
            dq_GE = _arr(q_key)

            # PE deviation series.  Missing PE for prices is treated as the
            # zero series — in the SOE PE definition equilibrium prices are
            # pinned at SS.  Missing PE for housing demand falls back to the
            # zero series too (i.e. no PE channel for this shock).
            dH_PE = _arr(H_key + '_pe')
            dq_PE = _arr(q_key + '_pe')
            if dH_PE is None:
                dH_PE = np.zeros(N_window)
            if dq_PE is None:
                dq_PE = np.zeros(N_window)

            if (dH_GE is None or dq_GE is None
                    or not np.isfinite(q_ss) or not np.isfinite(H_ss)):
                row['Wedge_W_' + region_suffix] = np.nan
                continue

            # Per-period wedge in levels.
            wedge_t = (q_ss * (dH_GE - dH_PE)
                       + H_ss * (dq_GE - dq_PE))
            # Cumulate, then normalise by the cumulative shock.
            row['Wedge_W_' + region_suffix] = (
                float(np.sum(wedge_t)) / cum_shock
            )

        row['_label']   = lbl
        row['_eps']     = eps_i
        row['_rho']     = rho_i
        row['_T_sum']   = T_eff
        row['_t_start'] = t_start_i
        results[lbl] = row

    if do_print:
        col_pe = [('PE_H_u_hh', 'H^u'), ('PE_H_r_hh', 'H^r')]
        col_ge = [('GE_q_u',   'q^u'), ('GE_q_r',   'q^r'),
                  ('GE_H_u_hh','H^u'), ('GE_H_r_hh','H^r')]
        col_w  = [('Wedge_W_u', 'W^u'), ('Wedge_W_r', 'W^r')]   # wealth wedge
        all_cols = col_pe + col_ge + col_w

        w_label = max(len(r['_label']) for r in results.values()) + 2
        w_cell  = 9

        pe_w = w_cell * len(col_pe)
        ge_w = w_cell * len(col_ge)
        wd_w = w_cell * len(col_w)
        header_row1 = (' ' * w_label
                       + f'{"PE":^{pe_w}}'
                       + f'{"GE":^{ge_w}}'
                       + f'{"Wedge":^{wd_w}}')
        header_row2 = (' ' * w_label
                       + ''.join(f'{lbl:>{w_cell}}' for _, lbl in col_pe)
                       + ''.join(f'{lbl:>{w_cell}}' for _, lbl in col_ge)
                       + ''.join(f'{lbl:>{w_cell}}' for _, lbl in col_w))
        rule = '-' * len(header_row2)

        print('\nCumulative multipliers   m_U = (1−ρ)·Σ_t dU_t / ε  =  Σ_t dU_t / Σ_t dZ_t')
        print(rule)
        print(header_row1)
        print(header_row2)
        print(rule)
        for _lbl, row in results.items():
            line = f'{row["_label"]:<{w_label}}'
            for key, _ in all_cols:
                v = row.get(key, np.nan)
                if not np.isfinite(v):
                    line += f'{"—":>{w_cell}}'
                else:
                    line += f'{v:>{w_cell}.3f}'
            print(line)
        print(rule)
        # Per-row parameters footer
        param_lines = ['(ε, ρ, window) per row:']
        for _lbl, row in results.items():
            param_lines.append(
                f'  {row["_label"]:<{w_label}}'
                f'ε={row["_eps"]:.4g}  ρ={row["_rho"]:.4g}  '
                f'window=[{row["_t_start"]}, {row["_T_sum"]})'
            )
        print('\n'.join(param_lines))

    return results







# ==========================================
# 3. Fake News machinery for computing household Jacobian
# ==========================================

def compute_age_specific_fake_news_matrices(
    model,
    inputname,
    outputname,
    dx=1e-4,
    age_horizon=None,
    selected_shock_ages=None,
    backward_window=None,
    do_print=False,
    prices_baseline=None,
):
    """Compute age-specific fake-news matrices for one HH input-output pair.

    This implements the core structure of Algorithm 1 in Bardoczy and
    Velasquez-Giraldo (2025) for a single input and output:
    1) Build baseline age profiles and expectation vectors once.
    2) For each selected shock age k, perturb input[k] by dx.
    3) Fill the age-specific fake-news objects using direct policy effects and
       one-step distribution effects propagated by expectation vectors.

    Important:
    - This function returns AGE-SPECIFIC fake-news matrices only.
    - It does not perform any age-weighted aggregation.

    Args:
        model: HANCHousingModel instance.
        inputname: One HH input in model.inputs_hh, e.g. 'r', 'w', 'q_u', 'q_r'.
        outputname: One HH output in {'A_hh','C_hh','H_u_hh','H_r_hh'}.
        dx: Finite-difference shock size.
        age_horizon: Number of ages A to include. Defaults to min(par.J, par.T).
        selected_shock_ages: Optional iterable of k ages to compute.
        backward_window: Number of backward one-step updates around each shock.
            If None or <=0, uses full backward solve for each k.
        do_print: Print progress.

    Returns:
        dict with keys:
            'F_by_age': list of length A; F_by_age[a] is a (T,T) matrix.
                The nonzero support follows Theorem 1 and is confined to the
                finite-lifetime block implied by age_horizon=A. Entries
                outside that block are structurally zero.
            'expectation_vectors': nested list E[a][h] (or None where invalid).
            'baseline_output_by_age': length-A baseline output profile.
            'baseline_distribution_by_age': list of length A with (Nz,Nh,Na) arrays.
            'shock_ages': sorted np.ndarray of computed k ages.
            'age_horizon': int A.
            'inputname': inputname.
            'outputname': outputname.
            'dx': float dx.
            'backward_window': backward_window.
    """
    par = model.par
    ss = model.ss
    T = int(par.T)

    if inputname not in model.inputs_hh:
        raise ValueError(f"inputname '{inputname}' not in model.inputs_hh={model.inputs_hh}")

    valid_outputs = {'A_hh', 'C_hh', 'H_u_hh', 'H_r_hh'}
    if outputname not in valid_outputs:
        raise ValueError(f"outputname must be one of {sorted(valid_outputs)}")

    if not hasattr(ss, 'D'):
        raise RuntimeError('Steady-state distribution ss.D is missing.')

    A = int(age_horizon) if age_horizon is not None else int(getattr(par, 'J', par.T))
    A = max(2, min(A, T))

    if selected_shock_ages is None:
        shock_ages = np.arange(A, dtype=int)
    else:
        shock_ages = np.asarray(sorted(set(int(k) for k in selected_shock_ages)), dtype=int)
        shock_ages = shock_ages[(shock_ages >= 0) & (shock_ages < A)]
        if shock_ages.size == 0:
            raise ValueError('selected_shock_ages contains no ages in [0, A-1].')

    Nh = par.Nh if hasattr(par, 'Nh') else 2
    Nz = int(par.Nz)
    Na = int(par.Na)
    Nstate = Nz * Nh * Na

    def _normalize_dist(D_in):
        D = np.asarray(D_in, dtype=float).copy()
        s = np.sum(D)
        if s > 0.0 and np.isfinite(s):
            D /= s
        return D

    def _output_loading_vector(output, a_pol, c_pol):
        if output == 'A_hh':
            return np.asarray(a_pol, dtype=float).reshape(-1)
        if output == 'C_hh':
            return np.asarray(c_pol, dtype=float).reshape(-1)

        g = np.zeros((Nz, Nh, Na))
        if output == 'H_u_hh':
            # Urban renters (state 1) get h_l units; urban owners (state 3) get h_u units
            g[:, 1, :] = float(getattr(par, 'h_l', par.h_r))
            g[:, 3, :] = par.h_u
        elif output == 'H_r_hh':
            # Rural renters (state 0) get h_l units; rural owners (state 2) get h_r units
            g[:, 0, :] = float(getattr(par, 'h_l', par.h_r))
            g[:, 2, :] = par.h_r
        return g.reshape(-1)

    def _build_transition_matrix(a_pol, pr_u_pol, age_idx, prices_dict):
        rows, cols, data, _ = steady_state.build_transition_coo(
            a_pol,
            pr_u_pol,
            par.z_trans[0] if par.z_trans.ndim == 3 else par.z_trans,
            np.asarray(par.z_grid),
            par.m_grid,
            float(prices_dict['r'][age_idx]),
            float(prices_dict['w'][age_idx]) * (np.exp(float(par.chi[min(age_idx + 1, par.T - 1)])) if hasattr(par, 'chi') else 1.0),
            float(prices_dict['q_u'][age_idx]),
            float(prices_dict['q_r'][age_idx]),
            par.cfloor,
            par.zeta,
            float(getattr(par, 'zeta_renter', 0.0)),
            par.h_u,
            par.h_r,
            float(getattr(par, 'h_l', par.h_r)),
            float(prices_dict['f_u'][age_idx]) if 'f_u' in prices_dict else float(prices_dict['q_u'][age_idx]) * ((1.0 + float(prices_dict['r'][age_idx])) - (1- par.theta) * (1.0 - par.delta_H)),
            float(prices_dict['f_r'][age_idx]) if 'f_r' in prices_dict else float(prices_dict['q_r'][age_idx]) * ((1.0 + float(prices_dict['r'][age_idx])) - (1- par.theta) * (1.0 - par.delta_H)),
            par.lambda_ltv, float(par.T_mort), float(par.delta_H),
            float(getattr(par, 'tau_wealth', 0.0)), float(getattr(par, 'tau_profits', 0.0)),
            Nstate * Nz * 4 * 2 + 1000,
        )
        return coo_matrix((data, (rows, cols)), shape=(Nstate, Nstate)).tocsr()

    if prices_baseline is not None:
        prices_ss = {k: np.asarray(v, dtype=float).ravel()[:par.T].copy()
                     for k, v in prices_baseline.items()}
        if 'kappa' not in prices_ss:
            prices_ss['kappa'] = np.full(par.T, float(getattr(ss, 'kappa', par.kappa)))
    else:
        prices_ss = {
            'r': np.full(par.T, float(ss.r)),
            'w': np.full(par.T, float(ss.w)),
            'q_u': np.full(par.T, float(ss.q_u)),
            'q_r': np.full(par.T, float(ss.q_r)),
            'f_u': np.full(par.T, float(ss.f_u)),
            'f_r': np.full(par.T, float(ss.f_r)),
            'kappa': np.full(par.T, float(getattr(ss, 'kappa', par.kappa))),
        }

    # Save model.sol arrays since one-step updates write to them as side effects.
    sol_backup = {}
    for nm in ('a', 'c', 'pr_urban', 'pr_rural'):
        if hasattr(model.sol, nm):
            sol_backup[nm] = np.asarray(getattr(model.sol, nm)).copy()

    hh = household_problem.HousingModel(model)

    try:
        # Baseline finite-horizon cohort solve.
        baseline_policy_endog, baseline_value_endog = hh.solve_dcegm(prices_path=prices_ss)

        baseline_a = [None] * A
        baseline_c = [None] * A
        baseline_pr_u = [None] * A
        baseline_g = [None] * A
        baseline_P = [None] * (A - 1)
        baseline_D = [None] * A
        baseline_y = np.zeros(A)

        if hasattr(ss, 'cohort_weights'):
            omega_ss = _normalize_cohort_weights(ss.cohort_weights, A)
        elif hasattr(par, 'cohort_weights_ss'):
            omega_ss = _normalize_cohort_weights(par.cohort_weights_ss, A)
        else:
            omega_ss = np.ones(A) / A  # uniform cohort sizes

        D_ss_cond = np.asarray(ss.D, dtype=float).copy()
        _s = np.sum(D_ss_cond)
        if _s > 0.0 and np.isfinite(_s):
            D_ss_cond /= _s

        baseline_D[0] = float(omega_ss[0]) * D_ss_cond
        # -----------------------------------------------------------------------

        for age in range(A):
            a_age, c_age, pr_u_age = _policy_arrays_from_endog(baseline_policy_endog, age, par, Nh)
            baseline_a[age] = a_age
            baseline_c[age] = c_age
            baseline_pr_u[age] = pr_u_age

            g_age = _output_loading_vector(outputname, a_age, c_age)
            baseline_g[age] = g_age
            baseline_y[age] = float(g_age @ baseline_D[age].reshape(-1))

            if age < A - 1:
                P_age = _build_transition_matrix(a_age, pr_u_age, age, prices_ss)
                baseline_P[age] = P_age
                # Propagate WITHOUT renormalizing so absolute mass is preserved.
                D_next = (P_age @ baseline_D[age].reshape(-1)).reshape((Nz, Nh, Na))
                baseline_D[age + 1] = D_next

        # Precompute expectation vectors E[a][h] mapping dD(a) -> dy(a+h).
        expectation_vectors = [[None for _ in range(A)] for _ in range(A)]
        for a in range(1, A):
            for h in range(A - a):
                out_age = a + h
                row = baseline_g[out_age].copy()
                for idx in range(out_age - 1, a - 1, -1):
                    row = baseline_P[idx].transpose() @ row
                expectation_vectors[a][h] = np.asarray(row, dtype=float).ravel()

    
        F_by_age = [np.zeros((T, T)) for _ in range(A)]

        for ik, k in enumerate(shock_ages):
            if do_print:
                print(f'  shock age k={k} ({ik + 1}/{shock_ages.size})')

            prices_pert = {nm: arr.copy() for nm, arr in prices_ss.items()}
            prices_pert[inputname][k] += dx

            # Start from baseline and only update a local backward window.
            pol_pert = {ih: dict(baseline_policy_endog[ih]) for ih in range(Nh)}
            val_pert = {ih: dict(baseline_value_endog[ih]) for ih in range(Nh)}

            if backward_window is None or int(backward_window) <= 0:
                pol_pert, val_pert = hh.solve_dcegm(prices_path=prices_pert)
            else:
                first_it = max(0, k - int(backward_window) + 1)
                for it in range(k, first_it - 1, -1):
                    hh.solve_dcegm_one_step(
                        it,
                        pol_pert,
                        val_pert,
                        prices_path=prices_pert,
                        force_non_terminal=False,
                    )

            for l in range(k + 1):
                j = k - l

                a_l, c_l, pr_u_l = _policy_arrays_from_endog(pol_pert, l, par, Nh)
                g_l = _output_loading_vector(outputname, a_l, c_l)

                # Direct output effect at age l with baseline D_ss(l).
                dy0 = float((g_l - baseline_g[l]) @ baseline_D[l].reshape(-1)) / dx
                F_by_age[l][0, j] = dy0

                if l >= A - 1:
                    continue

                # One-step distribution effect into age l+1.
                P_l = _build_transition_matrix(a_l, pr_u_l, l, prices_pert)
                dD1 = ((P_l @ baseline_D[l].reshape(-1)) - baseline_D[l + 1].reshape(-1)) / dx

                max_m = A - 1 - l
                for m in range(1, max_m + 1):
                    age_target = l + m
                    e_vec = expectation_vectors[l + 1][m - 1]
                    if e_vec is None:
                        continue
                    F_by_age[age_target][m, j] = float(e_vec @ dD1)

    finally:
        for nm, arr in sol_backup.items():
            getattr(model.sol, nm)[:] = arr

    return {
        'F_by_age': F_by_age,
        'expectation_vectors': expectation_vectors,
        'baseline_output_by_age': baseline_y,
        'baseline_distribution_by_age': baseline_D,
        'shock_ages': shock_ages,
        'age_horizon': A,
        'model_horizon': T,
        'inputname': inputname,
        'outputname': outputname,
        'dx': float(dx),
        'backward_window': backward_window,
    }


def jacobian_from_fake_news_matrix(F):
    """Convert one fake-news matrix to one Jacobian via diagonal sums.

    Implements equation:
        J[t,s] = sum_{k=0}^{min(t,s)} F[t-k, s-k]
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError('F must be a square 2D matrix')

    A = F.shape[0]
    J = np.zeros_like(F)

    for t in range(A):
        for s in range(A):
            kmax = min(t, s)
            acc = 0.0
            for k in range(kmax + 1):
                acc += F[t - k, s - k]
            J[t, s] = acc

    return J


def compute_age_specific_hh_jacobians(
    model,
    inputname,
    outputname,
    dx=1e-4,
    age_horizon=None,
    selected_shock_ages=None,
    backward_window=6,
    do_print=False,
    prices_baseline=None,
):
    """Compute age-specific fake-news matrices and corresponding Jacobians.

    This function first calls compute_age_specific_fake_news_matrices, then
    converts each age-specific fake-news matrix F(a) into J(a) using
    jacobian_from_fake_news_matrix.

    Returns:
        result dict from compute_age_specific_fake_news_matrices plus:
            'J_by_age': list of age-specific Jacobian matrices.
            'J_sum_unweighted': unweighted sum_a J(a).

    Notes:
        - Each F(a) and J(a) is represented as a full (T,T) matrix.
        - Lifetime restrictions enter through structural zeros in F(a).
    """
    res = compute_age_specific_fake_news_matrices(
        model=model,
        inputname=inputname,
        outputname=outputname,
        dx=dx,
        age_horizon=age_horizon,
        selected_shock_ages=selected_shock_ages,
        backward_window=backward_window,
        do_print=do_print,
        prices_baseline=prices_baseline,
    )

    F_by_age = res['F_by_age']
    J_by_age = [jacobian_from_fake_news_matrix(Fa) for Fa in F_by_age]

    # Build the aggregate calendar-time Jacobian from the age-specific matrices.

    A = res['age_horizon']
    T = res['model_horizon']

    omega_path = _get_path_cohort_weights(model, A)  # shape (T, A)

    J_agg = np.zeros((T, T))
    for t in range(T):
        for s in range(T):
            delta = t - s  # calendar lag (can be negative: anticipation)
            a_min = max(0, delta)
            for a in range(a_min, A):
                a_shock = a - delta
                if a_shock < 0 or a_shock >= T:
                    continue
                w = omega_path[t, a]
                if w == 0.0:
                    continue
                # J_by_age[a] is (T,T); only element [a, a_shock] is
                # meaningful for the calendar-time aggregate.
                if a < T and a_shock < T:
                    J_agg[t, s] += w * J_by_age[a][a, a_shock]

    out = dict(res)
    out['J_by_age'] = J_by_age
    out['J_sum_unweighted'] = J_agg  # kept for back-compat with downstream code
    return out


def _forward_pass_outputs_exact(model, policy_endog, prices_path_dict):
    """Forward distribution pass from ss.D using pre-solved policies and prices.

    Uses the Numba JIT _scatter_dcegm_forward for speed.  At each calendar time
    t, scatters D[t] -> D[t+1] under policy[t] and prices[t], then aggregates.

    Args
    ----
    model            : HANCHousingModelClass
    policy_endog     : output of hh.solve_dcegm() -- maps housing_state -> {age: pol}
    prices_path_dict : dict of 1-D length-T arrays (r, w, q_u, q_r, f_u, f_r, kappa)

    Returns
    -------
    dict with 'A_hh', 'H_u_hh', 'H_r_hh' as (T,) float64 arrays.
    """
    par = model.par
    ss  = model.ss
    T   = int(par.T)
    Nh  = par.Nh if hasattr(par, 'Nh') else 4
    Nz  = int(par.Nz)
    Na  = int(par.Na)

    z_trans = np.asarray(par.z_trans[0] if par.z_trans.ndim == 3 else par.z_trans,
                         dtype=np.float64)
    z_grid  = np.asarray(par.z_grid, dtype=np.float64)
    m_grid  = np.asarray(par.m_grid, dtype=np.float64)

    A_hh   = np.zeros(T)
    H_u_hh = np.zeros(T)
    H_r_hh = np.zeros(T)

    D     = np.asarray(ss.D, dtype=np.float64).reshape(Nz, Nh, Na).copy()
    D_out = np.zeros_like(D)

    theta      = float(par.theta)
    delta_H    = float(par.delta_H)
    cfloor     = float(par.cfloor)
    zeta       = float(par.zeta)
    zeta_renter = float(getattr(par, 'zeta_renter', 0.0))
    h_u        = float(par.h_u)
    h_r        = float(par.h_r)
    h_l        = float(getattr(par, 'h_l', par.h_r))
    lambda_ltv = float(par.lambda_ltv)
    T_mort     = float(par.T_mort)
    tau_wealth = float(getattr(par, 'tau_wealth', 0.0))
    tau_profits = float(getattr(par, 'tau_profits', 0.0))

    for t in range(T):
        a_t, _, pr_choices_t = _policy_arrays_from_endog(policy_endog, t, par, Nh)

        A_hh[t]   = float(np.dot(D.ravel(), a_t.ravel()))
        H_u_hh[t] = float(np.sum(D[:, 1, :])) * h_l + float(np.sum(D[:, 3, :])) * h_u
        H_r_hh[t] = float(np.sum(D[:, 0, :])) * h_l + float(np.sum(D[:, 2, :])) * h_r

        if t < T - 1:
            r_t   = float(prices_path_dict['r'][t])
            w_t   = float(prices_path_dict['w'][t])
            q_u_t = float(prices_path_dict['q_u'][t])
            q_r_t = float(prices_path_dict['q_r'][t])
            f_u_t = (float(prices_path_dict['f_u'][t]) if 'f_u' in prices_path_dict
                     else q_u_t * ((1.0 + r_t) - (1.0 - theta) * (1.0 - delta_H)))
            f_r_t = (float(prices_path_dict['f_r'][t]) if 'f_r' in prices_path_dict
                     else q_r_t * ((1.0 + r_t) - (1.0 - theta) * (1.0 - delta_H)))
            chi_t = float(par.chi[t]) if hasattr(par, 'chi') else 0.0

            _scatter_dcegm_forward(
                D, D_out,
                np.asarray(a_t,          dtype=np.float64),
                np.asarray(pr_choices_t, dtype=np.float64),
                z_trans, z_grid, m_grid,
                r_t, w_t * np.exp(chi_t), q_u_t, q_r_t,
                cfloor, zeta, zeta_renter, h_u, h_r, h_l,
                f_u_t, f_r_t, lambda_ltv, T_mort, delta_H,
                tau_wealth, tau_profits,
            )
            D[:] = D_out
            D_out[:] = 0.0

    return {'A_hh': A_hh, 'H_u_hh': H_u_hh, 'H_r_hh': H_r_hh}


def compute_jac_hh_exact(model, dx=1e-4, do_print=False):
    """Compute jac_hh by direct finite-difference sequence-space perturbations.

    For each input x_s (price at period s), perturbs prices_path[input][s] by
    dx, runs a full HH backward solve (J EGM steps), then a full forward
    distribution pass from ss.D, and records the response in all HH outputs.

    This is the exact sequence-space Jacobian -- no fake-news time-stationarity
    assumption.  It is valid around ANY steady state and is the correct method
    for large shocks where the SS-T fake-news approximation is inaccurate.

    Cost vs fake_news (T=83, J=49, n_inputs=5):
      - Backward EGM: (1 + 5*83) * 49 = 20,384 steps vs ~4,410 local steps
        for fake-news:  ~4.6x more EGM work.
      - Forward pass: uses Numba JIT scatter_dcegm_forward -- cheap.
      - Expected wall-clock: ~5-10x the fake_news compute time.

    Note: does NOT populate model.age_jac_hh (that requires the fake-news
    age-specific structure).

    Args
    ----
    model    : HANCHousingModelClass with a solved steady state.
    dx       : minimum finite-difference step (auto-scaled per input).
    do_print : print per-input progress with elapsed time and ETA.
    """
    import time as _time

    par = model.par
    ss  = model.ss
    T   = int(par.T)

    hh_outputs = ['H_u_hh', 'H_r_hh']  # A_hh not needed: no asset-market target in SOE
    hh_inputs  = model.inputs_hh

    def _dx_for_input(name):
        ss_val = float(getattr(ss, name, getattr(par, name, 1.0)))
        return max(dx, 2e-3 * abs(ss_val)) if abs(ss_val) >= 1.0 else dx

    f_u_ss = float(getattr(ss, 'f_u',
        ss.q_u * ((1.0 + ss.r) - (1.0 - par.theta) * (1.0 - par.delta_H))))
    f_r_ss = float(getattr(ss, 'f_r',
        ss.q_r * ((1.0 + ss.r) - (1.0 - par.theta) * (1.0 - par.delta_H))))
    prices_base = {
        'r':     np.full(T, float(ss.r)),
        'w':     np.full(T, float(ss.w)),
        'q_u':   np.full(T, float(ss.q_u)),
        'q_r':   np.full(T, float(ss.q_r)),
        'f_u':   np.full(T, f_u_ss),
        'f_r':   np.full(T, f_r_ss),
        'kappa': np.full(T, float(getattr(ss, 'kappa', par.kappa))),
    }

    hh = household_problem.HousingModel(model)

    if do_print:
        print('  [exact FD] Baseline backward solve + forward pass...')
    pol_base, _ = hh.solve_dcegm(prices_path=prices_base)
    Y_base = _forward_pass_outputs_exact(model, pol_base, prices_base)

    model.jac_hh = {}
    for output in hh_outputs:
        for inputname in hh_inputs:
            model.jac_hh[(output, inputname)] = np.zeros((T, T))

    n_total = sum(1 for inp in hh_inputs if inp in prices_base) * T
    n_done  = 0
    t0      = _time.time()

    for inputname in hh_inputs:
        if inputname not in prices_base:
            if do_print:
                print(f'  [exact FD] skipping {inputname} (not in SS prices dict)')
            continue
        dx_inp = _dx_for_input(inputname)
        if do_print:
            print(f'  [exact FD] input={inputname:8s}  dx={dx_inp:.2e}  ({T} solves)')

        for s in range(T):
            prices_pert = {k: v.copy() for k, v in prices_base.items()}
            prices_pert[inputname][s] += dx_inp

            pol_pert, _ = hh.solve_dcegm(prices_path=prices_pert)
            Y_pert = _forward_pass_outputs_exact(model, pol_pert, prices_pert)

            for output in hh_outputs:
                model.jac_hh[(output, inputname)][:, s] = (
                    Y_pert[output] - Y_base[output]
                ) / dx_inp

            n_done += 1
            if do_print and n_done % 10 == 0:
                elapsed = _time.time() - t0
                rate    = n_done / elapsed if elapsed > 0 else 1.0
                eta     = (n_total - n_done) / rate
                print(f'    [{n_done:4d}/{n_total}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s')

    if do_print:
        print(f'  [exact FD] done.  Total: {_time.time()-t0:.1f}s')




# ==========================================
# 3. Handle block structure and compute non-hh Jacobians
# ==========================================

def evaluate_path_blocks(model, use_hh_jac=False):
    """
    Evaluate all blocks sequentially along the path stored in model.path.

    """

    par = model.par
    ss = model.ss
    path = model.path

    # Ensure ini is populated from steady state (needed for lag() calls).
    # Skip if find_transition_path has frozen ini to custom initial-SS values.
    if hasattr(model, 'ini') and not getattr(model, '_ini_frozen', False):
        for varname in model.varlist:
            if hasattr(ss, varname):
                model.ini.__dict__[varname] = ss.__dict__[varname]

    # Detect ncols from any unknown's path array shape.
    ncols = 1
    for varname in model.unknowns:
        arr = path.__dict__.get(varname)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            ncols = arr.shape[1]
            break
    assert use_hh_jac or ncols == 1, \
        'Non-linear HH (use_hh_jac=False) requires ncols=1; widened paths must use the linearized HH branch.'

    # Reset all intermediate/output variables to SS before evaluating the DAG.
    # Unknowns and shocks are left untouched — the caller sets those.
    preserve = set(model.unknowns) | set(model.shocks)
    for varname in model.varlist:
        if varname in preserve:
            continue
        if hasattr(ss, varname) and hasattr(path, varname):
            arr = path.__dict__[varname]
            if arr.ndim == 2:
                arr[:, :] = ss.__dict__[varname]

    # Evaluate each block in DAG order
    for blockstr in model.blocks:

        if blockstr == 'hh':
            # Household block
            if use_hh_jac and hasattr(model, 'jac_hh'):
                # Linearised HH chain rule: intercept at SS level + per-input
                # price-deviation responses from jac_hh.  A_hh is excluded: no
                # asset-market clearing target in the SOE.
                _hh_scale_ss = float(getattr(ss, 'hh_scale',
                                             getattr(par, 'hh_scale', 1.0)))
                for varname in ['H_u_hh', 'H_r_hh']:
                    path.__dict__[varname][:, :] = ss.__dict__[varname]
                    # d_input has shape (T, ncols); jac @ d_input → (T, ncols).
                    for inputname in model.inputs_hh:  # 'r', 'w', 'q_u', 'q_r', ...
                        jac = model.jac_hh[(varname, inputname)]
                        d_input = path.__dict__[inputname] - ss.__dict__[inputname]
                        path.__dict__[varname][:, :] += jac @ d_input
                    # hh_scale contribution: H_i_hh = hh_scale * per_capita_i
    
                    if hasattr(path, 'hh_scale') and _hh_scale_ss > 0:
                        _per_cap = ss.__dict__[varname] / _hh_scale_ss
                        _d_hh_scale = path.hh_scale - ss.hh_scale
                        path.__dict__[varname][:, :] += _per_cap * _d_hh_scale

            else:
                # Non-linear household solution (ncols=1 only — assertion above).
                # OLG-correct: aggregates over cohorts using cohort_weights, with
                # a single calendar-age dcegm sweep replacing the legacy
                # single-cohort simulate_hh_path.
                _simulate_olg_path(model)
        
        else:
            # Production, mutual fund, or market clearing blocks
            module_name, func_name = blockstr.split('.')
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            # Get variable names from function signature
            sig = inspect.signature(func)
            varnames = [p for p in sig.parameters.keys() 
                       if p not in ['par', 'ini', 'ss']]
            
            # Build inputs dictionary
            inputs = {name: path.__dict__[name] for name in varnames}
            
            # Call block function
            if hasattr(func, 'py_func'):
                func.py_func(par, model.ini, ss, **inputs)
            else:
                func(par, model.ini, ss, **inputs)


def get_hh_variable_name(outputname):
    """
    Convert outputs_hh name to actual variable name.
    Framework convention: outputs_hh = ['a', 'c', 'h_u', 'h_r'] (lowercase)
    Maps to actual variables: ['A_hh', 'C_hh', 'H_u_hh', 'H_r_hh'] (capitalized with _hh)
    """
    return f'{outputname.upper()}_hh'








def _widen_path_for_jac(model, ncols):
    """Widen all (T, 1) path arrays to (T, ncols), tiling the existing column 0
    across the new columns.  Returns a dict {varname: original_array} so the
    caller can restore via _restore_path_arrays.

    Only arrays of shape (T, 1) are widened; shape-different arrays
    (e.g. distribution path D) are left untouched.
    """
    par = model.par
    saved = {}
    for varname in list(model.path.__dict__.keys()):
        arr = model.path.__dict__[varname]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 2 and arr.shape[0] == par.T and arr.shape[1] == 1:
            saved[varname] = arr
            new = np.empty((par.T, ncols), dtype=arr.dtype)
            new[:, :] = arr  # broadcasts (T, 1) → (T, ncols)
            model.path.__dict__[varname] = new
    return saved


def _restore_path_arrays(model, saved):
    """Restore path arrays previously widened by _widen_path_for_jac."""
    for varname, arr in saved.items():
        model.path.__dict__[varname] = arr


def _build_H_U_vectorized(model, baseline_targets, x_baseline, dx, do_print=False):
    """Build H_U via a single column-vectorized DAG sweep.

    Mirrors GEModelTools._compute_jac (GEModelClass.py:1269): widens path to
    (T, N_unknowns*T), perturbs each (unknown, period) cell by dx in its own
    column, evaluates evaluate_path_blocks once, and computes finite-
    difference derivatives column-by-column.

    Args
    ----
    model            : HANCHousingModelClass
    baseline_targets : (N_targets, T) baseline residuals (un-perturbed)
    x_baseline       : (N_unknowns, T) unknown values to perturb around
    dx               : finite-difference step (minimum; auto-scaled per unknown)
    do_print         : print per-unknown dx values actually used

    Writes to model.H_U.  Caller is responsible for restoring path state.
    """
    par = model.par
    ncols = len(model.unknowns) * par.T

    # Build per-unknown dx scaled to the magnitude of each unknown's values.
    # dx=1e-4 is safe for O(0.01) variables but gives a relative step of ~2e-5
    # for unknowns like H_u~5.4 or H_r~6.6, which falls below floating-point
    # noise and produces a near-identity H_U.  Scale so the relative step is
    # at least 1e-3 for any unknown larger than 1.
    n_unk = len(model.unknowns)
    dx_vec = np.empty(n_unk)
    for i in range(n_unk):
        scale = max(abs(float(np.mean(x_baseline[i, :]))), 1.0)
        dx_vec[i] = max(dx, 1e-3 * scale)
    # Column j = i_unk*T + s  →  divide by dx_vec[i_unk]
    denom = np.repeat(dx_vec, par.T)  # shape (ncols,)

    if do_print:
        for i, name in enumerate(model.unknowns):
            scale = max(abs(float(np.mean(x_baseline[i, :]))), 1.0)
            print(f'  H_U dx  {name:15s}  ss≈{scale:.4g}  dx_used={dx_vec[i]:.2e}')

    # Widen all (T, 1) path arrays to (T, ncols).
    saved = _widen_path_for_jac(model, ncols)

    try:
        # Set unknowns: each column is x_baseline plus dx_vec[i] in one (i_unk, s) slot.
        for i, unkname in enumerate(model.unknowns):
            arr = model.path.__dict__[unkname]  # (T, ncols)
            arr[:, :] = x_baseline[i, :, None]
            for s in range(par.T):
                arr[s, i * par.T + s] += dx_vec[i]

        # Single widened DAG sweep with linearized HH (use_hh_jac=True).
        evaluate_path_blocks(model, use_hh_jac=True)

        # Extract H_U: deriv = (perturbed - baseline) / dx_vec[i_unk], shaped (T, ncols).
        for i_targ, targname in enumerate(model.targets):
            t_arr = model.path.__dict__[targname]  # (T, ncols)
            deriv = (t_arr - baseline_targets[i_targ, :, None]) / denom[None, :]
            model.H_U[i_targ * par.T:(i_targ + 1) * par.T, :] = deriv
    finally:
        _restore_path_arrays(model, saved)


def compute_jacobians_complete(model, dx=1e-4, do_print=False, hh_method='fake_news'):
    """
    Compute all Jacobians needed for transition path:
    1. Household Jacobian using age-specific fake-news framework (hh_method='fake_news')
       or exact sequence-space finite differences (hh_method='exact_fd')
    2. Block Jacobians
    3. Accumulate along DAG to get total Jacobians from unknowns to targets

    Args:
        model: HANCHousingModel instance
        dx: Finite difference step size
        do_print: Print progress
        hh_method: 'fake_news' (default) or 'exact_fd'.
            'fake_news' uses the age-specific fake-news algorithm -- fast but
            relies on time-stationarity around the terminal steady state.
            'exact_fd' perturbs each (input, period) pair individually with a
            full backward EGM solve + forward distribution pass.
    """

    par = model.par

    if do_print: print('='*60)
    if do_print: print('COMPUTING JACOBIANS')
    if do_print: print('='*60)

    # Step 1: Household Jacobian
    if hh_method == 'exact_fd':
        if do_print: print('\n1. Household Jacobian (exact sequence-space finite differences):')
        compute_jac_hh_exact(model, dx=dx, do_print=do_print)
        # age_jac_hh is not available with exact_fd
        model.age_jac_hh = {}

    else:
        # 'fake_news' (default): age-specific fake-news framework
        if do_print: print('\n1. Household Jacobian (age-specific fake-news framework):')

        # Keep HH Jacobian local around steady-state shocks for consistency with H_U.
        for shockname in model.shocks:
            model.path.__dict__[shockname][:, 0] = model.ss.__dict__[shockname]

        # Compute age-specific Jacobians for each HH input-output pair,
        # store them on the model, and aggregate to build model.jac_hh.
        model.jac_hh = {}
        model.age_jac_hh = {}

        hh_outputs = ['H_u_hh', 'H_r_hh']  # A_hh not needed: no asset-market target in SOE
        hh_inputs = model.inputs_hh  # ['r', 'w', 'q_u', 'q_r', 'kappa']

        # Use input-specific dx scaled to the SS level so the induced change in
        # m_next (cash-on-hand) is always above the m_grid interpolation resolution.
        # dx = 1e-4 is fine for r (≈0.04) but invisibly small for w (≈97).
        def _dx_for_input(name):
            ss_val = float(getattr(model.ss, name, getattr(model.par, name, 1.0)))
            return max(dx, 2e-3 * abs(ss_val)) if abs(ss_val) >= 1.0 else dx

        for output in hh_outputs:
            for inputname in hh_inputs:
                dx_input = _dx_for_input(inputname)
                if do_print:
                    print(f'  {output:15s} wrt {inputname:8s}  (dx={dx_input:.2e})')
                
                # Compute age-specific Jacobians using fake-news framework
                result = compute_age_specific_hh_jacobians(
                    model=model,
                    inputname=inputname,
                    outputname=output,
                    dx=dx_input,
                    age_horizon=None,  # Use default (min(J, T))
                    selected_shock_ages=None,  # Use all shock ages
                    backward_window=20,  # Default local window
                    do_print=False
                )
    
                # Keep age-specific objects on the model for read-only plotting.
                # Store only fields needed for plotting/inspection to limit memory.
                model.age_jac_hh[(output, inputname)] = {
                    'J_by_age': result['J_by_age'],
                    'age_horizon': result.get('age_horizon', None),
                    'model_horizon': result.get('model_horizon', par.T),
                    'shock_ages': result.get('shock_ages', None),
                    'inputname': inputname,
                    'outputname': output,
                    'baseline_output_by_age': result.get('baseline_output_by_age', None),
                }
                
                # Extract unweighted sum of age-specific Jacobians
                # This is the aggregate Jacobian for the system
                jac_aggregate = result['J_sum_unweighted']
                
                # Store in dictionary with (output, input) tuple as key
                model.jac_hh[(output, inputname)] = jac_aggregate

    # Step 2: Full system Jacobian by finite differences (column-vectorized).
    # Mirrors GEModelTools._compute_jac (GEModelClass.py:1269): one widened
    # DAG sweep instead of N_unknowns*T sequential evaluations.
    if do_print: print('\n2. Full system Jacobian (vectorized over %d columns):' %
                       (len(model.unknowns) * par.T))

    from copy import deepcopy
    path_original = deepcopy(model.path)

    # Initialize H_U (targets w.r.t. unknowns)
    model.H_U = np.zeros((len(model.targets) * par.T, len(model.unknowns) * par.T))

    # Set shocks to steady state on the un-widened path (broadcast preserves this
    # across all columns when _widen_path_for_jac tiles column 0).
    for shockname in model.shocks:
        model.path.__dict__[shockname][:, 0] = model.ss.__dict__[shockname]

    # Baseline evaluation at steady state (un-widened, ncols=1).
    for unknownname in model.unknowns:
        model.path.__dict__[unknownname][:, 0] = model.ss.__dict__[unknownname]

    evaluate_path_blocks(model, use_hh_jac=True)

    baseline_targets = np.zeros((len(model.targets), par.T))
    for i, targetname in enumerate(model.targets):
        baseline_targets[i, :] = model.path.__dict__[targetname][:, 0]

    # Linearize around steady-state values.
    x_baseline = np.zeros((len(model.unknowns), par.T))
    for i, varname in enumerate(model.unknowns):
        x_baseline[i, :] = model.ss.__dict__[varname]

    # Single widened DAG sweep populates all columns of H_U.
    _build_H_U_vectorized(model, baseline_targets, x_baseline, dx, do_print=do_print)

    # Restore path
    model.path = path_original
    if do_print: print('\n' + '='*60)
    if do_print: print('JACOBIANS COMPUTED SUCCESSFULLY')
    if do_print: print('='*60 + '\n')




def broyden_solver(f, x0, jac, tol=1e-8, max_iter=100,
                   max_no_improvement=20, do_print=False, model=None,
                   labor_eq_weight=1.0, damping=0.5, tikhonov=0.0):
    """Quasi-Newton solver using Broyden rank-1 Jacobian updates.

    Implements equation (3.5) from Auclert et al. (2021):
        U^{j+1} = U^j - damping * [H_U(U_ss, Z_ss)]^{-1} H(U^j, Z)

    with Broyden updates to the Jacobian after each step.  

    Args:
        f:      callable, f(x) -> residual vector (len = N_targets * T)
        x0:     initial guess, shape (N_unknowns, T) or flat
        jac:    initial Jacobian (N_targets*T, N_unknowns*T), typically H_U
        tol:    convergence tolerance on max |residual|
        max_iter: maximum Newton iterations
        max_no_improvement: abort after this many iterations without improvement
        do_print: print progress
        model:  optional, for printing unknown diagnostics
        labor_eq_weight: kept for API compatibility; unused in simple solver
        damping: fixed step-size multiplier in (0, 1].  Default 0.5.  Use e.g.
                 0.5 to halve every Newton step.  Useful when the shock is
                 large and the initial Jacobian is a poor approximation
                 (demographic transitions).
        tikhonov: Tikhonov / Levenberg–Marquardt ridge on the Newton step.
             0.0 → standard Newton:  dx = -solve(jac, y).
             float > 0 → relative ridge:
                 dx = -(JᵀJ + λ·smax(J)²·I)⁻¹ Jᵀ y
             where `smax(J)` is the largest singular value of `jac` (so the
             effective regularisation scales with the Jacobian's magnitude
             and `tikhonov` is unit-free).  Values of `1e-6` to `1e-3` are
             typical: large enough to damp near-singular directions in a
             stiff `H_U`, small enough to leave the well-conditioned
             directions unchanged.  Use when `cond(H_U)` is large (e.g. at
             an interior SS where HH demand is highly elastic).

    Returns:
        x: solution vector (flat)
    """

    x = x0.ravel().copy()
    y = f(x)

    abs_diff_min = np.inf
    no_improvement = 0
    for it in range(max_iter):

        abs_diff = np.max(np.abs(y))

        if abs_diff < abs_diff_min:
            no_improvement = 0
            abs_diff_min = abs_diff
        else:
            no_improvement += 1
            if no_improvement > max_no_improvement:
                raise ValueError(
                    f'broyden_solver: No improvement for {max_no_improvement} iterations')

        if do_print:
            print(f' it = {it:3d} -> max. abs. error = {abs_diff:8.2e}')
            if model is not None and len(model.targets) > 1:
                y_ = y.reshape((len(model.targets), -1))
                for i, target in enumerate(model.targets):
                    print(f'   {np.max(np.abs(y_[i])):8.2e} in {target}')

        if abs_diff < tol:
            return x


        # Damped Newton step with catastrophe guard: if the damped step
        # is unsafe (NaN or residual blow-up) we halve alpha up to 4 times.
        def _trial_step(jac_cur):
            if tikhonov > 0.0:
                # Tikhonov-regularized step: dx = -(JᵀJ + λ·smax(J)²·I)⁻¹ Jᵀ y
                JtJ = jac_cur.T @ jac_cur
                scale = float(np.linalg.norm(jac_cur, ord=2))**2
                lam = tikhonov * max(scale, 1e-30)
                dx_cur = np.linalg.solve(JtJ + lam * np.eye(JtJ.shape[0]),
                                         -jac_cur.T @ y)
            else:
                dx_cur = np.linalg.solve(jac_cur, -y)
            alpha_cur = damping
            used_halving_cur = False
            ynew_cur = f(x + alpha_cur * dx_cur)
            for _bt in range(4):
                if np.any(np.isnan(ynew_cur)) or np.max(np.abs(ynew_cur)) > 10.0 * abs_diff + 1.0:
                    alpha_cur *= 0.5
                    used_halving_cur = True
                    ynew_cur = f(x + alpha_cur * dx_cur)
                else:
                    break
            safe_cur = (not np.any(np.isnan(ynew_cur))) and (np.max(np.abs(ynew_cur)) <= 10.0 * abs_diff + 1.0)
            descent_cur = (not np.any(np.isnan(ynew_cur))) and (np.max(np.abs(ynew_cur)) < abs_diff)
            return dx_cur, alpha_cur, ynew_cur, used_halving_cur, safe_cur, descent_cur

        dx, alpha, ynew, used_halving, safe_step, descent_found = _trial_step(jac)

        if np.any(np.isnan(ynew)):
            raise ValueError('broyden_solver: nan in residual after Newton step')


        if not safe_step:
            raise ValueError(
                'broyden_solver: no safe step found after safety halving')

        dx_actual = alpha * dx

        # Broyden rank-1 update on the actual step taken
        dy = ynew - y
        norm_sq = np.linalg.norm(dx_actual) ** 2
        if norm_sq > 0:
            jac = jac + np.outer((dy - jac @ dx_actual) / norm_sq, dx_actual)
        y = ynew
        x = x + dx_actual

    raise ValueError(
        f'broyden_solver: No convergence after {max_iter} iterations (tol={tol:.1e})')



def _set_unknowns(model, x):
    """Write unknown paths from flat vector x into model.path.

    Applies a positivity floor to stock unknowns (H_u, H_r).  Also enforces
    a hard terminal anchor: the last par.n_terminal_anchor periods of every
    unknown are overwritten with their SS value so that the model converges
    cleanly to the terminal steady state.
    """
    par = model.par
    x2d = x.reshape((len(model.unknowns), par.T))
    n_anchor = int(getattr(par, 'n_terminal_anchor', 0))

    for i, varname in enumerate(model.unknowns):
        vals = x2d[i, :]
        if varname in ('K_tilde', 'L_tilde', 'H_u', 'H_r'):
            vals = np.maximum(vals, 1e-10)
        if n_anchor > 0 and par.T > n_anchor:
            vals = vals.copy()
            vals[par.T - n_anchor:] = float(model.ss.__dict__[varname])
        model.path.__dict__[varname][:, 0] = vals


def _get_errors(model):
    """Read target residuals from model.path, return flat vector.

    For the last par.n_terminal_anchor periods the housing-clearing residuals
    are replaced with pin-to-SS conditions  (unknown[t] - ss_val),  so that
    the H_U Jacobian has a simple well-conditioned identity-like block there
    instead of a near-zero column from the pinned (constant) unknowns.
    Convention: targets[i] pairs with unknowns[i]  (clearing_H_u ↔ H_u, etc.).
    """
    par = model.par
    errors = np.zeros((len(model.targets), par.T))
    for i, varname in enumerate(model.targets):
        errors[i, :] = model.path.__dict__[varname][:, 0]

    n_anchor = int(getattr(par, 'n_terminal_anchor', 0))
    if n_anchor > 0 and par.T > n_anchor:
        n_pairs = min(len(model.unknowns), len(model.targets))
        for i in range(n_pairs):
            unk = model.unknowns[i]
            ss_val = float(model.ss.__dict__[unk])
            path_arr = model.path.__dict__[unk]
            errors[i, par.T - n_anchor:] = path_arr[par.T - n_anchor:, 0] - ss_val

    return errors.ravel()


def _evaluate_H(model, x):
    """Set unknowns, evaluate all blocks (linearized HH), return residuals."""
    _set_unknowns(model, x)
    evaluate_path_blocks(model, use_hh_jac=True)
    return _get_errors(model)

def _evaluate_H_nonlinear(model, x):
    """Set unknowns, evaluate all blocks (non-linear HH), return residuals."""
    _set_unknowns(model, x)
    evaluate_path_blocks(model, use_hh_jac=False)
    return _get_errors(model)


def find_transition_path(model, shocks=None, do_print=False, tol=1e-8, max_iter=100,
                         use_path_initial_guess=False, labor_eq_weight=1.0,
                         ini_values=None,
                         damping=0.5, max_no_improvement=None, solve_nonlinear=False,
                         tikhonov=0.0):
    """Find the non-linear transition path using Broyden's method.

    Follows Auclert et al. (2021), equation (3.5):
        U^{j+1} = U^j - [H_U]^{-1} H(U^j, Z)
    with rank-1 Broyden updates to the Jacobian.

    Prerequisites:
        - model.jac_hh must be computed  (household Jacobians)
        - model.H_U must be computed     (system Jacobian of targets w.r.t. unknowns)

    Args:
        model:   HANCHousingModelClass instance with SS solved
        shocks:  dict {varname: path_array} of shock paths (length T),
                 or None to use whatever is already on model.path
        do_print: print iteration progress
        tol:     convergence tolerance
        max_iter: max Broyden iterations
        use_path_initial_guess: if True, initialize unknown paths from current
                 model.path values instead of steady state (useful for continuation)
        labor_eq_weight: multiplicative weight on clearing_L in the Newton
               system (kept for API compatibility; ignored in simple solver)
        max_no_improvement: abort after this many consecutive non-improving
                 iterations. Defaults to max_iter (effectively disabled).
        ini_values: dict {varname: scalar} of initial-period (t=-1) values for
                 lagged variables (e.g. H_u, H_r, K from the initial SS).
                 When model.ss is the *terminal* SS but the economy starts from
                 a different *initial* SS, pass the initial SS values here so
                 that lag() calls in the block functions use the correct t=-1
                 state.  If None, ini defaults to model.ss (standard behaviour).
    """
    from copy import deepcopy
    import time

    par = model.par
    ss = model.ss
    path = model.path

    t0 = time.time()

    # a. Set shock paths
    if shocks is not None:
        for shockname in model.shocks:
            path.__dict__[shockname][:, 0] = ss.__dict__[shockname]
        for varname, shock_path in shocks.items():
            path.__dict__[varname][:, 0] = ss.__dict__[varname] + np.asarray(shock_path).ravel()[:par.T]

    # c. Set ini to terminal SS, then override entries the caller passed
    # in ini_values (initial-period t=-1 lag values).  For a permanent
    # transition from an old SS to the terminal SS, pass the full set of
    # initial-state lags (H_u, H_r, K, A_hh, ...) — the older v3.2 code
    # excluded H_u/H_r here, but that defeats the point of permanent-
    # transition initial conditions for a housing model.
    #
    # If H_u_initial differs strongly from H_u_terminal, IH_u[0] = H_u[0]
    # - (1-delta_H)*ini.H_u may be far off SS scale at the initial guess
    # x0=SS_T.  To keep the first evaluation tractable, the unknowns'
    # initial guess is interpolated below from ini_values toward SS_T at
    # t=T-1 (see "Linear ramp warm-start").
    if hasattr(model, 'ini'):
        for varname in model.varlist:
            if hasattr(ss, varname):
                model.ini.__dict__[varname] = ss.__dict__[varname]

    if hasattr(model, 'ini'):
        for varname in model.varlist:
            if hasattr(ss, varname):
                model.ini.__dict__[varname] = ss.__dict__[varname]
    print(f">>> [A] after ss-reset:        ini.H_u = {model.ini.H_u}")

    if ini_values is not None and hasattr(model, 'ini'):
        print(f">>> ini_values keys: {list(ini_values.keys())[:15]}")
        for varname, val in ini_values.items():
            if hasattr(model.ini, varname):
                model.ini.__dict__[varname] = val
                if varname in ('H_u', 'H_r'):
                    print(f">>>     overrode ini.{varname} = {val}")
            else:
                if varname in ('H_u', 'H_r'):
                    print(f">>>     SKIPPED ini.{varname} (hasattr False)")
    print(f">>> [B] after ini_values override: ini.H_u = {model.ini.H_u}")

    # Freeze ini so evaluate_path_blocks won't overwrite custom ini_values.
    model._ini_frozen = True
    print(f">>> [C] after freeze=True:     ini.H_u = {model.ini.H_u}")


    # Freeze ini so evaluate_path_blocks won't overwrite custom ini_values.
    model._ini_frozen = True

    # Clear any moving anchor left over from a previous call.
    # find_transition_path() evaluates H nonlinearly every iteration,
    # so no HH linear anchor should be active.
    model._anchor_hh = {}
    model._anchor_prices = {}

    # d. Solve with Broyden - linear or non-linear evaluation of H
    if solve_nonlinear:
        obj = lambda x: _evaluate_H_nonlinear(model, x)
    else:
        obj = lambda x: _evaluate_H(model, x)

    # b. Initial guess for unknown paths
    x0 = np.zeros((len(model.unknowns), par.T))
    for i, varname in enumerate(model.unknowns):
        if use_path_initial_guess and hasattr(path, varname):
            arr = np.asarray(path.__dict__[varname])
            if arr.ndim == 2:
                x0[i, :] = arr[:, 0]
            else:
                x0[i, :] = arr.ravel()[:par.T]
        else:
            x0[i, :] = ss.__dict__[varname]

    # Linear ramp warm-start on housing stocks only.  Toggled on/off by commenting/uncommenting
     #   T_ramp = par.T
      #  ramp = np.linspace(0.0, 1.0, T_ramp)
       # for i, varname in enumerate(model.unknowns):
        #    if varname in ('H_u', 'H_r') and varname in ini_values:
         #       v0 = float(ini_values[varname])
          #      vT = float(ss.__dict__[varname])
           #     x0[i, :] = v0 + ramp * (vT - v0)

    #if (not use_path_initial_guess) and (ini_values is not None):
     #   delta_H = float(par.delta_H)
      #  decay = (1.0 - delta_H) ** np.arange(par.T)
       # for i, varname in enumerate(model.unknowns):
        #    if varname in ('H_u', 'H_r') and varname in ini_values:
         #       v0 = float(ini_values[varname])
          #      vT = float(ss.__dict__[varname])
           #     x0[i, :] = vT + (v0 - vT) * decay

    if do_print:
        print(f'Finding transition path ({len(model.unknowns)} unknowns, '
              f'{len(model.targets)} targets, T={par.T}):')

    _max_no_impr = max_iter if max_no_improvement is None else max_no_improvement


    try:
        x = broyden_solver(obj, x0, model.H_U,
                            tol=tol, max_iter=max_iter,
                            do_print=do_print, model=model,
                            labor_eq_weight=labor_eq_weight,
                            damping=damping,
                            max_no_improvement=_max_no_impr,
                            tikhonov=float(tikhonov))
        print(f">>> [D] after broyden_solver:   ini.H_u = {model.ini.H_u}")

        # e. Final evaluation to leave model.path in solved state
        obj(x)
        print(f">>> [E] after final obj(x):     ini.H_u = {model.ini.H_u}")
    finally:
        model._ini_frozen = False
    print(f">>> [F] after finally (unfreeze): ini.H_u = {model.ini.H_u}")


    # f. End-of-path check
    if do_print:
        print(f'\nTerminal value check:')
        for varname in model.unknowns + model.targets:
            ssval = ss.__dict__[varname]
            if np.isnan(ssval):
                continue
            endval = path.__dict__[varname][-1, 0]
            status = '✓' if np.isclose(ssval, endval, rtol=1e-4) else '✗'
            print(f'  {status} {varname:20s}: path[-1] = {endval:12.6f}, ss = {ssval:12.6f}')

        dt = time.time() - t0
        print(f'\nTransition path found in {dt:.1f}s')

