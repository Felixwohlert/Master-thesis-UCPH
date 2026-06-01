"""
steady_state.py

Manual steady state computation for HANCHousingModel.

Structure:
1. solve_hh_ss() - Solve household policies (wraps DC-EGM solver)
2. simulate_hh_ss() - Compute stationary distribution (histogram method)
3. compute_aggregates() - Integrate to get A_hh, C_hh, etc.
4. supply_side() - Compute production and prices directly from equations
5. find_ss() - Root finding on asset market clearing (like brentq in notebook)

"""

import numpy as np
import household_problem
from numba import njit
from scipy.sparse import coo_matrix

# ==========================================
# 0. UTILITY HELPERS
# ==========================================

def _pr_choices_from_pr_urban(pr_urban, Nz, Nh, Nm):
    """Fallback: build (4, Nz, Nh, Nm) pr_choices from a marginal pr_urban array.

    Used when sol.pr_choices is not available (e.g. old checkpoints).
    Splits urban probability equally between urban_renter (1) and urban_owner (3),
    and rural probability equally between rural_renter (0) and rural_owner (2).
    """
    pr_ch = np.zeros((4, Nz, Nh, Nm))
    pr_ch[0] = (1.0 - pr_urban) * 0.5  # rural_renter
    pr_ch[2] = (1.0 - pr_urban) * 0.5  # rural_owner
    pr_ch[1] = pr_urban * 0.5           # urban_renter
    pr_ch[3] = pr_urban * 0.5           # urban_owner
    return pr_ch


# ==========================================
# 1. NUMBA-OPTIMIZED HELPERS
# ==========================================

@njit(cache=True)
def cash_on_hand_numba(a_today, z_shock, idc_current, idc_next, r, w, q_u, q_r,
                       zeta, zeta_renter, h_u, h_r, h_l, f_u, f_r, lambda_ltv, T_mort, delta_H,
                       tau_wealth, tau_profits):
    """Numba-optimized cash-on-hand computation.

    Housing states: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner.

    Mortgage mechanics:
      All owners hold mortgage at LTV lambda: b = lambda_ltv * q^j * h^j.
      Amortisation = lambda_ltv * q^j * h^j / T_mort per period.
      Total owner housing cost = lambda_ltv * q^j * h^j * (r + 1/T_mort).
      Renters pay rent f^j * h^j.
    """
    base = (1.0 + r) * a_today + w * np.exp(z_shock)

    # Ongoing housing cost for the next-period state
    is_urban_next = (idc_next % 2 == 1)  # states 1 (urban_renter) and 3 (urban_owner)
    is_owner_next = (idc_next >= 2)       # states 2, 3
    q_next = q_u if is_urban_next else q_r
    h_next = h_u if is_urban_next else h_r
    f_next = f_u if is_urban_next else f_r

    if is_owner_next:
        # Mortgage interest+amortisation plus a per-period wealth/property tax
        # on the house held as owner (tau_wealth=0 in the baseline).
        housing_cost = (lambda_ltv * q_next * h_next * (r + 1.0 / T_mort)
                        + tau_wealth * q_next * h_next)
    else:
        housing_cost = f_next * h_l  # renter pays for h_l units of housing

    # One-time adjustment from buying/selling
    adj = 0.0
    if idc_next != idc_current:
        # Selling proceeds: only if currently an owner
        if idc_current >= 2:
            is_urban_curr = (idc_current == 3)
            q_curr = q_u if is_urban_curr else q_r
            h_curr = h_u if is_urban_curr else h_r
            # Profit tax (tau_profits=0 baseline) retains a share of sale proceeds.
            adj += (1.0 - tau_profits) * zeta * q_curr * h_curr

        # Buying cost: only if becoming an owner
        if idc_next >= 2:
            is_urban_nx = (idc_next % 2 == 1)
            q_nx = q_u if is_urban_nx else q_r
            h_nx = h_u if is_urban_nx else h_r
            adj -= (1.0 - tau_profits) * zeta * q_nx * h_nx

        # Renter migration cost: both current and next are renters (states 0,1)
        # and the region differs (rural_renter <-> urban_renter).
        if idc_current < 2 and idc_next < 2:
            adj -= zeta_renter * w

    return base - housing_cost + adj



# Should speed up the transition matrix construction significantly.
@njit(cache=True)
def build_transition_coo(
    a_policy, pr_choices, z_trans, z_grid, m_grid,
    r, w, q_u, q_r, cfloor, zeta, zeta_renter, h_u, h_r, h_l,
    f_u, f_r, lambda_ltv, T_mort, delta_H,
    tau_wealth, tau_profits,
    max_transitions
):
    """
    Build COO sparse matrix components for the 4-state transition operator.

    Housing states: 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner.

    Arguments
    ---------
    a_policy  : (Nz, Nh, Nm)   – savings policy
    pr_choices: (4, Nz, Nh, Nm) – choice probabilities for each of 4 next states
    z_trans   : (Nz, Nz)        – productivity transition matrix
    z_grid    : (Nz,)
    m_grid    : (Nm,)
    f_u, f_r  : scalars         – regional rents from rental_sector block
    lambda_ltv: scalar          – LTV ratio for mortgages
    T_mort    : scalar          – mortgage repayment horizon (periods)
    max_transitions : pre-allocated buffer size

    Returns: (rows, cols, data, nnz)
    """
    Nz, Nh, Nm = a_policy.shape  # Nh = 4

    rows = np.zeros(max_transitions, dtype=np.int32)
    cols = np.zeros(max_transitions, dtype=np.int32)
    data = np.zeros(max_transitions, dtype=np.float64)

    nnz = 0

    for i_z in range(Nz):
        for i_h in range(Nh):
            for i_m in range(Nm):

                col = i_z * Nh * Nm + i_h * Nm + i_m
                a_next = a_policy[i_z, i_h, i_m]

                for i_h_next in range(4):  # loop over all 4 housing choices
                    pr_ch = pr_choices[i_h_next, i_z, i_h, i_m]
                    if pr_ch <= 0.0:
                        continue

                    for i_z_next in range(Nz):
                        p_z = z_trans[i_z, i_z_next]
                        if p_z == 0.0:
                            continue

                        z_shock = z_grid[i_z_next]

                        m_next = cash_on_hand_numba(
                            a_next, z_shock, i_h, i_h_next, r, w, q_u, q_r,
                            zeta, zeta_renter, h_u, h_r, h_l, f_u, f_r, lambda_ltv, T_mort, delta_H, tau_wealth, tau_profits
                        )
                        m_next = max(cfloor, m_next)

                        idx_lo = np.searchsorted(m_grid, m_next) - 1
                        idx_lo = max(0, min(idx_lo, Nm - 2))
                        idx_hi = idx_lo + 1

                        denom = m_grid[idx_hi] - m_grid[idx_lo]
                        if denom > 0.0:
                            w_lo = (m_grid[idx_hi] - m_next) / denom
                        else:
                            w_lo = 1.0

                        w_lo = max(0.0, min(w_lo, 1.0))
                        w_hi = 1.0 - w_lo

                        mass = p_z * pr_ch

                        row_lo = i_z_next * Nh * Nm + i_h_next * Nm + idx_lo
                        row_hi = i_z_next * Nh * Nm + i_h_next * Nm + idx_hi

                        if nnz >= max_transitions - 1:
                            break

                        rows[nnz] = row_lo
                        cols[nnz] = col
                        data[nnz] = mass * w_lo
                        nnz += 1

                        rows[nnz] = row_hi
                        cols[nnz] = col
                        data[nnz] = mass * w_hi
                        nnz += 1

    return rows[:nnz], cols[:nnz], data[:nnz], nnz


@njit(cache=True)
def scatter_dcegm_forward(
    D_in, D_out,
    a_policy, pr_choices, z_trans, z_grid, m_grid,
    r, w, q_u, q_r, cfloor, zeta, zeta_renter, h_u, h_r, h_l,
    f_u, f_r, lambda_ltv, T_mort, delta_H,
    tau_wealth, tau_profits,
):
    """Forward-propagate D_in -> D_out under the 4-state DC-EGM transition.

    Direct-scatter analogue of build_transition_coo + sparse matvec: writes
    D_out directly without ever materialising a sparse matrix.

    D_in, D_out:  (Nz, Nh, Nm)
    a_policy:     (Nz, Nh, Nm)
    pr_choices:   (4, Nz, Nh, Nm)
    z_trans:      (Nz, Nz)
    z_grid, m_grid: 1-D
    """
    Nz, Nh, Nm = D_in.shape

    for iz in range(Nz):
        for ih in range(Nh):
            for im in range(Nm):
                D_out[iz, ih, im] = 0.0

    for i_z in range(Nz):
        for i_h in range(Nh):
            for i_m in range(Nm):

                D_ = D_in[i_z, i_h, i_m]
                if D_ == 0.0:
                    continue

                a_next = a_policy[i_z, i_h, i_m]

                for i_h_next in range(4):
                    pr_ch = pr_choices[i_h_next, i_z, i_h, i_m]
                    if pr_ch <= 0.0:
                        continue

                    for i_z_next in range(Nz):
                        p_z = z_trans[i_z, i_z_next]
                        if p_z == 0.0:
                            continue

                        z_shock = z_grid[i_z_next]

                        m_next = cash_on_hand_numba(
                            a_next, z_shock, i_h, i_h_next, r, w, q_u, q_r,
                            zeta, zeta_renter, h_u, h_r, h_l, f_u, f_r, lambda_ltv, T_mort, delta_H, tau_wealth, tau_profits,
                        )
                        if m_next < cfloor:
                            m_next = cfloor

                        idx_lo = np.searchsorted(m_grid, m_next) - 1
                        if idx_lo < 0:
                            idx_lo = 0
                        elif idx_lo > Nm - 2:
                            idx_lo = Nm - 2
                        idx_hi = idx_lo + 1

                        denom = m_grid[idx_hi] - m_grid[idx_lo]
                        if denom > 0.0:
                            w_lo = (m_grid[idx_hi] - m_next) / denom
                        else:
                            w_lo = 1.0
                        if w_lo < 0.0:
                            w_lo = 0.0
                        elif w_lo > 1.0:
                            w_lo = 1.0
                        w_hi = 1.0 - w_lo

                        mass = D_ * pr_ch * p_z

                        D_out[i_z_next, i_h_next, idx_lo] += mass * w_lo
                        D_out[i_z_next, i_h_next, idx_hi] += mass * w_hi


@njit(cache=True)
def build_transition_coo_v2(
    a_policy, pr_choices_asav, a_grid, z_trans, z_grid, m_grid,
    r, w, q_u, q_r, cfloor, zeta, zeta_renter, h_u, h_r, h_l,
    f_u, f_r, lambda_ltv, T_mort, delta_H,
    tau_wealth, tau_profits,
    max_transitions
):
    """
    Build COO sparse matrix using z_next-conditional choice probabilities.

    Correctly captures the correlation between z' and h' by looking up
    P(h_next | a_next, z_next) instead of the z'-averaged E_{z'}[P(h_next | a, z')].

    Arguments
    ---------
    a_policy        : (Nz, Nh, Nm)        – savings policy on m_grid
    pr_choices_asav : (4, Nz_next, Nh, Na_sav) – P(h_next | savings=a, z_next)
    a_grid          : (Na_sav,)            – savings grid (uniform, same as par.a_grid)
    z_trans         : (Nz, Nz)            – row-stochastic productivity transition
    z_grid          : (Nz,)
    m_grid          : (Nm,)               – regular cash-on-hand grid
    """
    Nz, Nh, Nm = a_policy.shape
    Na_sav = a_grid.shape[0]

    a_lo = a_grid[0]
    a_hi = a_grid[Na_sav - 1]
    da = (a_hi - a_lo) / (Na_sav - 1) if Na_sav > 1 else 1.0

    rows = np.zeros(max_transitions, dtype=np.int32)
    cols = np.zeros(max_transitions, dtype=np.int32)
    data = np.zeros(max_transitions, dtype=np.float64)
    nnz = 0

    for i_z in range(Nz):
        for i_h in range(Nh):
            for i_m in range(Nm):

                col = i_z * Nh * Nm + i_h * Nm + i_m
                a_next = a_policy[i_z, i_h, i_m]

                # Linear interpolation index on the uniform savings grid
                if da > 0.0:
                    idx_a = int((a_next - a_lo) / da)
                else:
                    idx_a = 0
                idx_a = max(0, min(idx_a, Na_sav - 2))
                denom_a = a_grid[idx_a + 1] - a_grid[idx_a]
                if denom_a > 0.0:
                    frac_a = (a_next - a_grid[idx_a]) / denom_a
                else:
                    frac_a = 0.0
                frac_a = max(0.0, min(frac_a, 1.0))

                for i_h_next in range(4):
                    for i_z_next in range(Nz):
                        p_z = z_trans[i_z, i_z_next]
                        if p_z == 0.0:
                            continue

                        # P(h_next | a_next, z_next): linear interp on savings grid
                        pr_ch = (pr_choices_asav[i_h_next, i_z_next, i_h, idx_a] * (1.0 - frac_a)
                                 + pr_choices_asav[i_h_next, i_z_next, i_h, idx_a + 1] * frac_a)
                        if pr_ch <= 0.0:
                            continue

                        z_shock = z_grid[i_z_next]
                        m_next = cash_on_hand_numba(
                            a_next, z_shock, i_h, i_h_next, r, w, q_u, q_r,
                            zeta, zeta_renter, h_u, h_r, h_l, f_u, f_r, lambda_ltv, T_mort, delta_H, tau_wealth, tau_profits
                        )
                        m_next = max(cfloor, m_next)

                        idx_lo = np.searchsorted(m_grid, m_next) - 1
                        idx_lo = max(0, min(idx_lo, Nm - 2))
                        idx_hi = idx_lo + 1

                        denom = m_grid[idx_hi] - m_grid[idx_lo]
                        if denom > 0.0:
                            w_lo = (m_grid[idx_hi] - m_next) / denom
                        else:
                            w_lo = 1.0

                        w_lo = max(0.0, min(w_lo, 1.0))
                        w_hi = 1.0 - w_lo
                        mass = p_z * pr_ch

                        row_lo = i_z_next * Nh * Nm + i_h_next * Nm + idx_lo
                        row_hi = i_z_next * Nh * Nm + i_h_next * Nm + idx_hi

                        if nnz >= max_transitions - 1:
                            break

                        rows[nnz] = row_lo
                        cols[nnz] = col
                        data[nnz] = mass * w_lo
                        nnz += 1

                        rows[nnz] = row_hi
                        cols[nnz] = col
                        data[nnz] = mass * w_hi
                        nnz += 1

    return rows[:nnz], cols[:nnz], data[:nnz], nnz


# ==========================================
# 2. HOUSEHOLD BLOCK - STEADY STATE
# ==========================================

def solve_hh_ss(model, do_print=False, tol=1e-6, max_iter=100):
    """
    Solve household problem in steady state by iterating DC-EGM until convergence.

    """
    
    par = model.par
    ss = model.ss
    sol = model.sol
    old_a = sol.a.copy()
    
    # Iterate backward solution until convergence
    it = 0
    while it < max_iter:
        
        # a. Solving DC-EGM once
        household_problem.solve_hh_backwards(model)
        
        # b. Check convergence
        max_diff = np.max(np.abs(sol.a - old_a))
        
        if do_print and it % 10 == 0:
            print(f'  Iteration {it}: max policy change = {max_diff:.2e}')
        
        if max_diff < tol:
            if do_print:
                print(f'Household policies converged in {it} iterations')
            break
        
        # c. next iteration
        old_a = sol.a.copy()
        it += 1
    
    if it >= max_iter:
        print(f'WARNING: solve_hh_ss did not converge after {max_iter} iterations')


def _normalize_cohort_weights(raw_weights, J):
    """Return non-negative cohort weights summing to one (length J)."""
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


def _get_ss_cohort_weights(par, ss, J):
    """Get steady-state cohort weights (fallback: uniform)."""
    if hasattr(ss, 'cohort_weights'):
        return _normalize_cohort_weights(ss.cohort_weights, J)
    if hasattr(par, 'cohort_weights_ss'):
        return _normalize_cohort_weights(par.cohort_weights_ss, J)
    return np.ones(J) / J


def _get_newborn_distribution(par, ss, Nz, Nh, Nm):
    """Get newborn state distribution over (z,h,m), normalized.

    With 4 housing states (0=rural_renter,1=urban_renter,2=rural_owner,3=urban_owner),
    newborns enter as renters (states 0 and 1) since they have no prior ownership.
    """
    if hasattr(ss, 'D_birth') and isinstance(ss.D_birth, np.ndarray) and ss.D_birth.shape == (Nz, Nh, Nm):
        D_birth = ss.D_birth.copy()
        s = np.sum(D_birth)
        if s > 0.0 and np.isfinite(s):
            return D_birth / s

    D_birth = np.zeros((Nz, Nh, Nm))
    z_erg = par.z_ergodic[0] if par.z_ergodic.ndim == 2 else par.z_ergodic
    newborn_urban_share = float(np.clip(getattr(par, 'newborn_urban_share', 0.0), 0.0, 1.0))
    # Newborns enter as renters: rural_renter=0, urban_renter=1
    D_birth[:, 0, 0] = z_erg * (1.0 - newborn_urban_share)  # rural renters
    D_birth[:, 1, 0] = z_erg * newborn_urban_share            # urban renters
    s = np.sum(D_birth)
    if s > 0.0:
        D_birth /= s
    return D_birth


def _policy_arrays_from_endog(policy_endog, age_idx, par, Nh, Nz, Nm):
    """Interpolate endogenous-grid policy at age_idx to regular m-grid."""
    a_pol = np.zeros((Nz, Nh, Nm))
    c_pol = np.zeros((Nz, Nh, Nm))
    pr_choices_pol = np.zeros((4, Nz, Nh, Nm))  # 4-way choice probs

    for i_h in range(Nh):
        pol_h = policy_endog.get(i_h, {})
        if age_idx in pol_h:
            pol_age = pol_h[age_idx]
        elif 0 in pol_h:
            pol_age = pol_h[0]
        else:
            continue

        # Detect z-conditional structure: pol_age = {0: {'m':..., 'c':..., ...}, 1: {...}, ...}
        # vs flat structure: pol_age = {'m':..., 'c':..., ...}
        is_z_conditional = (isinstance(pol_age, dict) and len(pol_age) > 0
                            and isinstance(next(iter(pol_age)), int))

        if is_z_conditional:
            for i_z in range(Nz):
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
                # Legacy fallback: pr_rural stored as scalar
                pr_r_line = np.interp(par.m_grid, pol['m'], pol.get('pr_rural', np.full(len(pol['m']), 0.5)))
                pr_u_line = np.clip(1.0 - pr_r_line, 0.0, 1.0)
                pr_choices_pol[0, :, i_h, :] = pr_r_line  # rural_renter as proxy
                pr_choices_pol[1, :, i_h, :] = pr_u_line  # urban_renter as proxy
            for i_z in range(Nz):
                a_pol[i_z, i_h, :] = a_line
                c_pol[i_z, i_h, :] = c_line

    return a_pol, c_pol, pr_choices_pol


def simulate_hh_ss(model, do_print=False, tol=1e-10, max_iter=10000, use_numba=True, use_transition_cache=True, warm_start=True):

    par = model.par
    ss = model.ss
    sol = model.sol

    Nz = par.Nz
    Nm = par.m_grid.size
    Nh = par.Nh  # 4 housing states: 0=rural_renter,1=urban_renter,2=rural_owner,3=urban_owner

    # a. Unpacking grids
    m_grid = par.m_grid
    z_trans = par.z_trans[0] if par.z_trans.ndim == 3 else par.z_trans

    # b. Unpacking policies
    a_policy = sol.a[0]  # (Nz, Nh, Nm)
    # Use z_next-conditional probs (V2.2 fix) when available; fall back to z_curr-averaged
    if hasattr(sol, 'pr_choices_asav'):
        # (Nfix, Nz_next, Nh, 4, Na_sav) -> (4, Nz_next, Nh, Na_sav)
        pr_choices_asav = np.transpose(sol.pr_choices_asav[0], (2, 0, 1, 3))
        use_v2 = True
    elif hasattr(sol, 'pr_choices'):
        pr_choices_asav = np.transpose(sol.pr_choices[0], (2, 0, 1, 3))  # (4, Nz, Nh, Nm)
        use_v2 = False
    else:
        pr_choices_asav = _pr_choices_from_pr_urban(sol.pr_urban[0], Nz, Nh, Nm)
        use_v2 = False

    # c. unpack discretized productivity grid
    z_grid = np.asarray(par.z_grid)

    # d. building transition matrix
    Nstate = Nz * Nh * Nm

    def index(i_z, i_h, i_m):
        return i_z * Nh * Nm + i_h * Nm + i_m

    # e. Lightweight signature to detect if transition operator can be reused
    transition_signature = (
        Nz,
        Nm,
        Nh,
        float(ss.r),
        float(ss.w),
        float(ss.q_u),
        float(ss.q_r),
        float(getattr(ss, 'f_u', 0.0)),
        float(getattr(ss, 'f_r', 0.0)),
        float(par.cfloor),
        float(np.sum(a_policy)),
        float(np.sum(pr_choices_asav)),
    )

    P = None
    if use_transition_cache and hasattr(ss, '_P_cache') and hasattr(ss, '_P_cache_signature'):
        if ss._P_cache_signature == transition_signature:
            P = ss._P_cache
            if do_print:
                print('Reusing cached transition matrix.')

    if P is None:
        if use_numba:
            if do_print:
                print('Building transition matrix (Numba)...')

            max_transitions = Nstate * Nz * 4 * 2 + 1000  # 4 choices
            if use_v2:
                rows, cols, data, _ = build_transition_coo_v2(
                    a_policy, pr_choices_asav, par.a_grid, z_trans, z_grid, m_grid,
                    ss.r, ss.w, ss.q_u, ss.q_r, par.cfloor,
                    par.zeta, float(getattr(par, 'zeta_renter', 0.0)), par.h_u, par.h_r, float(getattr(par, 'h_l', par.h_r)),
                    float(getattr(ss, 'f_u', ss.q_u * ((1.0 + ss.r) - ((1- par.theta) * (1.0 - par.delta_H))))),
                    float(getattr(ss, 'f_r', ss.q_r * ((1.0 + ss.r) - ((1- par.theta) * (1.0 - par.delta_H))))),
                    float(par.lambda_ltv), float(par.T_mort), float(par.delta_H),
                    float(getattr(par, 'tau_wealth', 0.0)), float(getattr(par, 'tau_profits', 0.0)),
                    max_transitions
                )
            else:
                rows, cols, data, _ = build_transition_coo(
                    a_policy, pr_choices_asav, z_trans, z_grid, m_grid,
                    ss.r, ss.w, ss.q_u, ss.q_r, par.cfloor,
                    par.zeta, float(getattr(par, 'zeta_renter', 0.0)), par.h_u, par.h_r, float(getattr(par, 'h_l', par.h_r)),
                    float(getattr(ss, 'f_u', ss.q_u * ((1.0 + ss.r) - ((1- par.theta) * (1.0 - par.delta_H))))),
                    float(getattr(ss, 'f_r', ss.q_r * ((1.0 + ss.r) - ((1- par.theta) * (1.0 - par.delta_H))))),
                    float(par.lambda_ltv), float(par.T_mort), float(par.delta_H),
                    float(getattr(par, 'tau_wealth', 0.0)), float(getattr(par, 'tau_profits', 0.0)),
                    max_transitions
                )
            P = coo_matrix((data, (rows, cols)), shape=(Nstate, Nstate)).tocsr()
        else:
            if do_print:
                print('Building transition matrix (Python loops — slow)...')

            hh = household_problem.HousingModel(model)
            rows_list = []
            cols_list = []
            data_list = []

            for i_z in range(Nz):
                for i_h in range(Nh):  # 4 housing states
                    for i_m in range(Nm):
                        col = index(i_z, i_h, i_m)
                        a_next = a_policy[i_z, i_h, i_m]

                        for i_h_next in range(4):
                            # z_next-conditional interpolation on savings grid
                            a_lo_py = float(par.a_grid[0])
                            da_py = float(par.a_grid[1] - par.a_grid[0]) if len(par.a_grid) > 1 else 1.0
                            idx_a_py = int((a_next - a_lo_py) / da_py) if da_py > 0 else 0
                            idx_a_py = int(np.clip(idx_a_py, 0, len(par.a_grid) - 2))
                            denom_a_py = float(par.a_grid[idx_a_py + 1] - par.a_grid[idx_a_py])
                            frac_a_py = float((a_next - par.a_grid[idx_a_py]) / denom_a_py) if denom_a_py > 0 else 0.0
                            frac_a_py = float(np.clip(frac_a_py, 0.0, 1.0))

                            for i_z_next in range(Nz):
                                p_z = z_trans[i_z, i_z_next]
                                if p_z == 0:
                                    continue

                                if use_v2:
                                    pr_ch = float(
                                        pr_choices_asav[i_h_next, i_z_next, i_h, idx_a_py] * (1.0 - frac_a_py)
                                        + pr_choices_asav[i_h_next, i_z_next, i_h, idx_a_py + 1] * frac_a_py
                                    )
                                else:
                                    pr_ch = float(pr_choices_asav[i_h_next, i_z, i_h, i_m])
                                if pr_ch <= 0.0:
                                    continue

                                z_shock = z_grid[i_z_next]
                                m_next = hh._cash_on_hand(
                                    a_next, z_shock, i_h, i_h_next,
                                    ss.r, ss.w, ss.q_u, ss.q_r,
                                    float(getattr(ss, 'f_u', ss.q_u * ((1.0 + ss.r) - ((1- par.theta) * (1.0 - par.delta_H))))),
                                    float(getattr(ss, 'f_r', ss.q_r * ((1.0 + ss.r) - ((1- par.theta) * (1.0 - par.delta_H)))))
                                )
                                m_next = max(par.cfloor, m_next)

                                idx_lo = np.searchsorted(m_grid, m_next, side='right') - 1
                                idx_lo = np.clip(idx_lo, 0, Nm - 2)
                                idx_hi = idx_lo + 1

                                denom = m_grid[idx_hi] - m_grid[idx_lo]
                                w_lo = (m_grid[idx_hi] - m_next) / denom if denom > 0 else 1.0
                                w_lo = np.clip(w_lo, 0.0, 1.0)
                                w_hi = 1.0 - w_lo
                                mass = p_z * pr_ch

                                row_lo = index(i_z_next, i_h_next, idx_lo)
                                row_hi = index(i_z_next, i_h_next, idx_hi)

                                rows_list.append(row_lo); cols_list.append(col); data_list.append(mass * w_lo)
                                rows_list.append(row_hi); cols_list.append(col); data_list.append(mass * w_hi)

            P = coo_matrix((data_list, (rows_list, cols_list)), shape=(Nstate, Nstate)).tocsr()

        if use_transition_cache:
            ss._P_cache = P
            ss._P_cache_signature = transition_signature

    if do_print:
        print('Transition matrix ready.')




    # -------------------------------------------------
    # Simulate distribution by iterating on P until convergence
    # -------------------------------------------------

    # a. Reuse previous distribution as initial condition when available
    if warm_start and hasattr(ss, 'D') and isinstance(ss.D, np.ndarray) and ss.D.shape == (Nz, Nh, Nm):
        D = ss.D.reshape(-1).copy()
        s = np.sum(D)
        if s > 0.0 and np.isfinite(s):
            D /= s
        else:
            D = np.zeros(Nstate)
    else:
        D = np.zeros(Nstate)

    # b. fallback initial guess: 
    if np.sum(D) == 0.0:
        for i_z in range(Nz):
            for i_h in range(Nh):
                for i_m in range(Nm):
                    idx = index(i_z, i_h, i_m)
                    D[idx] = par.z_ergodic[0, i_z] / (Nh * Nm)

    for it in range(max_iter):

        D_new = P @ D
        err = np.max(np.abs(D_new - D))

        if do_print and it % 100 == 0:
            print(f"Iter {it}, error = {err:.2e}")

        if err < tol:
            if do_print:
                print(f"Distribution converged in {it} iterations.")
            break

        D = D_new

    # c. reshape back
    ss.D = D.reshape((Nz, Nh, Nm))

    # d. Compute aggregates
    a_policy = sol.a[0]  # (Nz, Nh, Nm)
    c_policy = sol.c[0]

    # hh_scale multiplies all aggregates: represents population size relative to
    # the 1992 base (par.hh_scale = 1.0 by default → no-op for the 1992 SS).
    # In the terminal SS, set par.hh_scale = N_T / N_1992 so that aggregate
    # housing demand scales with the 2069 population level.
    _hh_scale = float(getattr(par, 'hh_scale', 1.0))

    ss.A_hh = _hh_scale * np.sum(ss.D * a_policy)
    ss.C_hh = _hh_scale * np.sum(ss.D * c_policy)

    # e. Housing aggregates: urban = states 1 (urban_renter) and 3 (urban_owner)
    #                         rural = states 0 (rural_renter) and 2 (rural_owner)
    #    Renters occupy h_l units; owners occupy h_u (urban) or h_r (rural).
    h_l = float(getattr(par, 'h_l', par.h_r))
    ss.H_u_hh = _hh_scale * (np.sum(ss.D[:, 1, :]) * h_l + np.sum(ss.D[:, 3, :]) * par.h_u)
    ss.H_r_hh = _hh_scale * (np.sum(ss.D[:, 0, :]) * h_l + np.sum(ss.D[:, 2, :]) * par.h_r)
    ss.H_u_owner_hh = _hh_scale * np.sum(ss.D[:, 3, :]) * par.h_u
    ss.H_r_owner_hh = _hh_scale * np.sum(ss.D[:, 2, :]) * par.h_r

    if do_print:
        print(f'Aggregates: A_hh={ss.A_hh:.4f}, C_hh={ss.C_hh:.4f}')


# ==========================================================================
# OLG-correct steady-state aggregator
# ==========================================================================

def simulate_hh_ss_olg(model, do_print=False):
    """OLG-consistent steady-state aggregator.

    Closed-form OLG SS computation:
      D_age[0]      = D_birth
      D_age[a+1]    = P[a] @ D_age[a]            (age-specific policy)
      ss.D          = Σ_a omega_ss[a] · D_age[a]
      ss.<output>   = Σ_a omega_ss[a] · (loading_a @ D_age[a])

    Replaces the infinite-horizon `simulate_hh_ss` which uses a single
    representative policy and iterates D = P @ D to a Markov fixed point —
    which is correct for a no-aging Bewley model but wrong for OLG.

    Differences vs `simulate_hh_ss`:
      - No fixed-point iteration (OLG SS has a closed-form forward sweep).
      - Age-indexed policies obtained from `solve_dcegm(prices_path=None)`,
        which does proper backward induction with terminal age = J-1.
      - Aggregation weighted by `cohort_weights` (`omega_ss[a]`).
      - Writes `ss.D_cohort` (J, Nz, Nh, Nm) for use as initial cross-section
        in the transition (model.ini.D_cohort fallback chain).

    Aggregates written: ss.D, ss.D_cohort, ss.D_birth, ss.A_hh, ss.C_hh,
    ss.H_u_hh, ss.H_r_hh, ss.H_u_owner_hh, ss.H_r_owner_hh.
    """
    par = model.par
    ss = model.ss

    Nz = par.Nz
    Nh = par.Nh
    Nm = par.m_grid.size
    Nstate = Nz * Nh * Nm

    J = int(getattr(par, 'J', 1))
    J = max(1, J)

    omega_ss = _get_ss_cohort_weights(par, ss, J)
    D_birth = _get_newborn_distribution(par, ss, Nz, Nh, Nm)

    # --- Age-indexed policies from a SS backward sweep -----------------------
    hh = household_problem.HousingModel(model)
    policy_endog, _ = hh.solve_dcegm(prices_path=None)

    # SS prices for building per-age transition matrices
    r_ss = float(ss.r)
    w_ss = float(ss.w)
    q_u_ss = float(ss.q_u)
    q_r_ss = float(ss.q_r)
    f_u_ss = float(getattr(ss, 'f_u',
                           q_u_ss * ((1.0 + r_ss) - (1.0 - par.theta) * (1.0 - par.delta_H))))
    f_r_ss = float(getattr(ss, 'f_r',
                           q_r_ss * ((1.0 + r_ss) - (1.0 - par.theta) * (1.0 - par.delta_H))))

    z_grid = np.asarray(par.z_grid)
    z_trans = par.z_trans[0] if par.z_trans.ndim == 3 else par.z_trans
    max_transitions = Nstate * Nz * 4 * 2 + 1000
    h_l = float(getattr(par, 'h_l', par.h_r))

    # --- Forward sweep of cohorts --------------------------------------------
    D_cohort = np.zeros((J, Nz, Nh, Nm))
    D_cohort[0] = D_birth.copy()

    for age in range(J - 1):
        a_pol, c_pol, pr_pol = _policy_arrays_from_endog(
            policy_endog, age, par, Nh, Nz, Nm,
        )
        chi_age = float(par.chi[min(age + 1, par.T - 1)]) if hasattr(par, 'chi') else 0.0
        rows, cols, data, _ = build_transition_coo(
            a_pol, pr_pol, z_trans, z_grid, par.m_grid,
            r_ss, w_ss * np.exp(chi_age), q_u_ss, q_r_ss, par.cfloor,
            par.zeta, float(getattr(par, 'zeta_renter', 0.0)),
            par.h_u, par.h_r, h_l, f_u_ss, f_r_ss,
            par.lambda_ltv, float(par.T_mort), float(par.delta_H),
            float(getattr(par, 'tau_wealth', 0.0)), float(getattr(par, 'tau_profits', 0.0)),
            max_transitions,
        )
        P = coo_matrix((data, (rows, cols)), shape=(Nstate, Nstate)).tocsr()
        D_next = (P @ D_cohort[age].reshape(-1)).reshape((Nz, Nh, Nm))
        s = np.sum(D_next)
        if s > 0.0 and np.isfinite(s):
            D_cohort[age + 1] = D_next / s
        else:
            D_cohort[age + 1] = D_birth.copy()

    # --- Aggregate distribution ----------------------------------------------
    D_agg = np.zeros((Nz, Nh, Nm))
    for a in range(J):
        D_agg += omega_ss[a] * D_cohort[a]

    ss.D = D_agg
    ss.D_cohort = D_cohort
    ss.D_birth = D_birth.copy()

    # --- OLG aggregates -------------------------------------------------------
    _hh_scale = float(getattr(par, 'hh_scale', 1.0))
    A_total = 0.0
    C_total = 0.0
    H_u_total = 0.0
    H_r_total = 0.0
    H_u_owner_total = 0.0
    H_r_owner_total = 0.0

    for age in range(J):
        a_pol, c_pol, _ = _policy_arrays_from_endog(
            policy_endog, age, par, Nh, Nz, Nm,
        )
        D_a = D_cohort[age]
        w = omega_ss[age]
        A_total += w * float(np.sum(D_a * a_pol))
        C_total += w * float(np.sum(D_a * c_pol))
        H_u_total += w * (float(np.sum(D_a[:, 1, :])) * h_l
                          + float(np.sum(D_a[:, 3, :])) * par.h_u)
        H_r_total += w * (float(np.sum(D_a[:, 0, :])) * h_l
                          + float(np.sum(D_a[:, 2, :])) * par.h_r)
        H_u_owner_total += w * float(np.sum(D_a[:, 3, :])) * par.h_u
        H_r_owner_total += w * float(np.sum(D_a[:, 2, :])) * par.h_r

    ss.A_hh = _hh_scale * A_total
    ss.C_hh = _hh_scale * C_total
    ss.H_u_hh = _hh_scale * H_u_total
    ss.H_r_hh = _hh_scale * H_r_total
    ss.H_u_owner_hh = _hh_scale * H_u_owner_total
    ss.H_r_owner_hh = _hh_scale * H_r_owner_total

    if do_print:
        print(f'OLG aggregates: A_hh={ss.A_hh:.4f}, C_hh={ss.C_hh:.4f}, '
              f'H_u_hh={ss.H_u_hh:.4f}, H_r_hh={ss.H_r_hh:.4f}')


# ==========================================
# 2. GENERAL EQUILIBRIUM
# ==========================================

def household_ss(model, do_print=False, catch_errors=False):
    """
    Solve household block in steady state.
    
    This solves the household problem (iterating policies to convergence),
    then simulates the stationary distribution, and aggregates results.
    
    Parameters:
    - model: HANCHousingModelClass instance
    - do_print: print progress
    - catch_errors: if False, re-raise exceptions; if True, catch and return NaN
                   but still log the error for debugging
    
    Returns:
    - A_hh, C_hh, H_u_hh, H_r_hh: aggregate household variables
    """
    
    ss = model.ss
    
    try:
        # a. Solve household problem in steady state (iterate until policies converge)
        solve_hh_ss(model, do_print=do_print)

        # b. OLG-consistent steady-state aggregation: forward-sweep J cohorts
        simulate_hh_ss_olg(model, do_print=do_print)
        
        # Extract aggregates
        A_hh = ss.A_hh
        C_hh = ss.C_hh
        H_u_hh = ss.H_u_hh
        H_r_hh = ss.H_r_hh
        
        return A_hh, C_hh, H_u_hh, H_r_hh
        
    except Exception as e:
        # Always log errors for debugging, even if catch_errors=True
        print(f"\n[household_ss ERROR] {type(e).__name__}: {e}")
        
        if not catch_errors:
            raise
        else:
            return np.nan, np.nan, np.nan, np.nan


def find_ss_indirect_housing(model, do_print=False, free_delta_H=False, delta_H_method='weighted'):
    """
    Indirect steady-state implementation with regional housing clearing.

    """

    par = model.par
    ss = model.ss

    alpha = par.alpha
    mu_1 = par.mu_1
    mu_2 = par.mu_2

    # a. Set prices ex ante
    ss.r = float(getattr(par, 'r_ss_target', getattr(ss, 'r', getattr(par, 'r', 0.02))))
    ss.w = float(getattr(par, 'w_ss_target', getattr(ss, 'w', 1.0)))
    ss.q_u = float(getattr(par, 'q_u_ss_target', getattr(ss, 'q_u', getattr(par, 'q_u', 5.0))))
    ss.q_r = float(getattr(par, 'q_r_ss_target', getattr(ss, 'q_r', getattr(par, 'q_r', 1.0))))

    if ss.r <= -0.999:
        raise ValueError(f'Invalid target r={ss.r:.6f}. Must be greater than -1.')
    if ss.w <= 0.0 or ss.q_u <= 0.0 or ss.q_r <= 0.0:
        raise ValueError('Need strictly positive w, q_u, and q_r for indirect housing method.')

    # a2. Steady-state rents from rental-sector FOC (q_{t+1}=q_t at SS)
    #     f^j = q^j * [(1+r) - theta*(1-delta_H)]
    theta = float(getattr(par, 'theta', 0.8))
    ss.f_u = ss.q_u * ((1.0 + ss.r) - (1- theta) * (1.0 - par.delta_H))
    ss.f_r = ss.q_r * ((1.0 + ss.r) - (1- theta) * (1.0 - par.delta_H))

    # a3. Steady-state preference shifter (exogenous shock variable, symmetric
    #     to prices). Pinned to par.kappa unless an explicit SS target is given.
    ss.kappa = float(getattr(par, 'kappa_ss_target', par.kappa))

    # b. Household side
    A_hh, C_hh, H_u_hh, H_r_hh = household_ss(model, do_print=do_print, catch_errors=False)
    ss.A_hh = A_hh
    ss.C_hh = C_hh
    ss.H_u_hh = H_u_hh
    ss.H_r_hh = H_r_hh

    # b. Enforce stock identities from household demand
    ss.K = ss.A_hh - par.NFA_target
    ss.A = ss.A_hh
    ss.H_u = ss.H_u_hh
    ss.H_r = ss.H_r_hh
    ss.L_supply = float(getattr(ss, 'L_supply', getattr(par, 'L_supply', 1.0)))
    ss.L = ss.L_supply

    # c. Implied investment in housing services (IH) from household demand and prices
    ss.IH_u = par.delta_H * ss.H_u
    ss.IH_r = par.delta_H * ss.H_r


    # d. Backing out labor allocations to housing sectors from household demand and prices
    ss.L_u = ss.q_u * mu_2 * ss.IH_u / ss.w
    ss.L_r = ss.q_r * mu_2 * ss.IH_r / ss.w
    ss.L_tilde = ss.L - ss.L_u - ss.L_r

    # e. Back out capital allocations to housing sectors from household demand and prices
    ss.K_u = (ss.IH_u*(ss.L_u**(-mu_2))*(par.X_u**(mu_1+mu_2-1.0)))**(1.0/mu_1)
    ss.K_r = (ss.IH_r*(ss.L_r**(-mu_2))*(par.X_r**(mu_1+mu_2-1.0)))**(1.0/mu_1)
    ss.K_tilde = ss.K - ss.K_u - ss.K_r

    # f. Implied technology level, capital return, and depreciation rate from production side
    ss.Gamma = ss.w / ((1.0 - alpha) * (ss.K_tilde / ss.L_tilde) ** alpha)
    par.Gamma_ss = ss.Gamma
    ss.rK = alpha*ss.Gamma*(ss.K_tilde**(alpha-1))*(ss.L_tilde**(1-alpha))
    par.delta = ss.rK - ss.r

    # Guard: delta < 0 means rK < r, which implies K/Y > alpha/r (capital stock too large).
    # Root cause is almost always beta*(1+r) >= 1, making household savings explosive.
    # Fix: lower beta so that beta*(1+r) < 1, or lower r_ss_target.
    if par.delta < 0:
        import warnings
        beta_check = float(getattr(par, 'beta', float('nan')))
        r_check    = float(ss.r)
        warnings.warn(
            f"find_ss_indirect_housing: par.delta = {par.delta:.4f} < 0  "
            f"(rK={ss.rK:.4f}, r={r_check:.4f}).  "
            f"K/Y = {ss.K/ss.Y:.2f} but alpha/r = {par.alpha/r_check:.2f}.  "
            f"beta*(1+r) = {beta_check*(1+r_check):.4f} (must be < 1 for finite savings).  "
            "Lower beta or r_ss_target.",
            stacklevel=2,
        )

    # g. Production and investment in consumption goods from production side
    ss.Y = ss.Gamma * (ss.K_tilde ** alpha) * (ss.L_tilde ** (1.0 - alpha))
    ss.I = par.delta * ss.K

    # h. Residual sector allocations (construction + final goods = totals)
    ss.K_tilde = ss.K - ss.K_u - ss.K_r
    ss.L_tilde = ss.L - ss.L_u - ss.L_r

    # i. Market clearing residuals
    ss.clearing_A = ss.A - ss.A_hh
    ss.clearing_K = ss.K - ss.K_tilde - ss.K_u - ss.K_r
    ss.clearing_L = ss.L - ss.L_tilde - ss.L_u - ss.L_r
    ss.clearing_H_u = ss.H_u - ss.H_u_hh
    ss.clearing_H_r = ss.H_r - ss.H_r_hh
    ss.clearing_Y = ss.Y - ss.I - ss.C_hh

    ss.Gamma = float(ss.w) / ((1.0 - alpha) * (ss.K_tilde / ss.L_tilde) ** alpha) if ss.K_tilde > 0 and ss.L_tilde > 0 else 1.0
    par.Gamma_ss = ss.Gamma
    par.delta    = ss.rK - float(ss.r)

    # Recompute derived aggregates with updated factor allocations
    ss.Y = ss.Gamma * (ss.K_tilde ** alpha) * (ss.L_tilde ** (1.0 - alpha))
    ss.I = par.delta * float(ss.K)

    # Zero-profit holds by construction
    ss.q_u_model     = float(ss.q_u)
    ss.q_r_model     = float(ss.q_r)
    ss.q_u_gap       = 0.0
    ss.q_r_gap       = 0.0

    # Small-open-economy bookkeeping (drives configure_open_economy + market_clearing)
    h_ratio_inv = par.h_u / par.h_r
    ss.H_total    = ss.H_u + h_ratio_inv * ss.H_r
    ss.H_total_hh = ss.H_u_hh + h_ratio_inv * ss.H_r_hh
    ss.r_world    = float(ss.r)
    ss.NFA        = float(ss.A_hh - ss.K)        # ≈ 0 at the calibrated SS
    ss.NFA_to_Y   = ss.NFA / max(abs(ss.Y), 1e-12)
    ss.clearing_H_total = ss.H_total - ss.H_total_hh
    ss.clearing_H_u = ss.H_u - ss.H_u_hh
    ss.clearing_H_r = ss.H_r - ss.H_r_hh
    ss.clearing_resource = (ss.Y + ss.q_u * ss.IH_u + ss.q_r * ss.IH_r) - ss.C_hh - ss.I

    if do_print:
        print('\n' + '=' * 60)
        print('STEADY STATE FOUND (INDIRECT METHOD)')
        print('=' * 60)
        print('Given prices:')
        print(f'  r = {ss.r:.6f}, w = {ss.w:.6f}, q_u = {ss.q_u:.6f}, q_r = {ss.q_r:.6f}')
        print('Implied aggregates:')
        print(f'  K = {ss.K:.6f}, Y = {ss.Y:.6f}, Gamma = {ss.Gamma:.6f}, delta = {par.delta:.6f}')
        print(f'  H_u = {ss.H_u:.6f}, H_r = {ss.H_r:.6f}')
        if free_delta_H:
            print('Implied housing depreciation:')
            print(f'  delta_H_u (from q_u) = {ss.delta_H_u_implied:.6f}')
            print(f'  delta_H_r (from q_r) = {ss.delta_H_r_implied:.6f}')
            print(f'  delta_H (stored in par) = {par.delta_H:.6f}')
        print('Market residuals:')
        print(f'  Asset market (A - A_hh) = {ss.clearing_A:+.8e}')
        print(f'  Goods market (Y - C_hh - I) = {ss.clearing_Y:+.8e}')
        print(f'  Capital allocation (K - K_tilde - K_u - K_r) = {ss.clearing_K:+.8e}')
        print(f'  Labor market (L_supply - L_tilde - L_u - L_r) = {ss.clearing_L:+.8e}')
        print(f'  Urban housing (H_u - H_u_hh) = {ss.clearing_H_u:+.8e}')
        print(f'  Rural housing (H_r - H_r_hh) = {ss.clearing_H_r:+.8e}')
        print('Price consistency (zero-profit holds by construction):')
        print(f'  X_u (calibrated) = {par.X_u:.6f}')
        print(f'  X_r (calibrated) = {par.X_r:.6f}')
        print(f'  q_u gap = {ss.q_u_gap:+.8e}')
        print(f'  q_r gap = {ss.q_r_gap:+.8e}')
        print('=' * 60 + '\n')

    return ss.K



# ==========================================================================
# find_ss_prices: find SS (r, w) given fixed Gamma_ss, delta, X_u, X_r
# ==========================================================================

def find_ss_prices(model, do_print=False, x0=None, method='hybr',
                   tol=1e-6, max_iter=2000, log_space=True,
                   recalibrate_at_solution=False,
                   nfa_y_target=None):
    """
    Find steady-state factor prices (r, w) that clear all markets, holding
    Gamma_ss, delta, X_u, and X_r fixed (calibrated from SS_0 via
    find_ss_indirect_housing).

    At each candidate (r, w) housing prices are NOT independent free
    variables.  Instead they are derived analytically from the construction
    zero-profit conditions:

        q_u = (IH_u * (rK/mu_1)^(mu_1/c) * (w/mu_2)^(mu_2/c) / X_u)^(c/(mu_1+mu_2))
        q_r = similarly

    where IH_u = delta_H * H_u comes from the household solution at the
    current (r, w, q_u, q_r).  The solver iterates this fixed point: the
    HH solution updates IH, zero-profit updates q, until the outer 2-
    equation residual (production FOCs) converges.

    Two residual equations (outer solve):
        (1) alpha * Gamma * (K~/L~)^(alpha-1) = r + delta   [capital return]
        (2) (1-alpha) * Gamma * (K~/L~)^alpha  = w           [wage]

    Parameters
    ----------
    model     : HANCHousingModelClass (with par.Gamma_ss, par.delta,
                par.X_u, par.X_r already set from find_ss_indirect_housing)
    do_print  : bool – print progress
    x0        : array-like of shape (2,) – initial guess [r, w];
                defaults to current ss values.
    method    : 'hybr' (default), 'fsolve', 'least_squares', 'nelder-mead'
    tol       : convergence tolerance
    max_iter  : maximum function evaluations
    log_space : bool – solve in log-space for better conditioning (default True)
    recalibrate_at_solution : bool – if True, call find_ss_indirect_housing at
                the solved prices (this recalibrates Gamma_ss, delta, X_u, X_r).
                Default False keeps structural parameters fixed.

    Returns
    -------
    success : bool
    x_sol   : np.ndarray of shape (2,) – [r_sol, w_sol]
    """
    from scipy.optimize import fsolve, least_squares, root, minimize

    par = model.par
    ss  = model.ss

    alpha = float(par.alpha)
    mu_1  = float(par.mu_1)
    mu_2  = float(par.mu_2)
    c_    = 1.0 - mu_1 - mu_2

    Gamma = float(getattr(par, 'Gamma_ss', getattr(ss, 'Gamma', 1.0)))
    delta = float(par.delta)
    X_u   = float(par.X_u)
    X_r   = float(par.X_r)

    theta = float(getattr(par, 'theta', 0.8))

    # ── Open-economy branch: r and w are determined analytically ─────────────
    # In the small open economy r = r_world (exogenous) pins rK = r + delta.
    # The K/L ratio follows directly from the firm FOC, and w is determined by
    # r_world and Gamma alone.  The only free variables for the SS are (q_u, q_r)
    if getattr(par, 'open_economy', False):
        # r = r_world fixed; Gamma and delta fixed → rK and w are analytical.
        r_fixed  = float(ss.r)
        rK_fixed = r_fixed + delta
        kl_fixed = (alpha * Gamma / max(rK_fixed, 1e-12)) ** (1.0 / (1.0 - alpha))
        w_fixed  = (1.0 - alpha) * Gamma * kl_fixed ** alpha

        # Analytical inverse: given housing stocks and fixed factor prices,
        def _zp_prices(H_u_, H_r_):
            base = (rK_fixed / mu_1) ** (mu_1 / c_) * (w_fixed / mu_2) ** (mu_2 / c_)
            qu = (max(float(par.delta_H) * H_u_, 1e-12) * base / X_u) ** (c_ / (mu_1 + mu_2))
            qr = (max(float(par.delta_H) * H_r_, 1e-12) * base / X_r) ** (c_ / (mu_1 + mu_2))
            return float(qu), float(qr)

        # ── Initial guess ────────────────────────────────────────────────────
        if x0 is None:
            H_u_ref = max(float(getattr(ss, 'H_u', 0.0)),
                          float(getattr(ss, 'H_u_hh', 0.0)),
                          1e-2)
            H_r_ref = max(float(getattr(ss, 'H_r', 0.0)),
                          float(getattr(ss, 'H_r_hh', 0.0)),
                          1e-2)
            q_u0, q_r0 = _zp_prices(H_u_ref, H_r_ref)
        else:
            x0 = np.array(x0, dtype=float)
            q_u0 = float(x0[2]) if x0.size >= 3 else float(ss.q_u)
            q_r0 = float(x0[3]) if x0.size >= 4 else float(ss.q_r)

        if log_space:
            x0_solve = np.array([np.log(max(q_u0, 1e-8)), np.log(max(q_r0, 1e-8))])
        else:
            x0_solve = np.array([q_u0, q_r0])

        call_count = [0]

        def _residuals(x_in):
            call_count[0] += 1
            if log_space:
                q_u_ = float(np.exp(x_in[0]))
                q_r_ = float(np.exp(x_in[1]))
            else:
                q_u_ = float(x_in[0])
                q_r_ = float(x_in[1])

            if not np.all(np.isfinite([q_u_, q_r_])) or q_u_ <= 0 or q_r_ <= 0:
                return np.full(2, 1e6)

            ss.r = r_fixed;  ss.w = w_fixed
            ss.q_u = q_u_;   ss.q_r = q_r_
            ss.f_u = q_u_ * ((1.0 + r_fixed) - (1.0 - theta) * (1.0 - par.delta_H))
            ss.f_r = q_r_ * ((1.0 + r_fixed) - (1.0 - theta) * (1.0 - par.delta_H))

            try:
                A_hh, _, H_u_hh, H_r_hh = household_ss(model, do_print=False, catch_errors=True)
            except Exception:
                return np.full(2, 1e6)

            if not np.all(np.isfinite([A_hh, H_u_hh, H_r_hh])):
                return np.full(2, 1e6)

            q_u_zp, q_r_zp = _zp_prices(max(float(H_u_hh), 1e-12),
                                          max(float(H_r_hh), 1e-12))

            res = np.array([
                (q_u_zp - q_u_) / max(abs(q_u_), 1e-8),
                (q_r_zp - q_r_) / max(abs(q_r_), 1e-8),
            ])
            if not np.all(np.isfinite(res)):
                return np.full(2, 1e6)

            if do_print:
                print(f'  call={call_count[0]:4d}  r={r_fixed:.5f}  w={w_fixed:.4f}'
                      f'  q_u={q_u_:.4f}  q_r={q_r_:.4f}'
                      f'  res_q_u={res[0]:+.3e}  res_q_r={res[1]:+.3e}'
                      f'  |res|={np.max(np.abs(res)):.3e}')
            return res

        # ── Run solver ───────────────────────────────────────────────────────
        if method == 'fsolve':
            x_sol_raw, _, ier, _ = fsolve(_residuals, x0_solve, full_output=True,
                                           xtol=tol, maxfev=max_iter)
            max_res = float(np.max(np.abs(_residuals(x_sol_raw))))
            success = (ier == 1) and (max_res < 1e-4)
        elif method == 'hybr':
            result    = root(_residuals, x0_solve, method='hybr',
                             tol=tol, options={'maxfev': max_iter})
            x_sol_raw = result.x
            max_res   = float(np.max(np.abs(result.fun)))
            success   = result.success and (max_res < 5.7e-3)
        elif method == 'least_squares':
            result = least_squares(_residuals, x0_solve, method='trf',
                                   xtol=tol, ftol=tol, gtol=tol, max_nfev=max_iter)
            x_sol_raw = result.x
            max_res   = float(np.max(np.abs(result.fun)))
            success   = result.success and (max_res < 1e-4)
        else:
            def _obj(x): return float(np.sum(_residuals(x) ** 2))
            result    = minimize(_obj, x0_solve, method='Nelder-Mead',
                                 options={'xatol': tol, 'fatol': tol**2,
                                          'maxfev': max_iter, 'adaptive': True})
            x_sol_raw = result.x
            max_res   = float(np.sqrt(result.fun))
            success   = result.success and (max_res < 1e-4)

        if log_space:
            q_u_sol = float(np.exp(x_sol_raw[0]))
            q_r_sol = float(np.exp(x_sol_raw[1]))
        else:
            q_u_sol = float(x_sol_raw[0])
            q_r_sol = float(x_sol_raw[1])

        if do_print:
            status = 'CONVERGED' if success else 'NOT CONVERGED'
            print(f'\nfind_ss_prices (open economy): {status}')
            print(f'  r_world fixed={r_fixed:.6f}  w_fixed={w_fixed:.6f}'
                  f'  Gamma fixed={Gamma:.6f}')
            print(f'  Total HH-problem calls : {call_count[0]}')
            print(f'  Max |residual|         : {max_res:.3e}')
            print(f'  q_u={q_u_sol:.6f}  q_r={q_r_sol:.6f}')

        # ── Populate SS ──────────────────────────────────────────────────────
        ss.r = r_fixed;  ss.w = w_fixed
        ss.q_u = q_u_sol;  ss.q_r = q_r_sol
        ss.f_u = q_u_sol * ((1.0 + r_fixed) - (1.0 - theta) * (1.0 - par.delta_H))
        ss.f_r = q_r_sol * ((1.0 + r_fixed) - (1.0 - theta) * (1.0 - par.delta_H))

        if not success:
            return False, np.array([r_fixed, w_fixed])

        A_hh, C_hh, H_u_hh, H_r_hh = household_ss(model, do_print=False, catch_errors=False)
        ss.A_hh = float(A_hh);  ss.C_hh = float(C_hh)
        ss.H_u_hh = float(H_u_hh);  ss.H_r_hh = float(H_r_hh)
        ss.A   = ss.A_hh
        ss.H_u = ss.H_u_hh;  ss.H_r = ss.H_r_hh
        ss.L_supply = float(getattr(ss, 'L_supply', getattr(par, 'L_supply', 1.0)))
        ss.L = ss.L_supply
        ss.IH_u = float(par.delta_H) * ss.H_u
        ss.IH_r = float(par.delta_H) * ss.H_r
        ss.L_u = q_u_sol * mu_2 * ss.IH_u / w_fixed
        ss.L_r = q_r_sol * mu_2 * ss.IH_r / w_fixed
        ss.K_u = (max(ss.IH_u, 1e-12) / (max(ss.L_u, 1e-15) ** mu_2 * X_u ** c_)) ** (1.0 / mu_1)
        ss.K_r = (max(ss.IH_r, 1e-12) / (max(ss.L_r, 1e-15) ** mu_2 * X_r ** c_)) ** (1.0 / mu_1)
        ss.L_tilde = ss.L - ss.L_u - ss.L_r
        # Feasibility guard (analogous to the closed-economy branch's check
        # below).  If construction labour L_u + L_r exceeds total labour
        # supply, L_tilde <= 0 and Y = Gamma*K~^a*L~^(1-a) goes complex/NaN.
        # Fail loudly instead of returning a silent NaN steady state.
        if not np.isfinite(ss.L_tilde) or ss.L_tilde <= 0.0:
            raise RuntimeError(
                'Solved prices imply infeasible production allocations '
                f'(L_tilde={ss.L_tilde:.4e} <= 0): construction labour '
                'L_u + L_r exceeds total labour supply L.')
        ss.K_tilde = ss.L_tilde * kl_fixed
        ss.K   = ss.K_tilde + ss.K_u + ss.K_r
        ss.NFA = ss.A_hh - ss.K
        ss.Gamma = Gamma;  par.Gamma_ss = Gamma;  ss.rK = rK_fixed
        ss.Y = Gamma * (ss.K_tilde ** alpha) * (ss.L_tilde ** (1.0 - alpha)) + ss.q_u * ss.IH_u + ss.q_r * ss.IH_r
        ss.I = delta * ss.K
        ss.clearing_H_u = ss.H_u - ss.H_u_hh
        ss.clearing_H_r = ss.H_r - ss.H_r_hh
        ss.clearing_L   = ss.L - ss.L_tilde - ss.L_u - ss.L_r

        # Resource accounting — consistent with blocks.market_clearing:
        #   clearing_Y        = Y - C_hh - I       (net exports of the Y good)
        #   clearing_resource = Y_tot - C_hh - I   (Y_tot includes housing output)
        # In the SOE neither is zero; they are the trade balance, not a cleared
        # condition.  The genuine steady-state restriction is a zero current
        # account (NFA constant): NX + r*NFA = 0.  Housing investment q*IH sits
        # on both the output and absorption sides and cancels, so the trade
        # balance is NX = Y - C_hh - I = clearing_Y.  current_account below is
        # an *over-identifying* check — find_ss_prices' own residuals (analytic
        # FOCs + zero-profit) do not impose it, so a non-zero value flags an
        # inconsistency in the household-budget aggregation or a loose solve.
        Y_tot = ss.Y + ss.q_u * ss.IH_u + ss.q_r * ss.IH_r
        ss.clearing_Y        = ss.Y - ss.C_hh - ss.I
        ss.clearing_resource = Y_tot - ss.C_hh - ss.I
        ss.current_account   = ss.clearing_Y + ss.r * ss.NFA
        if do_print:
            _ca_rel = ss.current_account / max(abs(Y_tot), 1e-12)
            _flag = '' if abs(_ca_rel) < 1e-3 else '   <-- LARGE: check SS consistency'
            print(f'  SS current account  NX + r*NFA = {ss.current_account:+.4e}'
                  f'  ({_ca_rel:+.2e} of GDP){_flag}')

        h_ratio_inv = par.h_u / par.h_r
        ss.H_total    = ss.H_u + h_ratio_inv * ss.H_r
        ss.H_total_hh = ss.H_u_hh + h_ratio_inv * ss.H_r_hh
        ss.r_world    = r_fixed
        ss.NFA_to_Y   = ss.NFA / max(abs(ss.Y), 1e-12)
        return True, np.array([r_fixed, w_fixed])

    # ── Closed-economy (or open_economy=False) branch: 4-variable solve ──────
    # Solving only 2 outer equations (r, w) with q implicit fails because the
    # ZP-mediated q feedbacks cancel the direct r/w sensitivity, giving a
    # near-flat numerical Jacobian.  Making q explicit gives 4 well-conditioned
    # equations: the 2 production FOCs + 2 zero-profit conditions.
    if x0 is None:
        x0 = np.array([float(ss.r), float(ss.w),
                        float(ss.q_u), float(ss.q_r)])
    else:
        x0 = np.array(x0, dtype=float)
        if x0.size == 2:          # back-compat: old (r, w)-only guess
            x0 = np.append(x0, [float(ss.q_u), float(ss.q_r)])

    rK0 = float(x0[0]) + delta  # rK = r + delta > 0
    if log_space:
        x0_solve = np.array([
            np.log(max(rK0,           1e-8)),
            np.log(max(float(x0[1]), 1e-8)),
            np.log(max(float(x0[2]), 1e-8)),
            np.log(max(float(x0[3]), 1e-8)),
        ])
    else:
        x0_solve = np.array([rK0, float(x0[1]), float(x0[2]), float(x0[3])])

    call_count = [0]

    def _residuals(x_in):
        call_count[0] += 1
        if log_space:
            rK_, w_    = float(np.exp(x_in[0])), float(np.exp(x_in[1]))
            q_u_, q_r_ = float(np.exp(x_in[2])), float(np.exp(x_in[3]))
        else:
            rK_, w_    = float(x_in[0]), float(x_in[1])
            q_u_, q_r_ = float(x_in[2]), float(x_in[3])
        r_ = rK_ - delta

        if (w_ <= 0.0 or rK_ <= 0.0 or q_u_ <= 0.0 or q_r_ <= 0.0
                or not np.all(np.isfinite([rK_, w_, q_u_, q_r_]))):
            return np.full(4, 1e6)

        ss.r   = r_
        ss.w   = w_
        ss.q_u = q_u_
        ss.q_r = q_r_
        ss.f_u = q_u_ * ((1.0 + r_) - (1.0 - theta) * (1.0 - par.delta_H))
        ss.f_r = q_r_ * ((1.0 + r_) - (1.0 - theta) * (1.0 - par.delta_H))

        try:
            A_hh, _, H_u_hh, H_r_hh = household_ss(model, do_print=False,
                                                     catch_errors=True)
        except Exception:
            return np.full(4, 1e6)

        if not np.all(np.isfinite([A_hh, H_u_hh, H_r_hh])):
            return np.full(4, 1e6)

        K   = max(float(A_hh),   1e-12)
        H_u = max(float(H_u_hh), 1e-12)
        H_r = max(float(H_r_hh), 1e-12)
        L   = max(float(ss.L_supply), 1e-12)

        IH_u = float(par.delta_H) * H_u
        IH_r = float(par.delta_H) * H_r

        # Zero-profit conditions (explicit residuals for q_u, q_r)
        q_u_zp = (max(IH_u, 1e-12)
                  * (rK_ / mu_1) ** (mu_1 / c_)
                  * (w_  / mu_2) ** (mu_2 / c_)
                  / X_u) ** (c_ / (mu_1 + mu_2))
        q_r_zp = (max(IH_r, 1e-12)
                  * (rK_ / mu_1) ** (mu_1 / c_)
                  * (w_  / mu_2) ** (mu_2 / c_)
                  / X_r) ** (c_ / (mu_1 + mu_2))

        # Construction factor demands are pinned down by current candidate prices
        # (q_u_, q_r_), not by zero-profit-implied prices.
        L_u = max(q_u_ * mu_2 * IH_u / w_, 1e-15)
        L_r = max(q_r_ * mu_2 * IH_r / w_, 1e-15)
        K_u = (max(IH_u, 1e-12) / (L_u ** mu_2 * X_u ** c_)) ** (1.0 / mu_1)
        K_r = (max(IH_r, 1e-12) / (L_r ** mu_2 * X_r ** c_)) ** (1.0 / mu_1)

        K_tilde = K - K_u - K_r
        L_tilde = L - L_u - L_r

        if K_tilde <= 0.0 or L_tilde <= 0.0:
            return np.full(4, 1e6)

        ratio    = K_tilde / L_tilde
        rK_model = alpha * Gamma * ratio ** (alpha - 1.0)
        w_model  = (1.0 - alpha) * Gamma * ratio ** alpha

        res = np.array([
            (rK_model - rK_) / max(abs(rK_), 1e-8),    # capital FOC
            (w_model  - w_ ) / max(abs(w_),  1e-8),    # labour FOC
            (q_u_zp   - q_u_) / max(abs(q_u_), 1e-8), # urban zero-profit
            (q_r_zp   - q_r_) / max(abs(q_r_), 1e-8), # rural zero-profit
        ])

        if not np.all(np.isfinite(res)):
            return np.full(4, 1e6)

        if do_print:
            print(f'  call={call_count[0]:4d}  r={r_:.5f}  w={w_:.4f}'
                  f'  q_u={q_u_:.4f}  q_r={q_r_:.4f}  |res|={np.max(np.abs(res)):.3e}')

        return res

    # --- run solver --------------------------------------------------------
    if method == 'fsolve':
        x_sol_raw, _, ier, _ = fsolve(_residuals, x0_solve, full_output=True,
                                      xtol=tol, maxfev=max_iter)
        max_res = float(np.max(np.abs(_residuals(x_sol_raw))))
        success = (ier == 1) and (max_res < 1e-4)
    elif method == 'hybr':
        result    = root(_residuals, x0_solve, method='hybr',
                         tol=tol, options={'maxfev': max_iter})
        x_sol_raw = result.x
        max_res   = float(np.max(np.abs(result.fun)))
        success   = result.success and (max_res < 5.7e-03)
    elif method == 'least_squares':
        if log_space:
            lsq_bounds = (-np.inf, np.inf)
        else:
            lb = np.full(4, 1e-8)
            ub = np.full(4, np.inf)
            lsq_bounds = (lb, ub)
        # x_scale=1.0 (not 'jac'): the penalty branch in _residuals returns a
        # constant 1e6, so finite-difference Jacobian columns can be exactly
        # zero in the penalty region.  With x_scale='jac', that triggers
        # 1/0 -> Inf in scipy's scaling and NaNs in the SVD.
        result = least_squares(_residuals, x0_solve, method='trf',
                               bounds=lsq_bounds, x_scale=1.0,
                               xtol=tol, ftol=tol, gtol=tol,
                               max_nfev=max_iter)
        x_sol_raw = result.x
        max_res = float(np.max(np.abs(result.fun)))
        success = result.success and (max_res < 1e-4)
    elif method == 'nelder-mead':
        def _objective(x_in): return float(np.sum(_residuals(x_in) ** 2))
        result    = minimize(_objective, x0_solve, method='Nelder-Mead',
                             options={'xatol': tol, 'fatol': tol**2,
                                      'maxfev': max_iter, 'adaptive': True})
        x_sol_raw = result.x
        max_res   = float(np.sqrt(result.fun))
        success   = result.success and (max_res < 1e-4)
    else:
        raise ValueError(
            f"method must be 'hybr', 'fsolve', 'least_squares', or 'nelder-mead', got '{method}'")

    if log_space:
        x_sol = np.array([
            np.exp(x_sol_raw[0]) - delta,  # rK → r
            np.exp(x_sol_raw[1]),           # w
            np.exp(x_sol_raw[2]),           # q_u
            np.exp(x_sol_raw[3]),           # q_r
        ])
    else:
        x_sol = np.array([
            x_sol_raw[0] - delta,           # rK → r
            x_sol_raw[1],                   # w
            x_sol_raw[2],                   # q_u
            x_sol_raw[3],                   # q_r
        ])

    r_sol, w_sol     = float(x_sol[0]), float(x_sol[1])
    q_u_sol, q_r_sol = float(x_sol[2]), float(x_sol[3])

    # If the nonlinear solver did not converge, do not run the post-solve
    # accounting block. That block assumes a feasible root and can raise a
    # misleading K_tilde/L_tilde RuntimeError when we only have a trial point.
    if not success:
        ss.r = r_sol
        ss.w = w_sol
        ss.q_u = q_u_sol
        ss.q_r = q_r_sol
        if do_print:
            print(f'\nfind_ss_prices: NOT CONVERGED')
            print(f'  Total HH-problem calls : {call_count[0]}')
            print(f'  Max |residual|         : {max_res:.3e}')
            print(f'  r={r_sol:.6f}  w={w_sol:.6f}'
                  f'  q_u={ss.q_u:.6f}  q_r={ss.q_r:.6f}')
        return False, np.array([r_sol, w_sol])

    # Populate SS objects at solved prices.
    if recalibrate_at_solution:
        ss.r   = r_sol
        ss.w   = w_sol
        ss.q_u = q_u_sol
        ss.q_r = q_r_sol
        _saved = {k: getattr(par, k, None)
                  for k in ('r_ss_target', 'w_ss_target',
                            'q_u_ss_target', 'q_r_ss_target')}
        par.r_ss_target   = r_sol
        par.w_ss_target   = w_sol
        par.q_u_ss_target = q_u_sol
        par.q_r_ss_target = q_r_sol
        find_ss_indirect_housing(model, do_print=do_print)
        for k, v in _saved.items():
            if v is not None:
                setattr(par, k, v)
            elif hasattr(par, k):
                delattr(par, k)
    else:
        ss.r   = r_sol
        ss.w   = w_sol
        ss.q_u = q_u_sol
        ss.q_r = q_r_sol
        ss.f_u = q_u_sol * ((1.0 + r_sol) - (1.0 - theta) * (1.0 - par.delta_H))
        ss.f_r = q_r_sol * ((1.0 + r_sol) - (1.0 - theta) * (1.0 - par.delta_H))

        A_hh, C_hh, H_u_hh, H_r_hh = household_ss(model, do_print=False, catch_errors=False)
        ss.A_hh = float(A_hh)
        ss.C_hh = float(C_hh)
        ss.H_u_hh = float(H_u_hh)
        ss.H_r_hh = float(H_r_hh)

        ss.K = ss.A_hh
        ss.A = ss.A_hh
        ss.H_u = ss.H_u_hh
        ss.H_r = ss.H_r_hh
        ss.L_supply = float(getattr(ss, 'L_supply', getattr(par, 'L_supply', 1.0)))
        ss.L = ss.L_supply

        ss.IH_u = float(par.delta_H) * ss.H_u
        ss.IH_r = float(par.delta_H) * ss.H_r

        ss.L_u = ss.q_u * mu_2 * ss.IH_u / ss.w
        ss.L_r = ss.q_r * mu_2 * ss.IH_r / ss.w
        ss.K_u = (max(ss.IH_u, 1e-12) / (max(ss.L_u, 1e-15) ** mu_2 * X_u ** c_)) ** (1.0 / mu_1)
        ss.K_r = (max(ss.IH_r, 1e-12) / (max(ss.L_r, 1e-15) ** mu_2 * X_r ** c_)) ** (1.0 / mu_1)
        ss.K_tilde = ss.K - ss.K_u - ss.K_r
        ss.L_tilde = ss.L - ss.L_u - ss.L_r

        if ss.K_tilde <= 0.0 or ss.L_tilde <= 0.0:
            raise RuntimeError('Solved prices imply infeasible production allocations (K_tilde<=0 or L_tilde<=0).')

        ss.Gamma = Gamma
        ss.rK = r_sol + delta
        ss.Y = ss.Gamma * (ss.K_tilde ** alpha) * (ss.L_tilde ** (1.0 - alpha))
        ss.I = delta * ss.K

        ss.clearing_A = ss.A - ss.A_hh
        ss.clearing_K = ss.K - ss.K_tilde - ss.K_u - ss.K_r
        ss.clearing_L = ss.L - ss.L_tilde - ss.L_u - ss.L_r
        ss.clearing_H_u = ss.H_u - ss.H_u_hh
        ss.clearing_H_r = ss.H_r - ss.H_r_hh
        ss.clearing_Y = ss.Y - ss.I - ss.C_hh

        ss.q_u_model = (max(ss.IH_u, 1e-12)
                        * (ss.rK / mu_1) ** (mu_1 / c_)
                        * (ss.w  / mu_2) ** (mu_2 / c_)
                        / X_u) ** (c_ / (mu_1 + mu_2))
        ss.q_r_model = (max(ss.IH_r, 1e-12)
                        * (ss.rK / mu_1) ** (mu_1 / c_)
                        * (ss.w  / mu_2) ** (mu_2 / c_)
                        / X_r) ** (c_ / (mu_1 + mu_2))
        ss.q_u_gap = ss.q_u_model - ss.q_u
        ss.q_r_gap = ss.q_r_model - ss.q_r

        # Small-open-economy bookkeeping
        h_ratio_inv = par.h_u / par.h_r
        ss.H_total    = ss.H_u + h_ratio_inv * ss.H_r
        ss.H_total_hh = ss.H_u_hh + h_ratio_inv * ss.H_r_hh
        ss.r_world    = float(ss.r)
        ss.NFA        = float(ss.A_hh - ss.K)
        ss.NFA_to_Y   = ss.NFA / max(abs(ss.Y), 1e-12)
        ss.clearing_H_total = ss.H_total - ss.H_total_hh
        ss.clearing_H_r = ss.H_r - ss.H_r_hh
        ss.clearing_resource = (ss.Y + ss.q_u * ss.IH_u + ss.q_r * ss.IH_r) - ss.C_hh - ss.I

    if do_print:
        print(f'\nfind_ss_prices: {"CONVERGED" if success else "NOT CONVERGED"}')
        print(f'  Total HH-problem calls : {call_count[0]}')
        print(f'  Max |residual|         : {max_res:.3e}')
        print(f'  r={r_sol:.6f}  w={w_sol:.6f}'
              f'  q_u={ss.q_u:.6f}  q_r={ss.q_r:.6f}')

    return success, np.array([r_sol, w_sol])


# ==========================================
# Comparative-statics machinery for the supply/demand diagrams
# ==========================================

def comparative_statics_tenure_supply_demand(
    model_baseline,
    shock='kappa',
    shock_size=0.05,
    n_points=15,
    rel_price_span=(2.0, 0.5),
    do_print=False,
):
    """Compute the data needed for the 2x2 tenure supply/demand diagrams.

    Splits the *expensive* HH solve loop out of `plots.plot_comparative_
    statics_tenure_supply_demand` so the data dict can be precomputed once
    and re-plotted many times.

    Parameters
    ----------
    model_baseline : HANCHousingModelClass
        Pre-solved baseline model.  A deep copy is made internally so the
        caller's model is not mutated.
    shock : {'kappa', 'r_world', 'hh_scale', 'cohort_16_36'}
        Which structural input to perturb.  'cohort_16_36' (alias
        'young_cohorts') raises the population share of the 16–36 cohorts by
        `shock_size`, holding total population fixed (a pure composition
        change).
    shock_size : float
        Relative shock magnitude (0.05 = +5%).  Applied as
            new = ss + shock_size · |ss|         for kappa / r_world / hh_scale
            ω   = ω_ss + shock_size · ω_ss        on the 16–36 band for
                                                  cohort_16_36 (then renormalised)
        The |·| keeps a positive `shock_size` numerically increasing in all
        three scalar cases — important for kappa, which is negative at SS so
        a multiplicative rule would have flipped the sign meaning.
    n_points : int
        Number of grid points along each price-ratio axis (rounded up to
        odd for symmetry around the baseline).  Default 15 — enough for
        smooth curves over a wide span without making the HH-solve loop
        too slow.
    rel_price_span : float or (urban, rural) tuple
        Half-width of the log-linear grid around the baseline ratio.
        Scalar form applies the same span to both panels:
            span=2.0  →  baseline · [1/3, 3]
            span=1.0  →  baseline · [1/2, 2]
            span=0.5  →  baseline · [2/3, 3/2]
            span=0.05 →  baseline · [0.95, 1.05]   (tiny zoom)
        Tuple form `(urban_span, rural_span)` lets you set per-panel
        spans — useful because rural supply scales cubically in q_r and
        large X_r makes it MUCH more elastic than urban; a narrower
        rural span keeps the rural supply curve in the same x range
        as the rural demand schedules.  Default `(2.0, 0.5)` reproduces
        a textbook urban supply/demand and keeps the rural panel readable.
        Decoupled from `shock_size`: the shock magnitude governs the
        baseline-vs-shock comparison, the span governs the diagnostic
        price range you want the curves visible over.
    do_print : bool
        Forward to internal `solve_hh_ss` / `simulate_hh_ss_olg` calls.

    Returns
    -------
    dict
        Flat dict holding all arrays plot_comparative_statics_tenure_supply_
        demand needs:
            metadata             : 'shock', 'shock_size', 'shock_desc',
                                   'n_points'
            SS scalars           : 'q_u0', 'q_r0', 'f_u0', 'f_r0'
            grids                : 'ratio_q_grid', 'ratio_f_grid'
            q-grid demand        : 'q_d_{u|r}_{owner|rent|total}_{base|shock}'
            f-grid demand        : 'f_d_{u|r}_{owner|rent|total}_{base|shock}'
            q-grid supply        : 'q_s_{u|r}_{base|shock}'
            f-grid supply        : 'f_s_{u|r}_{base|shock}'
    """
    import copy

    # --- Sanitise inputs ------------------------------------------------------
    if not isinstance(shock, str):
        shock = 'kappa'
    shock = shock.strip().lower()

    if not np.isscalar(shock_size):
        shock_size = 0.05
    shock_size = float(shock_size)

    if not np.isscalar(n_points):
        n_points = 15
    n_points = int(max(3, n_points))
    if n_points % 2 == 0:
        n_points += 1

    # rel_price_span — half-width of the log-linear grid around the baseline
    # ratio.  Decoupled from shock_size.  Accepts either:
    #   - scalar  : same span for urban and rural panels.
    #   - 2-tuple (urban_span, rural_span) : per-panel spans, useful because
    #     rural supply is much more elastic in q_r than urban supply is in
    #     q_u (rural land X_r >> urban land X_u, so IH_r ∝ q_r^pow scales
    #     ~10× faster per unit price change).  A smaller rural span keeps
    #     the supply curve in the same x range as the demand schedules.
    if rel_price_span is None:
        rel_price_span = (2.0, 0.5)
    if isinstance(rel_price_span, (tuple, list)) and len(rel_price_span) == 2:
        rel_price_span_u = float(max(0.0, rel_price_span[0]))
        rel_price_span_r = float(max(0.0, rel_price_span[1]))
    elif np.isscalar(rel_price_span):
        rel_price_span_u = float(max(0.0, rel_price_span))
        rel_price_span_r = rel_price_span_u
    else:
        rel_price_span_u = 2.0
        rel_price_span_r = 0.5
    # Keep `rel_price_span` as the *urban* value for backward compatibility
    # with any downstream code that reads it.
    rel_price_span = rel_price_span_u

    # --- Local helpers --------------------------------------------------------
    def _tenure_quantities_from_ss(ss):
        # NOTE: H_*_hh already carry the population scaling `hh_scale`
        # (simulate_hh_ss_olg sets ss.H_*_hh = hh_scale · totals), so we must
        # NOT multiply by hh_scale again here — doing so double-counts the
        # population size and inflates a +α% cohort/hh_scale shock to +α%²-ish.
        H_u_owner = float(getattr(ss, 'H_u_owner_hh', 0.0))
        H_r_owner = float(getattr(ss, 'H_r_owner_hh', 0.0))
        H_u_total = float(getattr(ss, 'H_u_hh', 0.0))
        H_r_total = float(getattr(ss, 'H_r_hh', 0.0))
        return {
            'urban_owner':  H_u_owner,
            'urban_rental': max(0.0, H_u_total - H_u_owner),
            'rural_owner':  H_r_owner,
            'rural_rental': max(0.0, H_r_total - H_r_owner),
        }

    def _ensure_distribution_solved(m):
        ss_ = m.ss
        need = (
            not hasattr(ss_, 'D') or ss_.D is None
            or not hasattr(ss_, 'H_u_hh')
            or not hasattr(ss_, 'H_r_hh')
            or not hasattr(ss_, 'H_u_owner_hh')
            or not hasattr(ss_, 'H_r_owner_hh')
        )
        if need:
            solve_hh_ss(m, do_print=do_print)
            simulate_hh_ss_olg(m, do_print=do_print)

    def _solve_hh_demand_at_prices(m, q_u=None, q_r=None, f_u=None, f_r=None):
        ss_ = m.ss
        par_ = m.par
        if q_u is not None:
            ss_.q_u = float(q_u)
        if q_r is not None:
            ss_.q_r = float(q_r)
        if f_u is None:
            ss_.f_u = float(ss_.q_u * ((1.0 + ss_.r) - (1.0 - par_.theta) * (1.0 - par_.delta_H)))
        else:
            ss_.f_u = float(f_u)
        if f_r is None:
            ss_.f_r = float(ss_.q_r * ((1.0 + ss_.r) - (1.0 - par_.theta) * (1.0 - par_.delta_H)))
        else:
            ss_.f_r = float(f_r)
        solve_hh_ss(m, do_print=do_print)
        simulate_hh_ss_olg(m, do_print=do_print)
        return _tenure_quantities_from_ss(ss_)

    def _supply_from_prices(m, q_u=None, q_r=None, f_u=None, f_r=None):
        par_ = m.par
        ss_ = m.ss
        delta_H = float(getattr(par_, 'delta_H', 0.0))
        mu_1 = float(getattr(par_, 'mu_1', 0.0))
        mu_2 = float(getattr(par_, 'mu_2', 0.0))
        c_ = 1.0 - mu_1 - mu_2
        if delta_H <= 0.0 or mu_1 <= 0.0 or mu_2 <= 0.0 or c_ <= 0.0:
            return {'urban_total': np.nan, 'rural_total': np.nan}
        theta_ = float(getattr(par_, 'theta', 0.8))
        xi = (1.0 + float(ss_.r)) - (1.0 - theta_) * (1.0 - delta_H)
        if q_u is None and f_u is not None:
            q_u = float(f_u) / max(xi, 1e-12)
        if q_r is None and f_r is not None:
            q_r = float(f_r) / max(xi, 1e-12)
        if q_u is None:
            q_u = float(ss_.q_u)
        if q_r is None:
            q_r = float(ss_.q_r)
        q_u = float(max(q_u, 1e-12))
        q_r = float(max(q_r, 1e-12))
        w_ = float(max(ss_.w, 1e-12))
        rK = float(max(float(ss_.r) + float(getattr(par_, 'delta', 0.0)), 1e-12))
        X_u = float(max(getattr(par_, 'X_u', 1.0), 1e-12))
        X_r = float(max(getattr(par_, 'X_r', 1.0), 1e-12))
        coef_u = ((rK / mu_1) ** (mu_1 / c_)) * ((w_ / mu_2) ** (mu_2 / c_)) / X_u
        coef_r = ((rK / mu_1) ** (mu_1 / c_)) * ((w_ / mu_2) ** (mu_2 / c_)) / X_r
        pow_ = (mu_1 + mu_2) / c_
        IH_u = (q_u ** pow_) / max(coef_u, 1e-12)
        IH_r = (q_r ** pow_) / max(coef_r, 1e-12)
        return {
            'urban_total': float(max(IH_u / delta_H, 0.0)),
            'rural_total': float(max(IH_r / delta_H, 0.0)),
        }

    def _apply_selected_shock(m):
        par_ = m.par
        ss_ = m.ss
        s_pct = 100.0 * shock_size

        if shock == 'kappa':
            old = float(getattr(par_, 'kappa', getattr(ss_, 'kappa', 0.0)))
            new = old + shock_size * abs(old)   # sign-preserving direction
            par_.kappa = new
            ss_.kappa = new
            return f'kappa: {old:.4f} → {new:.4f}  ({s_pct:+.2f}%)'

        if shock in ('r_world', 'r'):
            old = float(getattr(ss_, 'r_world',
                                getattr(ss_, 'r',
                                        getattr(par_, 'r_world',
                                                getattr(par_, 'r', 0.0)))))
            new = old + shock_size * abs(old)   # sign-preserving direction
            ss_.r_world = new
            ss_.r = new
            par_.r_world = new
            if hasattr(par_, 'r'):
                par_.r = new
            if hasattr(par_, 'r_ss_target'):
                par_.r_ss_target = new
            return f'r_world: {old:.4f} → {new:.4f}  ({s_pct:+.2f}%)'

        if shock in ('hh_scale', 'household_scale'):
            old = float(getattr(ss_, 'hh_scale', getattr(par_, 'hh_scale', 1.0)))
            new = old + shock_size * abs(old)   # sign-preserving direction
            if new <= 0.0:
                raise ValueError('hh_scale shock produced non-positive level.')
            ss_.hh_scale = new
            par_.hh_scale = new
            return f'hh_scale: {old:.4f} → {new:.4f}  ({s_pct:+.2f}%)'

        if shock in ('young_cohorts', 'cohort_16_36'):
            # Age-targeted COMPOSITION shock: raise the weight on the 16–36
            # cohorts by `shock_size`, holding TOTAL population fixed.  The
            # cohort weights are normalised to sum=1 downstream by
            # `_get_ss_cohort_weights`, so storing the bumped vector makes the
            # young band's share rise while every other cohort's share falls
            # proportionally; hh_scale (∝ total population) is left untouched.
            J = int(getattr(par_, 'J', 0))
            if J <= 0:
                raise ValueError('Need par.J > 0 for cohort_16_36 shock.')
            if hasattr(ss_, 'cohort_weights'):
                omega_ss = np.asarray(ss_.cohort_weights, dtype=float).ravel()
            elif hasattr(par_, 'cohort_weights_ss'):
                omega_ss = np.asarray(par_.cohort_weights_ss, dtype=float).ravel()
            else:
                omega_ss = np.ones(J) / J
            if omega_ss.size != J:
                _fix = np.zeros(J)
                _n = min(J, omega_ss.size)
                _fix[:_n] = omega_ss[:_n]
                omega_ss = _fix

            # Map ages 16–36 to cohort indices (age j ↔ index j − age_min).
            age_min = int(getattr(par_, 'age_min', 16))
            lo_age, hi_age = 16, 36
            i_lo = max(0, lo_age - age_min)
            i_hi = min(J, hi_age - age_min + 1)
            if i_hi <= i_lo:
                raise ValueError(
                    f'cohort_16_36 shock: age band [{lo_age},{hi_age}] does '
                    f'not intersect the cohort grid (age_min={age_min}, J={J}).'
                )
            young = slice(i_lo, i_hi)
            dev = np.zeros(J)
            dev[young] = shock_size * omega_ss[young]
            new_omega = np.maximum(omega_ss + dev, 0.0)

            ss_omega_mass = float(np.sum(omega_ss))
            total_mass = float(np.sum(new_omega))
            if total_mass > 0.0 and ss_omega_mass > 0.0:
                # store at baseline mass; downstream normalisation then shifts
                # only the SHAPE (composition), not the population level
                ss_.cohort_weights = new_omega * (ss_omega_mass / total_mass)
            else:
                ss_.cohort_weights = omega_ss.copy()
            par_.cohort_weights_ss = ss_.cohort_weights.copy()

            base_share = (float(np.sum(omega_ss[young])) / ss_omega_mass
                          if ss_omega_mass > 0.0 else float('nan'))
            new_share = (float(np.sum(new_omega[young])) / total_mass
                         if total_mass > 0.0 else float('nan'))
            return (f'cohort_16_36: ages {lo_age}-{hi_age} (idx {i_lo}:{i_hi}) '
                    f'{s_pct:+.2f}%; population FIXED (composition only).  '
                    f'young share {base_share:.4f}→{new_share:.4f}')

        raise ValueError(
            "Unknown shock. Use one of: "
            "'kappa', 'r_world', 'hh_scale', 'cohort_16_36'."
        )

    # --- Baseline SS at the unshocked prices ----------------------------------
    base = model_baseline
    _ensure_distribution_solved(base)

    q_u0 = float(base.ss.q_u)
    q_r0 = float(base.ss.q_r)
    f_u0 = float(getattr(base.ss, 'f_u',
                          q_u0 * ((1.0 + base.ss.r)
                                  - (1.0 - base.par.theta) * (1.0 - base.par.delta_H))))
    f_r0 = float(getattr(base.ss, 'f_r',
                          q_r0 * ((1.0 + base.ss.r)
                                  - (1.0 - base.par.theta) * (1.0 - base.par.delta_H))))

    ratio_q0 = q_u0 / max(q_r0, 1e-12)
    ratio_f0 = f_u0 / max(f_r0, 1e-12)

    # Urban panel grid (varies the urban price, baseline ratio q_u0/q_r0).
    log_span_u = np.log(max(1e-6, 1.0 + rel_price_span_u))
    ratio_q_grid = ratio_q0 * np.exp(np.linspace(-log_span_u, log_span_u, n_points))
    ratio_f_grid = ratio_f0 * np.exp(np.linspace(-log_span_u, log_span_u, n_points))

    # Rural panel grid (varies the rural price, baseline ratio q_r0/q_u0).
    # Built independently with its own (typically narrower) span so the
    # rural supply curve stays in the demand x-range.
    inv_ratio_q0 = q_r0 / max(q_u0, 1e-12)
    inv_ratio_f0 = f_r0 / max(f_u0, 1e-12)
    log_span_r = np.log(max(1e-6, 1.0 + rel_price_span_r))
    ratio_q_grid_r_native = inv_ratio_q0 * np.exp(
        np.linspace(-log_span_r, log_span_r, n_points))
    ratio_f_grid_r_native = inv_ratio_f0 * np.exp(
        np.linspace(-log_span_r, log_span_r, n_points))

    # --- Copy + apply the shock to the *shocked* model ------------------------
    model_base = copy.deepcopy(base)
    model_shock = copy.deepcopy(base)
    shock_desc = _apply_selected_shock(model_shock)

    # Build per-panel price ratio grids.  The URBAN panel varies the urban
    # price (q_u or f_u) and holds the rural price fixed; the RURAL panel
    # varies the rural price (q_r or f_r) and holds the urban price fixed.
    # Without this split the rural panel's supply curve would be vertical:
    # `_supply_from_prices` returns IH_r = (q_r/coef_r)^pow / X_r, which only
    # depends on q_r — so sweeping q_u while holding q_r leaves rural supply
    # constant across the grid.
    ratio_q_grid_u = ratio_q_grid              # urban panel grid (varies q_u, span = rel_price_span_u)
    ratio_q_grid_r = ratio_q_grid_r_native     # rural panel grid (varies q_r, span = rel_price_span_r)
    ratio_f_grid_u = ratio_f_grid              # urban rent panel grid (varies f_u)
    ratio_f_grid_r = ratio_f_grid_r_native     # rural rent panel grid (varies f_r)

    # --- Sweep 1: q_u varying, q_r fixed → URBAN q-panel ---------------------
    q_d_u_owner_base  = np.zeros(n_points); q_d_u_owner_shock  = np.zeros(n_points)
    q_d_u_rent_base   = np.zeros(n_points); q_d_u_rent_shock   = np.zeros(n_points)
    q_s_u_base        = np.zeros(n_points); q_s_u_shock        = np.zeros(n_points)
    for i, ratio_q in enumerate(ratio_q_grid_u):
        q_u_i = float(ratio_q) * q_r0
        d_b = _solve_hh_demand_at_prices(model_base,  q_u=q_u_i, q_r=q_r0, f_u=f_u0, f_r=f_r0)
        d_s = _solve_hh_demand_at_prices(model_shock, q_u=q_u_i, q_r=q_r0, f_u=f_u0, f_r=f_r0)
        s_b = _supply_from_prices(model_base,         q_u=q_u_i, q_r=q_r0)
        s_s = _supply_from_prices(model_shock,        q_u=q_u_i, q_r=q_r0)
        q_d_u_owner_base[i]  = d_b['urban_owner']
        q_d_u_rent_base[i]   = d_b['urban_rental']
        q_d_u_owner_shock[i] = d_s['urban_owner']
        q_d_u_rent_shock[i]  = d_s['urban_rental']
        q_s_u_base[i]  = s_b['urban_total']
        q_s_u_shock[i] = s_s['urban_total']

    # --- Sweep 2: q_r varying, q_u fixed → RURAL q-panel ---------------------
    # Grid is ratio_q^r/q^u = q_r_i / q_u0, sorted ascending.
    q_d_r_owner_base  = np.zeros(n_points); q_d_r_owner_shock  = np.zeros(n_points)
    q_d_r_rent_base   = np.zeros(n_points); q_d_r_rent_shock   = np.zeros(n_points)
    q_s_r_base        = np.zeros(n_points); q_s_r_shock        = np.zeros(n_points)
    for i, ratio_q in enumerate(ratio_q_grid_r):
        q_r_i = float(ratio_q) * q_u0
        d_b = _solve_hh_demand_at_prices(model_base,  q_u=q_u0, q_r=q_r_i, f_u=f_u0, f_r=f_r0)
        d_s = _solve_hh_demand_at_prices(model_shock, q_u=q_u0, q_r=q_r_i, f_u=f_u0, f_r=f_r0)
        s_b = _supply_from_prices(model_base,         q_u=q_u0, q_r=q_r_i)
        s_s = _supply_from_prices(model_shock,        q_u=q_u0, q_r=q_r_i)
        q_d_r_owner_base[i]  = d_b['rural_owner']
        q_d_r_rent_base[i]   = d_b['rural_rental']
        q_d_r_owner_shock[i] = d_s['rural_owner']
        q_d_r_rent_shock[i]  = d_s['rural_rental']
        q_s_r_base[i]  = s_b['rural_total']
        q_s_r_shock[i] = s_s['rural_total']

    # --- Sweep 3: f_u varying, f_r fixed → URBAN rent-panel ------------------
    f_d_u_owner_base  = np.zeros(n_points); f_d_u_owner_shock  = np.zeros(n_points)
    f_d_u_rent_base   = np.zeros(n_points); f_d_u_rent_shock   = np.zeros(n_points)
    f_s_u_base        = np.zeros(n_points); f_s_u_shock        = np.zeros(n_points)
    for i, ratio_f in enumerate(ratio_f_grid_u):
        f_u_i = float(ratio_f) * f_r0
        d_b = _solve_hh_demand_at_prices(model_base,  q_u=q_u0, q_r=q_r0, f_u=f_u_i, f_r=f_r0)
        d_s = _solve_hh_demand_at_prices(model_shock, q_u=q_u0, q_r=q_r0, f_u=f_u_i, f_r=f_r0)
        s_b = _supply_from_prices(model_base,         f_u=f_u_i, f_r=f_r0)
        s_s = _supply_from_prices(model_shock,        f_u=f_u_i, f_r=f_r0)
        f_d_u_owner_base[i]  = d_b['urban_owner']
        f_d_u_rent_base[i]   = d_b['urban_rental']
        f_d_u_owner_shock[i] = d_s['urban_owner']
        f_d_u_rent_shock[i]  = d_s['urban_rental']
        f_s_u_base[i]  = s_b['urban_total']
        f_s_u_shock[i] = s_s['urban_total']

    # --- Sweep 4: f_r varying, f_u fixed → RURAL rent-panel ------------------
    f_d_r_owner_base  = np.zeros(n_points); f_d_r_owner_shock  = np.zeros(n_points)
    f_d_r_rent_base   = np.zeros(n_points); f_d_r_rent_shock   = np.zeros(n_points)
    f_s_r_base        = np.zeros(n_points); f_s_r_shock        = np.zeros(n_points)
    for i, ratio_f in enumerate(ratio_f_grid_r):
        f_r_i = float(ratio_f) * f_u0
        d_b = _solve_hh_demand_at_prices(model_base,  q_u=q_u0, q_r=q_r0, f_u=f_u0, f_r=f_r_i)
        d_s = _solve_hh_demand_at_prices(model_shock, q_u=q_u0, q_r=q_r0, f_u=f_u0, f_r=f_r_i)
        s_b = _supply_from_prices(model_base,         f_u=f_u0, f_r=f_r_i)
        s_s = _supply_from_prices(model_shock,        f_u=f_u0, f_r=f_r_i)
        f_d_r_owner_base[i]  = d_b['rural_owner']
        f_d_r_rent_base[i]   = d_b['rural_rental']
        f_d_r_owner_shock[i] = d_s['rural_owner']
        f_d_r_rent_shock[i]  = d_s['rural_rental']
        f_s_r_base[i]  = s_b['rural_total']
        f_s_r_shock[i] = s_s['rural_total']

    # --- Pack everything into a flat data dict --------------------------------
    return {
        # Metadata
        'shock':      shock,
        'shock_size': shock_size,
        'shock_desc': shock_desc,
        'n_points':   n_points,
        # SS scalars
        'q_u0': q_u0, 'q_r0': q_r0, 'f_u0': f_u0, 'f_r0': f_r0,
        # Per-panel price-ratio grids.
        # NOTE: ratio_q_grid_u contains q^u/q^r ratios for the URBAN panel
        #       (varies q_u); ratio_q_grid_r contains q^r/q^u ratios for the
        #       RURAL panel (varies q_r).  Same convention for f.
        'ratio_q_grid_u': ratio_q_grid_u,
        'ratio_q_grid_r': ratio_q_grid_r,
        'ratio_f_grid_u': ratio_f_grid_u,
        'ratio_f_grid_r': ratio_f_grid_r,
        # Legacy keys (alias to the urban grid; kept so downstream code that
        # only inspects the urban panel still works).
        'ratio_q_grid':  ratio_q_grid_u,
        'ratio_f_grid':  ratio_f_grid_u,
        # q-grid URBAN-panel demand (varies q_u)
        'q_d_u_owner_base':  q_d_u_owner_base,
        'q_d_u_owner_shock': q_d_u_owner_shock,
        'q_d_u_rent_base':   q_d_u_rent_base,
        'q_d_u_rent_shock':  q_d_u_rent_shock,
        'q_d_u_total_base':  q_d_u_owner_base  + q_d_u_rent_base,
        'q_d_u_total_shock': q_d_u_owner_shock + q_d_u_rent_shock,
        # q-grid RURAL-panel demand (varies q_r)
        'q_d_r_owner_base':  q_d_r_owner_base,
        'q_d_r_owner_shock': q_d_r_owner_shock,
        'q_d_r_rent_base':   q_d_r_rent_base,
        'q_d_r_rent_shock':  q_d_r_rent_shock,
        'q_d_r_total_base':  q_d_r_owner_base  + q_d_r_rent_base,
        'q_d_r_total_shock': q_d_r_owner_shock + q_d_r_rent_shock,
        # f-grid URBAN-panel demand (varies f_u)
        'f_d_u_owner_base':  f_d_u_owner_base,
        'f_d_u_owner_shock': f_d_u_owner_shock,
        'f_d_u_rent_base':   f_d_u_rent_base,
        'f_d_u_rent_shock':  f_d_u_rent_shock,
        'f_d_u_total_base':  f_d_u_owner_base  + f_d_u_rent_base,
        'f_d_u_total_shock': f_d_u_owner_shock + f_d_u_rent_shock,
        # f-grid RURAL-panel demand (varies f_r)
        'f_d_r_owner_base':  f_d_r_owner_base,
        'f_d_r_owner_shock': f_d_r_owner_shock,
        'f_d_r_rent_base':   f_d_r_rent_base,
        'f_d_r_rent_shock':  f_d_r_rent_shock,
        'f_d_r_total_base':  f_d_r_owner_base  + f_d_r_rent_base,
        'f_d_r_total_shock': f_d_r_owner_shock + f_d_r_rent_shock,
        # Supply schedules — now non-vertical for the rural panels.
        'q_s_u_base': q_s_u_base, 'q_s_u_shock': q_s_u_shock,
        'q_s_r_base': q_s_r_base, 'q_s_r_shock': q_s_r_shock,
        'f_s_u_base': f_s_u_base, 'f_s_u_shock': f_s_u_shock,
        'f_s_r_base': f_s_r_base, 'f_s_r_shock': f_s_r_shock,
    }
