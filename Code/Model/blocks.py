"""
blocks.py – DAG-ordered block functions for the HANC Housing Model.

Each block reads inputs (from unknowns, shocks, or outputs of earlier blocks)
and writes outputs in-place.  All path arrays have shape (T, 1).

DAG order (small open economy, Version 3.7):
  1. production_firm_prices  :  Gamma, r_world            →  rK, w
  2. housing_investment      :  H_u, H_r                  →  IH_u, IH_r
  3. construction_firm       :  rK, w, IH_u, IH_r         →  q_u, q_r, K_u, L_u, K_r, L_r
  4. labor_market_close      :  L_supply, L_u, L_r        →  L_tilde  (analytical closure)
  5. production_firm_qtys    :  Gamma, L_tilde, rK        →  K_tilde, Y
  6. mutual_fund             :  K_tilde, K_u, K_r, r_world →  K, r
  7. rental_sector           :  r, q_u, q_r               →  f_u, f_r
  8. hh                      :  r, w, q_u, q_r, f_u, f_r  →  A_hh, C_hh, H_u_hh, H_r_hh
  9. market_clearing         :  ...                        →  clearing_H_u, clearing_H_r
                                                              (clearing_L ≡ 0 by construction)

Open-economy closure:
  - rK = r_world + delta, w independent of L_tilde → labor market closed analytically.
  - r = r_world inside mutual_fund.
  - NFA = A_hh - K is the residual; no clearing_A imposed.

Unknowns: (H_u, H_r);  Targets: (clearing_H_u, clearing_H_r).
"""

import numpy as np
import numba as nb

from GEModelTools import lag, lead


# ==========================================================================
# 0. Interest rate (exogenous world rate, small open economy)
# ==========================================================================

@nb.njit
def interest_rate(par, ini, ss, r_world, r):
    """Small open economy: the real rate is the exogenous world rate."""
    r[:] = r_world


# ==========================================================================
# 1. Production firm (final-goods sector)
# ==========================================================================

@nb.njit
def production_firm_prices(par, ini, ss, Gamma, r, rK, w):
    """Open-economy sub-block: compute factor prices (rK, w) only.

    rK = r + delta where r comes from `interest_rate` (the SGU-adjusted rate
    in SOE).  When SGU is off and ss.r == ss.r_world, this collapses to the
    legacy rK = r_world + delta.  Using r (rather than r_world directly)
    keeps the firm-side capital cost consistent with whatever SGU premium
    was pinned at the steady state via find_ss_prices(nfa_y_target=...).

    w is independent of L_tilde (open economy); this block runs early so w
    is available to the construction firm FOCs.  L_tilde is closed
    analytically afterward by labor_market_close.
    """
    rK[:] = r + par.delta
    kl_ratio = (par.alpha * Gamma / np.maximum(rK, 1e-12)) ** (1.0 / (1.0 - par.alpha))
    w[:] = (1.0 - par.alpha) * Gamma * kl_ratio ** par.alpha


@nb.njit
def production_firm_quantities(par, ini, ss, Gamma, K_tilde, L_tilde, rK, Y):
    """Open-economy sub-block: compute K_tilde and Y from the closed L_tilde.

    Must be called after labor_market_close has set
    L_tilde = L_supply - L_u - L_r.
    """
    L_safe = np.maximum(L_tilde, 1e-12)
    for t in range(K_tilde.shape[0]):
        rK_t = max(rK[t, 0], 1e-12)
        K_tilde[t, 0] = L_safe[t, 0] * (par.alpha * Gamma[t, 0] / rK_t) ** (1.0 / (1.0 - par.alpha))
    K_safe = np.maximum(K_tilde, 1e-12)
    Y[:] = Gamma * K_safe ** par.alpha * L_safe ** (1.0 - par.alpha)


# ==========================================================================
# 2. Housing investment (law of motion for housing stock in each region)
# ==========================================================================

@nb.njit
def housing_investment(par, ini, ss, H_u, H_r, IH_u, IH_r):

    IH_u[:] = H_u - (1.0 - par.delta_H) * lag(ini.H_u, H_u)
    IH_r[:] = H_r - (1.0 - par.delta_H) * lag(ini.H_r, H_r)


# ==========================================================================
# 3. Construction firm (housing-construction sector)
# ==========================================================================

@nb.njit
def construction_firm(par, ini, ss, rK, w, IH_u, IH_r,
                      q_u, q_r, K_u, L_u, K_r, L_r):
    """ Construction firm block """

    mu_1 = par.mu_1
    mu_2 = par.mu_2
    c    = 1.0 - mu_1 - mu_2          # land share

    # One-sided smooth max: IH_safe ≈ max(IH, 0), smooth everywhere.  This is needed for the production-side price FOCs below, which have IH in the numerator and thus can produce NaNs if IH < 0 during Broyden iterations.  The smoothness ensures that the Jacobian is well-defined everywhere, which helps Broyden's method find a safe step back toward the solution when it ventures into negative-IH territory.
    _kappa_softplus = 0.01
    _x_u = IH_u / _kappa_softplus
    _x_r = IH_r / _kappa_softplus
    _x_u_cap = np.minimum(_x_u, 50.0)
    _x_r_cap = np.minimum(_x_r, 50.0)
    IH_u_safe = np.maximum(_kappa_softplus * np.log1p(np.exp(_x_u_cap)), 1e-8)
    IH_r_safe = np.maximum(_kappa_softplus * np.log1p(np.exp(_x_r_cap)), 1e-8)

    # Guard rK to be strictly positive.  During Broyden iterations a bad step
    # can push r + delta < 0; without clamping, (negative)^(1/3)...
    rK_clamped = np.maximum(rK, 1e-8)

    # a. Production-side housing prices from the zero-profit formula.
    q_u_prod = (IH_u_safe * (rK_clamped / mu_1) ** (mu_1 / c)
                          * (w  / mu_2) ** (mu_2 / c)
                          / par.X_u) ** (c / (mu_1 + mu_2))

    q_r_prod = (IH_r_safe * (rK_clamped / mu_1) ** (mu_1 / c)
                          * (w  / mu_2) ** (mu_2 / c)
                          / par.X_r) ** (c / (mu_1 + mu_2))

    # b. Labor from L FOC, evaluated at the production-side price (the factor
    L_u[:] = q_u_prod * mu_2 * IH_u_safe / w
    L_r[:] = q_r_prod * mu_2 * IH_r_safe / w

    # c. Capital from production function (matches SS derivation)
    L_u_safe = np.maximum(L_u, 1e-15)
    L_r_safe = np.maximum(L_r, 1e-15)
    K_u[:] = (IH_u_safe / (L_u_safe ** mu_2 * par.X_u ** c)) ** (1.0 / mu_1)
    K_r[:] = (IH_r_safe / (L_r_safe ** mu_2 * par.X_r ** c)) ** (1.0 / mu_1)

    # Zero-profit production price (floored for solver safety).
    q_u[:] = np.maximum(q_u_prod, 1e-3 * ss.q_u)
    q_r[:] = np.maximum(q_r_prod, 1e-3 * ss.q_r)


# ==========================================================================
# 4. Mutual fund (asset market / capital aggregation)
# ==========================================================================

@nb.njit
def mutual_fund(par, ini, ss, K_tilde, K_u, K_r, rK, K, r):
    """K = K_tilde + K_u + K_r.  In closed economy also r = rK - delta.
    """

    K[:] = K_tilde + K_u + K_r
    if not par.open_economy:
        r[:] = rK - par.delta




# ==========================================================================
# 5. Rental sector (competitive investors)
# ==========================================================================

@nb.njit
def rental_sector(par, ini, ss, r, q_u, q_r, f_u, f_r):
    """Rental-investor FOC (eq. 3.22).
    """
    # Terminal lead closure with partial mean reversion:
    q_u_T = ss.q_u 
    q_r_T = ss.q_r 
    q_u_next = lead(q_u, q_u_T)
    q_r_next = lead(q_r, q_r_T)

    f_u[:] = q_u * (1.0 + r) - (1-par.theta) * q_u_next * (1.0 - par.delta_H)
    f_r[:] = q_r * (1.0 + r) - (1-par.theta) * q_r_next * (1.0 - par.delta_H)


@nb.njit
def labor_market_close(par, ini, ss, L_supply, L_u, L_r, L_tilde):
    """Analytically close the labor market (open-economy only).
    """
    L_tilde[:] = np.maximum(L_supply - L_u - L_r, 1e-12)


@nb.njit
def market_clearing(par, ini, ss,
                    A_hh, K, Y, C_hh, q_u, q_r, IH_u, IH_r,
                    K_tilde, K_u, K_r, L_tilde, L_u, L_r,
                    L_supply,
                    H_u, H_r, H_u_hh, H_r_hh, H_total, H_total_hh,
                    NFA, NFA_to_Y,
                    I, clearing_L, clearing_Y, clearing_resource,
                    clearing_H_r, clearing_H_u):
    """Compute market-clearing residuals (small open economy).

    Housing clearing uses the two regional residuals directly:
        clearing_H_u[t] = H_u[t] - H_u_hh[t]
        clearing_H_r[t] = H_r[t] - H_r_hh[t]

    External balance:
        NFA[t] = A_hh[t] - K[t]
    is the residual.  No clearing equation is imposed on NFA — it is the
    new degree of freedom under r_world exogeneity.
    """

    clearing_L[:] = L_supply - L_tilde - L_u - L_r

    # Housing: keep H_total/H_total_hh as diagnostics only.
    h_ratio_inv = par.h_u / par.h_r
    H_total[:]    = H_u + h_ratio_inv * H_r
    H_total_hh[:] = H_u_hh + h_ratio_inv * H_r_hh

    clearing_H_u[:] = H_u - H_u_hh
    clearing_H_r[:] = H_r - H_r_hh

    I[:]                 = K - (1.0 - par.delta) * lag(ini.K, K)
    clearing_Y[:]        = Y - C_hh - I
    Y_tot                = Y + q_u * IH_u + q_r * IH_r
    clearing_resource[:] = Y_tot - C_hh - I

    # External balance (small open economy).  NFA is the residual.
    NFA[:] = A_hh - K
    Y_safe = np.maximum(np.abs(Y_tot), 1e-12)
    NFA_to_Y[:] = NFA / Y_safe