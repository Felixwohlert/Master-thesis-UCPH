

import warnings
import numpy as np
from scipy.optimize import minimize

# ── 1. Specifications for the Nelder-Mead solver (overwritten in the notebook) ──────────────────────────────────────────

# 'par_name': (initial_value, lower_bound, upper_bound)
# NOTE: 'delta' is computed inside find_ss_indirect_housing and is not a free
#       calibration parameter here.
PARAM_SPEC = {    
    'kappa': (-0.697248, -5.0, 25.0),
    'h_u': (28.5612, 1.0, 40.0),
    'h_r': (13.4599, 1.0, 40.0),
    'h_l': (10.456, 1.0, 20.0),
    'beta': (0.932612, 0.91, 0.99),
    'sigma_psi': (0.12723, 0.1, 0.3),
    'mu_1': (0.429555, 0.0, 0.75),
}

# 'data_panel_title' (must match key in load_datagraphs output): relative weight
TARGET_SPEC = {
    #'Capital/GDP':             1.0,
    #'Urban housing/GDP':       1.0,   # datagraphs units broken (~1000x too small)
    #'Rural housing/GDP':       1.0,   # — target construction/GDP instead
    'Urban/rural price ratio': 1.0,
    'Urban population share':  1.0,
    'Share of renters - Urban region': 1.0,
    'Share of renters - Rural region': 1.0,
    #'Investment/GDP':          1.0,
    'Urban construction/GDP':  1.0,
    'Rural construction/GDP':  1.0,
    #'Top 10% wealth share':    1.0,
    'Middle 40% wealth share': 1.0,  # collinear: Top10 + Mid40 + Bot50 = 1, so
    #   targeting all three is redundant — uncomment only if you drop one.  Also
    #   needs alias 'Middle 40':'Middle 40% wealth share' in plots.load_datagraphs
    #   for the data target to load.
    'Bottom 50% wealth share': 1.0,
    #'Net Foreign Assets/GDP':  1.0,   # 1992 NFA/GDP from datagraphs.xlsx
}


MOMENT_BOUNDS = {
    #'Investment/GDP': (0.14, None),
    #'Urban housing/GDP': (0.10, 0.25),   # broken target — see TARGET_SPEC
    #'Rural housing/GDP': (0.10, 0.25),
    #'Capital/GDP': (1.0, 3.8),  # Relaxed from (3.3, 3.6) — your model gives ~1.5
    'Urban population share': (0.28, 0.35),
    'Share of renters - Urban region': (0.00, 1.00),
    'Share of renters - Rural region': (0.00, 1.00),
    'Urban construction/GDP': (0.0015, 0.010),
    'Rural construction/GDP': (0.003, 0.10),  # Relaxed from (0.02, 0.08) — your model gives ~0.0046
    #'Top 10% wealth share': (0.20, 0.75),   # Relaxed from (0.5, 0.7) — your model gives ~0.25
    'Middle 40% wealth share': (0.30, 0.65),  # data ~0.45; uncomment to bound it
    'Bottom 50% wealth share': (0.0, 0.15),  # Relaxed from (0.0, 0.05) — your model gives ~0.094
    #'Net Foreign Assets/GDP': (-0.3, -0.1),
}

# Manual overrides of data-loaded target values.  Use for moments whose
# datagraphs.xlsx value is unreliable or structurally unreachable by the model.
# Applied after load_targets; an entry takes effect only if the moment is also
# in TARGET_SPEC (i.e. actually targeted).
TARGET_OVERRIDES = {
    'Bottom 50% wealth share': 0.000,  # data = -0.0074; model net wealth >= 0
}

# Explicit feasibility constraints for economically valid steady states.
# Values are (lower, upper), where None means one-sided.
# These are applied as quadratic penalties inside the objective.
SS_BOUNDS = {
    'q_u': (1e-8, None),
    'q_r': (1e-8, None),
    'I': (1e-12, None),       # Very loose; near-zero is OK if solver converges
    'K_tilde': (-1e20, 1e20),  # Very loose bounds; main check is finiteness
    'L_tilde': (1e-12, None),
    'Y': (1e-12, None),
    'rK': (1e-12, None),
}


def _ss_validity_penalty(model, weight=1e4):
    """Quadratic penalty for invalid steady-state allocations/prices.

    This supplements the moment loss with explicit feasibility checks so the
    optimizer avoids non-economic SS points even when moment ratios are finite.
    """
    ss = model.ss
    penalty = 0.0

    for name, (lo, hi) in SS_BOUNDS.items():
        try:
            val = float(getattr(ss, name))
        except Exception:
            # Missing required SS object -> heavy penalty.
            return 1e6

        if not np.isfinite(val):
            return 1e6

        ref = abs(lo if lo is not None else hi) or 1.0
        if lo is not None and val < lo:
            penalty += weight * ((lo - val) / ref) ** 2
        if hi is not None and val > hi:
            penalty += weight * ((val - hi) / ref) ** 2

    # Extra consistency checks that are naturally bounded.
    try:
        urb_share = float(np.sum(ss.D[:, 1, :]) + np.sum(ss.D[:, 3, :]))
        d_mass = float(np.sum(ss.D))
        if d_mass > 0.0 and np.isfinite(d_mass):
            urb_share /= d_mass
            if not (0.0 <= urb_share <= 1.0):
                penalty += weight * (abs(urb_share - np.clip(urb_share, 0.0, 1.0)) / 1.0) ** 2
    except Exception:
        penalty += weight

    return float(penalty)




# ── 2. To compute the model implied wealth shares for calibration purposes ────────────────────────────────────────────

def _compute_wealth_shares(model):
    """Compute top-1%, top-10%, middle-40% and bottom-50% shares of net wealth.

    Net wealth per agent = liquid savings (a)
                         + housing equity for owners: (1 - lambda_ltv) * q^j * h^j

    Uses age-specific policies from model.hh_policy when demographics are
    enabled; falls back to the stationary sol.a[0] otherwise.

    Returns (top1, top10, mid40, bot50). NaNs if the distribution is unavailable.
    """
    import steady_state as ss_mod

    par = model.par
    ss  = model.ss

    lam = float(par.lambda_ltv)
    # Housing equity by state: renters = 0, owners = (1 - λ)*q*h
    eq_by_h = np.array([
        0.0,                                                    # 0: rural_renter
        0.0,                                                    # 1: urban_renter
        (1.0 - lam) * float(ss.q_r) * float(par.h_r),         # 2: rural_owner
        (1.0 - lam) * float(ss.q_u) * float(par.h_u),         # 3: urban_owner
    ])

    Nz = par.Nz
    Nh = par.Nh
    Nm = par.m_grid.size

    D = getattr(ss, 'D', None)
    if D is None:
        return np.nan, np.nan, np.nan, np.nan
    a_pol       = model.sol.a[0]   # (Nz, Nh, Nm)
    wealth_flat = (a_pol + eq_by_h[np.newaxis, :, np.newaxis]).ravel()
    mass_flat   = D.ravel()

    pos = mass_flat > 0
    if pos.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    wealth_flat = wealth_flat[pos]
    mass_flat   = mass_flat[pos] / mass_flat[pos].sum()

    order      = np.argsort(wealth_flat)
    w_sorted   = wealth_flat[order]
    m_sorted   = mass_flat[order]
    cum_mass   = np.cumsum(m_sorted)
    cum_wealth = np.cumsum(m_sorted * w_sorted)
    total_w    = cum_wealth[-1]

    if not np.isfinite(total_w) or total_w == 0.0:
        return np.nan, np.nan, np.nan, np.nan

    cut99  = np.searchsorted(cum_mass, 0.99, side='left')
    top1   = (total_w - (cum_wealth[cut99 - 1] if cut99 > 0 else 0.0)) / total_w

    cut90  = np.searchsorted(cum_mass, 0.90, side='left')
    top10  = (total_w - (cum_wealth[cut90 - 1] if cut90 > 0 else 0.0)) / total_w

    cut50  = np.searchsorted(cum_mass, 0.50, side='left')
    bot50  = (cum_wealth[cut50 - 1] if cut50 > 0 else 0.0) / total_w

    # Middle 40% = 50th-90th percentile share — taken as the residual so the
    # three partition shares sum to 1 (matches simulation.wealth_statistics).
    mid40 = 1.0 - float(top10) - float(bot50)

    return float(top1), float(top10), float(mid40), float(bot50)


# ── 3. To compute steady-state moments for calibration purposes ────────────────────────────────────────────────────────

def compute_ss_moments(model):
    """
    Compute calibration moments from a solved steady state.

    Returns a dict whose keys match the titles in TARGET_SPEC / load_datagraphs.
    Unavailable quantities are np.nan.
    """
    ss = model.ss

    def _f(name):
        v = getattr(ss, name, None)
        try:
            return float(v) if (v is not None and np.isfinite(float(v))) else np.nan
        except Exception:
            return np.nan

    Y      = _f('Y');    K      = _f('K')
    q_u    = _f('q_u'); q_r    = _f('q_r')
    H_u    = _f('H_u'); H_r    = _f('H_r')
    IH_u   = _f('IH_u'); IH_r  = _f('IH_r')
    I      = _f('I')

    # GDP definition consistent with plots.py: Y_tot = Y + q_u*IH_u + q_r*IH_r
    if all(np.isfinite([Y, q_u, IH_u, q_r, IH_r])):
        Y_tot = Y + q_u * IH_u + q_r * IH_r
    else:
        Y_tot = np.nan

    def _ratio(num, den=None):
        if den is None:
            den = Y_tot
        return num / den if (np.isfinite(num) and np.isfinite(den) and den > 0) else np.nan

    # Urban population share: mass of individuals in housing state 1 (urban)
    # ss.D has shape (Nz, Nh, Na); Nh index 0 = rural, 1 = urban
    D = getattr(ss, 'D', None)
    if D is not None and isinstance(D, np.ndarray) and D.ndim == 3:
        d_total = D.sum()
        # Urban = states 1 (urban_renter) and 3 (urban_owner)
        urb_share = float((D[:, 1, :].sum() + D[:, 3, :].sum()) / d_total) if d_total > 0 else np.nan
    else:
        urb_share = np.nan

    urban_renter_share = D[:, 1, :].sum() / (D[:, 1, :].sum()+D[:, 3, :].sum()) if (D[:, 1, :].sum()+D[:, 3, :].sum()) > 0 else np.nan
    rural_renter_share = D[:, 0, :].sum() / (D[:, 0, :].sum()+D[:, 2, :].sum()) if (D[:, 0, :].sum()+D[:, 2, :].sum()) > 0 else np.nan

    moments = {
        'Capital/GDP':
            _ratio(K, Y_tot),
        'Urban housing/GDP':
            _ratio(q_u * H_u, Y_tot) if np.isfinite(q_u * H_u) else np.nan,
        'Rural housing/GDP':
            _ratio(q_r * H_r, Y_tot) if np.isfinite(q_r * H_r) else np.nan,
        'Urban/rural price ratio':
            q_u / q_r if (np.isfinite(q_u * q_r) and q_r > 0) else np.nan,
        'Urban population share':
            urb_share,
        'Share of renters - Urban region':
            urban_renter_share if np.isfinite(urban_renter_share) else np.nan,
        'Share of renters - Rural region':
            rural_renter_share if np.isfinite(rural_renter_share) else np.nan,
        'Investment/GDP':
            _ratio(I, Y_tot),
        'Urban construction/GDP':
            _ratio(q_u * IH_u, Y_tot) if np.isfinite(q_u * IH_u) else np.nan,
        'Rural construction/GDP':
            _ratio(q_r * IH_r, Y_tot) if np.isfinite(q_r * IH_r) else np.nan,
        'Net Foreign Assets/GDP':
            _ratio(_f('NFA'), Y_tot),
    }

    # Wealth distribution moments
    try:
        top1_share, top10_share, mid40_share, bot50_share = _compute_wealth_shares(model)
    except Exception:
        top1_share = top10_share = mid40_share = bot50_share = np.nan
    moments['Top 1% wealth share']     = top1_share
    moments['Top 10% wealth share']    = top10_share
    moments['Middle 40% wealth share'] = mid40_share
    moments['Bottom 50% wealth share'] = bot50_share

    # External balance: NFA / GDP (matches the datagraphs 'Net Foreign Assets' column,
    # which is in NFA/GDP units — first observation = 1992).
    NFA = _f('NFA')
    moments['Net Foreign Assets/GDP'] = _ratio(NFA, Y_tot)

    return moments


# ── 4. Target loading from data contained in datagraphs.xlsx ─────────────────────────────────────────────────────────

def load_targets(data_path, target_spec, base_year=1992):
    """
    Read target values for all moments in *target_spec* from an xlsx file.

    Returns a dict {moment_key: float} containing only moments for which a
    finite value at *base_year* was found.  A warning is printed for missing
    entries.
    """
    from plots import load_datagraphs

    data = load_datagraphs(data_path, start_year=base_year)
    targets = {}
    for key in target_spec:
        if key not in data:
            print(f"  [warn] '{key}' not found in {data_path} — excluded")
            continue
        yrs, vals = data[key]
        idx = np.where(np.asarray(yrs) == base_year)[0]
        if len(idx) == 0:
            print(f"  [warn] year {base_year} not found for '{key}' — excluded")
            continue
        v = float(vals[idx[0]])
        if not np.isfinite(v):
            print(f"  [warn] value for '{key}' at {base_year} is not finite — excluded")
            continue
        targets[key] = v
    return targets

# ── 5. Calibration ────────────────────────────────────────────────────────────

def calibrate(model, param_spec=None, target_spec=None, moment_bounds=None,
              target_overrides=None,
              data_path='datagraphs.xlsx', base_year=1992,
              maxiter=150, max_evals=None, xatol=1e-5, fatol=1e-7,
              initial_simplex_scale=0.1,
              verbose=True):
    """
    Calibrate *model* to *base_year* data targets via Nelder-Mead MDE.

    Parameters
    ----------
    model         : HANCHousingModelClass (set up and allocate'd)
    param_spec    : dict {'par_name': (initial, lb, ub)} — defaults to PARAM_SPEC
    target_spec   : dict {'moment_key': weight}          — defaults to TARGET_SPEC
    moment_bounds : dict {'moment_key': (lower, upper)}  — defaults to MOMENT_BOUNDS
                    Use None on either side for a one-sided bound.
                    A quadratic penalty (weight 1e3) is added to the loss when
                    the model moment violates the bound.
                    Additional SS feasibility penalties are always enforced
                    via SS_BOUNDS (q_u, q_r, I, K_tilde, L_tilde, Y, rK, delta).
    target_overrides : dict {'moment_key': value} — manual replacements for
                    data-loaded target values; defaults to TARGET_OVERRIDES.
                    Applied after load_targets; only targeted moments affected.
    data_path     : path to datagraphs.xlsx
    base_year     : calibration year (default 1992)
    maxiter       : Nelder-Mead max iterations
    max_evals     : hard cap on objective evaluations (the 'iter N' counter in
                    the output) — maps to Nelder-Mead's maxfev.  None (default)
                    lets scipy use its large default.  Set it to stop early,
                    read x_cal, and restart the next run from those values.
    xatol/fatol   : Nelder-Mead tolerances
    initial_simplex_scale : fraction of each parameter's (ub - lb) range used as
                    the initial simplex step size.  Default 0.1 (10% of range).
                    Scipy's default builds the simplex by perturbing x0 by only 5%
                    of |x0|, which can be tiny for e.g. q_r_ss_target≈50.  Setting
                    this to 0.1–0.25 gives larger, better-scaled initial steps and
                    helps Nelder-Mead explore the loss surface more aggressively.
    verbose       : print progress every 25 evaluations and a final report

    Returns
    -------
    result      : scipy OptimizeResult
    x_cal       : calibrated parameter array (same order as param_names)
    moments_cal : dict of moments at the calibrated SS
    """
    import steady_state  # local import so module loads without the full GE stack

    if param_spec    is None: param_spec    = PARAM_SPEC
    if target_spec   is None: target_spec   = TARGET_SPEC
    if moment_bounds is None: moment_bounds = MOMENT_BOUNDS
    if target_overrides is None: target_overrides = TARGET_OVERRIDES

    # Stash the resolved spec so a later report(model=...) reflects the targets
    # this run actually used — not the (possibly stale) module-level TARGET_SPEC.
    model._calib_target_spec = target_spec

    # Load targets
    targets = load_targets(data_path, target_spec, base_year=base_year)

    # Apply manual target overrides (datagraphs values that are unreliable or
    # structurally unreachable).  Only moments present in target_spec take effect.
    for _m, _v in target_overrides.items():
        if _m in target_spec:
            targets[_m] = float(_v)

    if not targets:
        raise ValueError("No valid targets found — cannot calibrate.")

    if verbose:
        print("Active calibration targets:")
        for m, v in targets.items():
            print(f"  {m:35s} {v:.4f}  (w = {target_spec[m]:.1f})")

    # Parameter arrays
    param_names = list(param_spec.keys())
    x0 = np.array([param_spec[p][0] for p in param_names])
    lb = np.array([param_spec[p][1] for p in param_names])
    ub = np.array([param_spec[p][2] for p in param_names])

    # Ensure calibration starts from PARAM_SPEC initials for calibrated params.
    for name, val in zip(param_names, x0):
        setattr(model.par, name, float(val))

    x_original = {p: float(x0[i]) for i, p in enumerate(param_names)}

    # Objective
    _iter = [0]
    # Warm-start guess [r, w, q_u, q_r] for find_ss_prices.  Updated after every
    # successful solve so each Nelder-Mead evaluation starts from the previous
    # SS solution (nearby parameters → nearby prices → faster, more robust solve).
    _warm_x0 = [None]
    # Best (lowest-loss) point seen so far.  Mirrored onto model._calib_best so
    # an interrupted run is still recoverable — Nelder-Mead otherwise leaves the
    # model at the *last* trial point, not the best.
    _best = {'loss': np.inf}

    def objective(x):
        _iter[0] += 1
        x_c = np.clip(x, lb, ub)   # Nelder-Mead doesn't enforce bounds natively
        for name, val in zip(param_names, x_c):
            setattr(model.par, name, float(val))
        
        # ── Enforce derived-parameter constraints ─────────────────────────────
        # These are fixed relationships between parameters that should not be
        # calibrated independently. Enforce them immediately after each parameter update.
        if hasattr(model.par, 'q_r_ss_target'):
            model.par.q_u_ss_target = model.par.q_r_ss_target * 1.549
        if hasattr(model.par, 'X_r'):
            model.par.X_u = model.par.X_r * 0.065
        # ──────────────────────────────────────────────────────────────────────
        
        # Rebuild grids from the updated parameters before solving: z_trans
        # depends on rho_z, z_grid on sigma_psi (and its de-meaning on z_trans),
        # and the housing grids on h_u / h_r.  Without this rebuild rho_z and
        # sigma_psi are silently inert during the Nelder-Mead search.
        model.create_grids()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ss_success, _ = steady_state.find_ss_prices(
                    model, do_print=False, x0=_warm_x0[0])
        except Exception:
            return 1e6

        # Carry a converged SS solution forward as the next call's initial
        # guess.  Only update on success so a failed solve can't poison the
        # warm start with garbage prices.
        if ss_success:
            cand = np.array([float(model.ss.r),   float(model.ss.w),
                             float(model.ss.q_u), float(model.ss.q_r)])
            if np.all(np.isfinite(cand)):
                _warm_x0[0] = cand

        moments = compute_ss_moments(model)
        loss = 0.0

        # Explicit feasibility penalties for economically valid SS.
        # These constraints are separate from data moments and keep the
        # optimizer away from non-economic local minima.
        loss += _ss_validity_penalty(model)

        for m, tgt in targets.items():
            mv = moments.get(m, np.nan)
            if not np.isfinite(mv):
                loss += 1e4   # heavy penalty for missing moment
                continue
            # Relative error normally; absolute error when the target is 0
            # (relative error is undefined / hypersensitive near a zero target).
            if tgt != 0.0:
                loss += target_spec[m] * ((mv - tgt) / tgt) ** 2
            else:
                loss += target_spec[m] * (mv - tgt) ** 2

        # Moment bounds: quadratic penalty when a model moment violates a bound.
        # Penalty weight 1e3 >> typical MDE loss (~1) so the bound acts as a
        # near-hard constraint without requiring a constrained solver.
        _BOUND_WEIGHT = 1e3
        for m, (lo, hi) in moment_bounds.items():
            mv = moments.get(m, np.nan)
            if not np.isfinite(mv):
                continue
            ref = abs(lo if lo is not None else hi) or 1.0
            if lo is not None and mv < lo:
                loss += _BOUND_WEIGHT * ((lo - mv) / ref) ** 2
            if hi is not None and mv > hi:
                loss += _BOUND_WEIGHT * ((mv - hi) / ref) ** 2

        # Record the best point so far (recoverable after an interrupt).
        if loss < _best['loss']:
            _best['loss'] = float(loss)
            model._calib_best = {
                'loss':   float(loss),
                'x':      np.asarray(x_c, dtype=float).copy(),
                'params': {p: float(v) for p, v in zip(param_names, x_c)},
            }

        if verbose and _iter[0] % 5 == 0:
            print(f"  iter {_iter[0]:4d}  loss = {loss:.4e}")
        return loss

    if verbose:
        print(f"\nCalibrating {len(param_names)} parameter(s): {param_names}")
        print(f"Against     {len(targets)} moment(s)\n")

    # Build explicit initial simplex so step sizes are proportional to each
    # parameter's allowed range rather than a fixed 5% of |x0| (scipy default).
    # Shape: (n_params + 1, n_params).  Row 0 = x0; row i = x0 with parameter
    # i-1 displaced by initial_simplex_scale * (ub[i-1] - lb[i-1]).
    n_p = len(x0)
    initial_simplex = np.empty((n_p + 1, n_p))
    initial_simplex[0] = x0
    for _i in range(n_p):
        row = x0.copy()
        step = float(initial_simplex_scale) * (ub[_i] - lb[_i])
        row[_i] = np.clip(x0[_i] + step, lb[_i], ub[_i])
        initial_simplex[_i + 1] = row

    if verbose:
        _l0 = objective(x0)
        print(f"  Initial loss (x0) = {_l0:.4e}")
        if _l0 == 1e6:
            print("  WARNING: SS solver failed at x0 (returned sentinel 1e6). "
                  "Check that PARAM_SPEC initial values give a solvable steady state.")

    nm_options = {
        'maxiter':         maxiter,
        'xatol':           xatol,
        'fatol':           fatol,
        'disp':            False,
        'adaptive':        True,      # scales simplex; important for >2 params
        'initial_simplex': initial_simplex,
    }
    # max_evals caps objective evaluations (the 'iter N' in the output) via
    # Nelder-Mead's maxfev.  Left unset by default so scipy uses its large
    # default; set it to stop early and restart the next run from result.x.
    if max_evals is not None:
        nm_options['maxfev'] = int(max_evals)

    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options=nm_options,
    )

    # Final clean SS at calibrated parameters — also warm-started from the
    # last converged solution found during the search.
    x_cal = np.clip(result.x, lb, ub)
    for name, val in zip(param_names, x_cal):
        setattr(model.par, name, float(val))
    steady_state.find_ss_prices(model, do_print=False, x0=_warm_x0[0])
    moments_cal = compute_ss_moments(model)

    if verbose:
        report(result, param_names, x_original, x_cal, targets, target_spec, moments_cal, model=model)
        # Paste-ready PARAM_SPEC body to restart the next run from this result.
        print("\n  PARAM_SPEC initials for the next run "
              "(calibrated value, lb, ub):")
        for name, xc in zip(param_names, x_cal):
            lo, hi = param_spec[name][1], param_spec[name][2]
            print(f"    '{name}': ({float(xc):.6g}, {lo}, {hi}),")

    return result, x_cal, moments_cal, param_names




# ── 6. Report ─────────────────────────────────────────────────────────────────

def report(result=None, param_names=None, x_original=None, x_cal=None,
        targets=None, target_spec=None, moments_cal=None, model=None,
        data_path='datagraphs.xlsx', base_year=1992):
    """Print calibration summary and/or model moments vs data + SS levels.

    Supports two workflows:
    1) Calibration report (legacy): pass result, parameter objects, targets and moments_cal.
    2) Direct SS report: pass model only; targets and moments are built internally.
    """

    if target_spec is None:
        # Prefer the spec the last calibrate() run actually used, so the report
        # doesn't silently fall back to a stale module-level TARGET_SPEC.
        target_spec = getattr(model, '_calib_target_spec', None) or TARGET_SPEC

    # Direct-use path: construct targets/moments from model when not supplied.
    if targets is None:
        targets = load_targets(data_path, target_spec, base_year=base_year)

    if moments_cal is None:
        if model is None:
            raise ValueError('report(): provide either moments_cal or model.')
        moments_cal = compute_ss_moments(model)

    # Calibration header and parameter table (only shown when provided).
    if result is not None:
        print("\n" + "=" * 60)
        print("CALIBRATION RESULT")
        print("=" * 60)
        print(f"  Converged : {result.success}  —  {result.message}")
        print(f"  Objective : {result.fun:.4e}")
        print(f"  Iterations: {result.nit}  (func evals: {result.nfev})")

        if (param_names is not None) and (x_original is not None) and (x_cal is not None):
            print(f"\n  {'Parameter':22s} {'Original':>10s} {'Calibrated':>12s}")
            print("  " + "-" * 46)
            for name, xc in zip(param_names, x_cal):
                xo = x_original.get(name, np.nan) if isinstance(x_original, dict) else np.nan
                print(f"  {name:22s} {xo:10.4f} {xc:12.4f}")

    # Moment table (always shown when targets exist).
    print(f"\n  {'Moment':35s} {'Model':>8s} {'Target':>8s} {'Dev %':>8s}")
    print("  " + "-" * 64)
    for m, tgt in targets.items():
        mv  = moments_cal.get(m, np.nan)
        dev = (mv / tgt - 1.0) * 100 if (np.isfinite(mv) and tgt != 0) else np.nan
        flag = "  ✓" if (np.isfinite(dev) and abs(dev) <= 5) else "  ✗"
        print(f"  {m:35s} {mv:8.4f} {tgt:8.4f} {dev:+8.2f}%{flag}")

    # ── Informational sections: wealth ratios and wealth inequality ─────────
    # Always shown, independent of TARGET_SPEC. Data values are pulled from
    # the same xlsx file used by load_targets (silent if a key is missing).
    try:
        from plots import load_datagraphs
        data_full = load_datagraphs(data_path, start_year=base_year)
    except Exception:
        data_full = {}

    def _data_at(key, year):
        if key not in data_full:
            return np.nan
        yrs, vals = data_full[key]
        idx = np.where(np.asarray(yrs) == year)[0]
        if len(idx) == 0:
            return np.nan
        v = float(vals[idx[0]])
        return v if np.isfinite(v) else np.nan

    def _print_section(title, items):
        print(f"\n  {title}")
        print(f"  {'Moment':35s} {'Model':>8s} {'Data':>8s} {'Dev %':>8s}")
        print("  " + "-" * 64)
        for key, label in items:
            mv = moments_cal.get(key, np.nan)
            dv = _data_at(key, base_year)
            if np.isfinite(mv) and np.isfinite(dv) and dv != 0:
                dev = (mv / dv - 1.0) * 100
                dev_str = f"{dev:+8.2f}%"
                flag = "  ✓" if abs(dev) <= 5 else "  ✗"
            else:
                dev_str = f"{'n/a':>8s} "
                flag = ""
            dv_str = f"{dv:8.4f}" if np.isfinite(dv) else f"{'n/a':>8s}"
            mv_str = f"{mv:8.4f}" if np.isfinite(mv) else f"{'n/a':>8s}"
            print(f"  {label:35s} {mv_str} {dv_str} {dev_str}{flag}")

    _print_section("Wealth ratios", [
        ('Capital/GDP',            'Capital/GDP'),
        ('Urban housing/GDP',      'Urban housing/GDP'),
        ('Rural housing/GDP',      'Rural housing/GDP'),
        ('Net Foreign Assets/GDP', 'NFA/GDP'),
    ])
    _print_section("Wealth inequality", [
        ('Top 1% wealth share',     'Top 1%'),
        ('Top 10% wealth share',    'Top 10%'),
        ('Middle 40% wealth share', 'Middle 40%'),
        ('Bottom 50% wealth share', 'Bottom 50%'),
    ])

    # SS levels table for current model.ss.
    if model is not None and hasattr(model, 'ss'):
        ss = model.ss
        ss_vars = [
            'r', 'rK', 'w', 'q_u', 'q_r', 'f_u', 'f_r',
            'A_hh', 'C_hh', 'H_u_hh', 'H_r_hh',
            'H_u', 'H_r', 'NFA',
            'K', 'K_tilde', 'K_u', 'K_r',
            'L', 'L_tilde', 'L_u', 'L_r',
            'IH_u', 'IH_r',
            'X_u', 'X_r',
            'Gamma', 'delta',
        ]

        print(f"\n  {'SS variable':22s} {'Value':>14s}")
        print("  " + "-" * 38)
        for name in ss_vars:
            # Most entries live on ss; these structural terms live on par.
            if name in ('X_u', 'X_r', 'delta') and hasattr(model, 'par') and hasattr(model.par, name):
                val = getattr(model.par, name)
            else:
                val = getattr(ss, name, np.nan)
            try:
                val_f = float(val)
            except Exception:
                val_f = np.nan
            print(f"  {name:22s} {val_f:14.6f}")
    print()


# ── 7. One-at-a-time elasticity matrix ────────────────────────────────────────

def elasticity_matrix(m, param_names=None, step=0.05, do_print=True):
    """One-at-a-time relative-sensitivity matrix S[moment, par] = d ln(moment) / d ln(par).

    For each parameter p, perturb it by `step * baseline_value`, re-solve the
    steady state via steady_state.find_ss_prices, and record the relative
    change in every moment returned by compute_ss_moments.

    Mirrors derived-parameter side effects (X_u/X_r ratio, q_u_ss_target
    ratio, h_u/h_r/sigma_psi grid rebuild) that calibration.objective does
    NOT perform, so the elasticities reflect the true model response.

    Args:
        m            : HANCHousingModel instance with a solved baseline ss.
        param_names  : iterable of par-attribute names; default PARAM_SPEC keys.
        step         : relative perturbation size (default 5%).
        do_print     : per-parameter progress + final table.

    Returns:
        pandas.DataFrame (moments x parameters) of elasticities.
    """
    import pandas as pd
    import steady_state

    if param_names is None:
        param_names = list(PARAM_SPEC.keys())
    param_names = list(param_names)

    def _apply_side_effects():
        if hasattr(m.par, 'q_r_ss_target'):
            m.par.q_u_ss_target = m.par.q_r_ss_target * 1.549
        if hasattr(m.par, 'X_r'):
            m.par.X_u = m.par.X_r * 0.065
        m.par.h_u_grid = np.array([0.0, m.par.h_u])
        m.par.h_r_grid = np.array([0.0, m.par.h_r])
        psi = m.par.sigma_psi * np.sqrt(m.par.Nz - 1)
        m.par.z_grid = np.linspace(-psi, psi, m.par.Nz)

    def _solve_moments(warm=None):
        try:
            steady_state.find_ss_prices(m, do_print=False, x0=warm)
        except Exception:
            return None
        return compute_ss_moments(m)

    x0 = {p: float(getattr(m.par, p)) for p in param_names}

    # --- baseline ------------------------------------------------------------
    for p, v in x0.items():
        setattr(m.par, p, v)
    _apply_side_effects()
    base = _solve_moments()
    if base is None:
        raise RuntimeError('Baseline SS solve failed — fix the model state first.')
    warm = np.array([m.ss.r, m.ss.w, m.ss.q_u, m.ss.q_r])
    moment_names = list(base.keys())
    if do_print:
        print(f'baseline solved: {len(param_names)} parameters x {len(moment_names)} moments')

    # --- one-at-a-time perturbations ----------------------------------------
    S = pd.DataFrame(index=moment_names, columns=param_names, dtype=float)
    for p in param_names:
        for q, v in x0.items():                       # reset every parameter
            setattr(m.par, q, v)
        bp = x0[p]
        dp = step * bp if abs(bp) > 1e-12 else step   # absolute step if param ~ 0
        setattr(m.par, p, bp + dp)
        _apply_side_effects()
        mom = _solve_moments(warm=warm)
        rel = dp / bp if abs(bp) > 1e-12 else np.nan
        for mn in moment_names:
            b = base.get(mn, np.nan)
            v2 = mom.get(mn, np.nan) if mom is not None else np.nan
            S.loc[mn, p] = (((v2 - b) / b) / rel
                            if (np.isfinite(b) and b != 0.0 and
                                np.isfinite(v2) and np.isfinite(rel)) else np.nan)
        if do_print:
            print(f'  {p:24s} {"ok" if mom is not None else "SS SOLVE FAILED"}')

    # restore baseline and leave model.ss at the baseline solution
    for p, v in x0.items():
        setattr(m.par, p, v)
    _apply_side_effects()
    steady_state.find_ss_prices(m, do_print=False, x0=warm)

    if do_print:
        pd.set_option('display.width', 200, 'display.max_columns', 50)
        print('\nElasticity matrix  (% change in moment per % change in parameter):')
        print(S.round(2))

        S_norm = S.div(S.abs().max(axis=1).replace(0.0, np.nan), axis=0)
        print('\nRow-normalised (cells near +/-1 mark the identifying parameter of each moment):')
        print(S_norm.round(2))

        print('\nDominant parameter for each moment:')
        for mn in moment_names:
            row = S.loc[mn].astype(float)
            if row.abs().notna().any():
                p_star = row.abs().idxmax()
                print(f'  {mn:34s} <- {p_star:24s} ({row[p_star]:+.2f})')

        print('\nDominant moment for each parameter:')
        for p in param_names:
            col = S[p].astype(float)
            if col.abs().notna().any():
                m_star = col.abs().idxmax()
                print(f'  {p:24s} -> {m_star:34s} ({col[m_star]:+.2f})')

    return S