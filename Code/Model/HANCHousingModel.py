


##############
# 1. imports #
##############

import numpy as np

# consav package

# EconModel and GEModelTools
from EconModel import EconModelClass
from GEModelTools import GEModelClass


# local modules


# local modules for GE
import steady_state
import household_problem
import blocks
import plots
import transition


def _rouwenhorst_transition(n, rho):
    """Build Rouwenhorst transition matrix for AR(1) process."""
    if n <= 1:
        return np.ones((1, 1))

    p = (1.0 + rho) / 2.0
    q = p
    P = np.array([[1.0]])

    for k in range(2, n + 1):
        P_old = P
        P = np.zeros((k, k))

        P[:-1, :-1] += p * P_old
        P[:-1, 1:] += (1.0 - p) * P_old
        P[1:, :-1] += (1.0 - q) * P_old
        P[1:, 1:] += q * P_old

        P[1:-1, :] *= 0.5

    return P


def _stationary_distribution(P, tol=1e-14, max_iter=100000):
    """Compute stationary distribution for Markov transition matrix."""
    n = P.shape[0]
    pi = np.ones(n) / n

    for _ in range(max_iter):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new / np.sum(pi_new)
        pi = pi_new

    return pi / np.sum(pi)


class HANCHousingModelClass(EconModelClass,GEModelClass):    

    # ==========================================================================
    # 1. Model settings
    # ==========================================================================

    def settings(self):
        """ fundamental settings """

        # a. namespaces      
        self.namespaces = ['par','ini','sim','ss','path','sol']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a', 'c', 'pr_urban', 'pr_choices'] # policy functions
        self.inputs_hh = ['r', 'w', 'q_u', 'q_r', 'f_u', 'f_r', 'kappa'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a', 'c', 'h_u', 'h_r'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE — small open economy
        # Open-economy closure: r_world is exogenous, the firm's FOC rK = r_world + delta
        # pins K_tilde analytically inside production_firm, the asset-market clearing
        # A = A_hh is dropped, and NFA = A_hh - K becomes the residual degree of freedom.
        self.shocks = ['Gamma', 'L_supply', 'hh_scale', 'kappa', 'r_world']
        self.unknowns = ['H_u', 'H_r']
        self.targets = ['clearing_H_u', 'clearing_H_r']

        # d. DAG-ordering
        # Open-economy split: production_firm_prices runs first (gives w, rK
        # without L_tilde), construction_firm gives L_u/L_r, then
        # labor_market_close analytically pins L_tilde = L_supply - L_u - L_r,
        # then production_firm_quantities computes K_tilde and Y.
        # clearing_L is kept in market_clearing as a diagnostic only.
        self.blocks = [ # list of strings to block-functions
            'blocks.interest_rate',          # SGU debt-elastic premium → writes r
            'blocks.production_firm_prices',
            'blocks.housing_investment',
            'blocks.construction_firm',
            'blocks.labor_market_close',
            'blocks.production_firm_quantities',
            'blocks.mutual_fund',
            'blocks.rental_sector',
            'hh', # household block
            'blocks.market_clearing']


        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.plot_policy = plots.plot_policy
        self.plot_value_function = plots.plot_value_function
        self.plot_transition_paths = plots.plot_transition_paths
        self.plot_hh_jacobians = plots.plot_hh_jacobians
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.find_ss_indirect_housing = steady_state.find_ss_indirect_housing
        self.find_ss_prices = steady_state.find_ss_prices
        self.compute_age_specific_fake_news_matrices = transition.compute_age_specific_fake_news_matrices
        self.jacobian_from_fake_news_matrix = transition.jacobian_from_fake_news_matrix
        self.compute_age_specific_hh_jacobians = transition.compute_age_specific_hh_jacobians

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # a. Grid parameters
        par.T = 79  
        par.Na = 500  
        par.Nh = 4  # housing states: 0=rural_renter,1=urban_renter,2=rural_owner,3=urban_owner
        par.ngridm = par.Na
        par.mmax = 200.0
        par.expn = 5
        par.a_max = 1000.0
        
        # b. Fixed types
        par.Nfix = 1  # number of fixed types (could expand for heterogeneity)
        
        # c. Productivity types
        par.Nz = 5
        par.Nz_total = par.Nfix * par.Nz
    

        # d. Production and technology
        par.alpha = 0.30  # capital share in production
        par.delta = 0.025  # depreciation rate      
        par.Gamma = 1.0  # total factor productivity
        par.L_supply = 1.0  # labor-force scale (normalized to 1 in base year)
        par.hh_scale = 1.0  # direct HH level scale shock (N_t), defaults to no scaling
        par.X_r = 20.0 # land in the rural region
        par.X_u = par.X_r*0.065 # land in the urban region
        par.mu_1 = 0.429555
        par.mu_2 = 0.75 - par.mu_1
        par.delta_H = 0.025 # housing depreciation
        par.NFA_target = -40.0
        
        # e. partial equilibrium objects
        par.r = 0.05592
        par.q_r = 0.1496   # price of rural housing
        par.q_u = 0.3353 # price of urban housing relative to rural
        par.f_r = 0.08010  # rent in rural region (per unit of housing service)
        par.f_u = 0.17954  # rent in urban region relative to rural

        # f. preferences
        par.beta = 0.932612
        par.rho = 2.0
        par.gamma = 0.9999
        par.kappa = -0.697248  # additive preference shifter for urban housing in marginal utility

        # g. housing parameters
        par.h_u = 28.5612  # urban housing service level
        par.h_r = 13.4599  # rural housing service level
        par.h_l = 10.456   # rental unit housing service level (same for both regions; h_l < h_u, h_r gives incentive to own)
        par.zeta = 0.4  # transaction cost as fraction of house value when switching housing types
        par.zeta_renter = par.zeta  # renter migration cost as fraction of aggregate wage when changing region (rural_renter <-> urban_renter)
        par.lambda_ltv = 0.8  # loan-to-value ratio (LTV) on new mortgages
        par.T_mort = 10000000   # mortgage repayment horizon (years; amortisation = b/T_mort per period)

        # g2. small-open-economy closure
        par.open_economy = True            # r pinned to r_world; K_tilde analytical
        par.r_world = par.r                # SS world real rate (level)
        par.n_terminal_anchor = 2          # pin (H_u, H_r) to SS in last n periods (hard terminal condition)
        
        # h. income parameters
        par.rho_z = 0.85 # AR(1) parameter
        par.sigma_psi = 0.12723 # std. of persistent shock
        par.cfloor = 0.001

        # i. smoothing parameter for choice probabilities
        par.sigma = 1.5

        # j. rental regulation
        par.theta = 0.4662  # fraction of sales revenue handed to government by rental investors
        
        # j. simulation
        par.simT = par.T  # number of periods to simulate
        par.simN = 1000  # number of households to simulate
        par.J = min(49, par.T)  # lifecycle length (number of ages in the backward DC-EGM)
        par.newborn_urban_share = 0.3

        # k. Taxes
        par.tau_wealth = 0.0
        par.tau_profits = 0.0


        # l. steady state targets (for indirect method - will be overwritten if specified)
        par.r_ss_target = 0.03  # default target interest rate
        par.w_ss_target = 1.0   # default target wage
        par.q_r_ss_target = 1.0 # default target rural house price
        par.q_u_ss_target = par.q_r_ss_target * 1.5490 # default target urban house price
        par.rK_ss_target = 0.055  # default target rental rate of capital


    # ==========================================================================
    # 2. Allocation of variables to GE framework
    # ==========================================================================    

    def allocate_GE(self):
        """
        Custom allocation method 
        """
        par = self.par
        
        # Allocate steady state and path arrays for GE variables
        # This is the essential part of allocate_GE without the assertions
        
        # a. Allocate household aggregate outputs: A_hh, C_hh, H_u_hh, H_r_hh
        for varname in self.outputs_hh:
            # Map lowercase outputs_hh names to actual variable names
            # 'a' -> 'A_hh', 'c' -> 'C_hh', 'h_u' -> 'H_u_hh', 'h_r' -> 'H_r_hh'
            if varname == 'a':
                Varname_hh = 'A_hh'
            elif varname == 'c':
                Varname_hh = 'C_hh'
            elif varname == 'h_u':
                Varname_hh = 'H_u_hh'
            elif varname == 'h_r':
                Varname_hh = 'H_r_hh'
            else:
                # Fallback: capitalize first letter only
                Varname_hh = f'{varname[0].upper()}{varname[1:]}_hh'
            
            # ss: scalar values
            setattr(self.ss, Varname_hh, 0.0)
            # path: time series
            setattr(self.path, Varname_hh, np.zeros((par.T, 1)))
        
        # b. Allocate GE equilibrium variables from varlist (if it exists)
        if hasattr(self, 'unknowns'):
            for varname in self.unknowns:
                if not hasattr(self.ss, varname):
                    setattr(self.ss, varname, 0.0)
                if (not hasattr(self.path, varname)) or (not isinstance(getattr(self.path, varname), np.ndarray)) or (getattr(self.path, varname).shape != (par.T, 1)):
                    setattr(self.path, varname, np.zeros((par.T, 1)))
        
        if hasattr(self, 'targets'):
            for varname in self.targets:
                if not hasattr(self.ss, varname):
                    setattr(self.ss, varname, 0.0)
                if (not hasattr(self.path, varname)) or (not isinstance(getattr(self.path, varname), np.ndarray)) or (getattr(self.path, varname).shape != (par.T, 1)):
                    setattr(self.path, varname, np.zeros((par.T, 1)))
        
        # c. Allocate shocks
        if hasattr(self, 'shocks'):
            for varname in self.shocks:
                if not hasattr(self.ss, varname):
                    setattr(self.ss, varname, getattr(par, varname, 0.0))
                if (not hasattr(self.path, varname)) or (not isinstance(getattr(self.path, varname), np.ndarray)) or (getattr(self.path, varname).shape != (par.T, 1)):
                    ss_val = getattr(self.ss, varname, getattr(par, varname, 0.0))
                    setattr(self.path, varname, np.full((par.T, 1), ss_val))

        # d. Allocate household inputs
        for varname in self.inputs_hh:
            if not hasattr(self.ss, varname):
                setattr(self.ss, varname, 0.0)
            if (not hasattr(self.path, varname)) or (not isinstance(getattr(self.path, varname), np.ndarray)) or (getattr(self.path, varname).shape != (par.T, 1)):
                setattr(self.path, varname, np.zeros((par.T, 1)))
        
        # e. Build varlist from block signatures (single source of truth)
        self.varlist = []

        # f. Scan all block function signatures and allocate every variable
        # that appears as a parameter (except par, ini, ss, and 'hh' block).
        import inspect, importlib
        for blockstr in self.blocks:
            if blockstr == 'hh':
                continue
            module_name, func_name = blockstr.split('.')
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            sig = inspect.signature(func)
            for pname in sig.parameters.keys():
                if pname in ('par', 'ini', 'ss'):
                    continue
                if pname not in self.varlist:
                    if not hasattr(self.ss, pname):
                        setattr(self.ss, pname, 0.0)
                    if (not hasattr(self.path, pname)) or (not isinstance(getattr(self.path, pname), np.ndarray)) or (getattr(self.path, pname).shape != (par.T, 1)):
                        setattr(self.path, pname, np.zeros((par.T, 1)))
                    self.varlist.append(pname)



    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids - create first
        self.create_grids()
        
        # b. solution - custom GE allocation (bypasses GEModelTools assertions)
        self.allocate_GE()
        
        # c. Re-ensure grids are correct (allocate_GE might reset them)
        if not hasattr(par, 'a_grid') or par.a_grid is None or len(par.a_grid) < 2 or not np.any(par.a_grid > 0):
            # Grid was overwritten, recreate it
            from consav.grids import equilogspace
            par.a_grid = equilogspace(0.0, par.a_max, par.Na)
            par.m_grid = par.a_grid.copy()

        # d. Allocate household policy arrays on regular grids
        Nfix = par.Nfix
        Nz = par.Nz
        Nh = 4  # 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner
        Na = par.Na
        self.sol.a = np.zeros((Nfix, Nz, Nh, Na))
        self.sol.c = np.zeros((Nfix, Nz, Nh, Na))
        self.sol.pr_choices = np.zeros((Nfix, Nz, Nh, 4, Na))  # prob of each of 4 choices
        self.sol.pr_urban = np.zeros((Nfix, Nz, Nh, Na))  # marginal urban prob (compatibility)
        
        # e. Allocate distribution arrays with housing dimension
        Nz = par.Nz
        Nh = 4  # 0=rural_renter, 1=urban_renter, 2=rural_owner, 3=urban_owner
        Nm = par.Na
        T = par.T
        
        self.ss.D = np.zeros((Nz, Nh, Nm))
        self.path.D = np.zeros((T, Nz, Nh, Nm))

        # f. OLG cohort containers (steady-state weights used by the fake-news
        # framework to aggregate age-specific responses; path-time cohort
        # weights are not stored because they are stationary in this model).
        J = par.J

        if (not hasattr(self.ss, 'cohort_weights')) or (not isinstance(self.ss.cohort_weights, np.ndarray)) or (self.ss.cohort_weights.shape != (J,)):
            self.ss.cohort_weights = np.ones(J) / J

        if not hasattr(self.ss, 'D_birth'):
            D_birth = np.zeros((Nz, Nh, Nm))
            z_ergodic = par.z_ergodic[0] if par.z_ergodic.ndim == 2 else par.z_ergodic
            newborn_urban_share = float(np.clip(getattr(par, 'newborn_urban_share', 0.0), 0.0, 1.0))
            # Newborns enter as renters (no prior ownership)
            D_birth[:, 0, 0] = z_ergodic * (1.0 - newborn_urban_share)  # rural_renter
            D_birth[:, 1, 0] = z_ergodic * newborn_urban_share            # urban_renter
            s = np.sum(D_birth)
            if s > 0.0:
                D_birth /= s
            self.ss.D_birth = D_birth
        
        # Note: outputs_hh = ['a', 'c', 'h_u', 'h_r'] map to A_hh, C_hh, H_u_hh, H_r_hh (aggregates)
        # They're allocated by allocate_GE() and computed by simulate_hh_ss()
        # Policy arrays (sol.a, sol.c, sol.pr_urban, sol.pr_rural) have shape (Nfix, Nz, Nh, Nm)

    







    # ==========================================================================
    # 3. Transition path methods
    # ==========================================================================
    
    def configure_open_economy(self, r_world_path=None, nfa_to_y_target=0.0):
        """Configure small-open-economy closure with exogenous world real rate.

        Args:
            r_world_path: optional length-T path for r_world. If None, holds at
                par.r_world for all t.
            nfa_to_y_target: reference NFA/Y target (diagnostic only; not
                enforced as a clearing equation in this configuration).
        """
        par = self.par
        ss = self.ss
        path = self.path

        par.open_economy = True
        par.nfa_to_y_target = float(nfa_to_y_target)

        r0 = float(getattr(par, 'r_world', getattr(ss, 'r', getattr(par, 'r', 0.03))))
        if r_world_path is None:
            par.r_world = r0
            ss.r_world = r0
            path.r_world[:, 0] = r0
        else:
            arr = np.asarray(r_world_path, dtype=float).ravel()
            if arr.size != par.T:
                raise ValueError(f"r_world_path must have length T={par.T}, got {arr.size}")
            # Last element is the terminal/long-run rate; first is t=0.
            par.r_world = float(arr[0])
            ss.r_world = float(arr[-1])
            path.r_world[:, 0] = arr

    # ==========================================================================
    # 4. Grid creation
    # ==========================================================================


    def create_grids(self):
        """ create grids for model """

        par = self.par

        # Asset grid - use equilogspace for better numerical stability
        from consav.grids import equilogspace
        par.a_grid = equilogspace(0.0, par.a_max, par.Na)

        # Validate grid
        if len(par.a_grid) < 2:
            raise ValueError(f"Asset grid too short: {len(par.a_grid)} points")

        grid_diff = np.diff(par.a_grid)
        if np.any(grid_diff <= 0):
            raise ValueError("Asset grid created by equilogspace is not strictly increasing")

        # Cash-on-hand grid (same as asset grid for now)
        par.m_grid = par.a_grid.copy()

        # Housing grids (discrete, but create for compatibility)
        par.h_u_grid = np.array([0.0, par.h_u])
        par.h_r_grid = np.array([0.0, par.h_r])

        # Productivity process: log-AR(1) discretized by Rouwenhorst
        rho = float(par.rho_z)
        sigma_z = float(par.sigma_psi)

        z_trans_2d = _rouwenhorst_transition(par.Nz, rho)

        if par.Nz == 1:
            z_grid = np.array([0.0])
        else:
            psi = sigma_z * np.sqrt(par.Nz - 1)
            z_grid = np.linspace(-psi, psi, par.Nz)

        z_ergodic_1d = _stationary_distribution(z_trans_2d)

        # Normalize levels so mean efficiency is one: E[exp(z)] = 1
        mean_eff = np.sum(z_ergodic_1d * np.exp(z_grid))
        if mean_eff > 0.0:
            z_grid = z_grid - np.log(mean_eff)

        par.z_grid = z_grid
        par.z_trans = np.tile(z_trans_2d[None, :, :], (par.Nfix, 1, 1))
        par.z_ergodic = np.tile(z_ergodic_1d[None, :], (par.Nfix, 1))
        par.z_trans_cumsum = np.cumsum(par.z_trans, axis=2)

        # Age-specific earnings profile from MAKRO model
        # (MAKRO_life_cycle_profiles.xlsx, column 'vW').
        # chi[j] = log(vW_j) - log(vW_18), normalized so chi[0] = 0 at age 16.
        # Ages 16 and 17 are assumed to earn the same as age 18 (first data point).
        import os as _os
        import openpyxl as _openpyxl
        _fname = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               'MAKRO_life_cycle_profiles.xlsx')
        _wb = _openpyxl.load_workbook(_fname)
        _ws = _wb.active
        _rows = list(_ws.iter_rows(values_only=True))
        _header = [str(c).strip() if c is not None else '' for c in _rows[0]]
        # Locate the 'Age' and 'vW' columns by name (file has additional series now)
        _i_age = _header.index('Age')
        _i_vw  = _header.index('vW')
        _age_data = {int(r[_i_age]): float(r[_i_vw]) for r in _rows[1:]
                     if r[_i_age] is not None}
        par.age_min = 16
        _vW0 = _age_data[18]
        _last_age = max(_age_data)
        _chi_list = []
        for _j in range(par.T):
            _age = par.age_min + _j
            if _age < 18:
                _vW = _age_data[18]
            elif _age > _last_age:
                _vW = _age_data[_last_age]
            else:
                _vW = _age_data[_age]
            _chi_list.append(np.log(_vW) - np.log(_vW0))
        par.chi = np.array(_chi_list)
    







