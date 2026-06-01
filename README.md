# Housing Price Disparities and Wealth Inequality in Denmark

Replication code for my master's thesis. The repository contains a
**heterogeneous-agent overlapping-generations (HA-OLG) housing model** of a small
open economy, calibrated to the Danish economy in the 1992 steady state and used
to study how regional house-price disparities shape the wealth distribution along
a 1992–2070 transition path.

---

## The model

Households are finitely lived ($J = 49$ ages) and face idiosyncratic, persistent
labor-productivity risk

$$\log z_{t+1} = \rho_z \log z_t + \psi_{t+1}, \qquad \psi \sim \mathcal{N}(0,\sigma_\psi^2),$$

discretized with Rouwenhorst into $N_z = 5$ states. Each period a household occupies
one of $N_h = 4$ **discrete housing/tenure states** — {rural renter, urban renter,
rural owner, urban owner} — and chooses consumption, savings, and next-period tenure.
Switching tenure or region incurs a transaction cost $\zeta$, and new mortgages are
capped by a loan-to-value ratio $\lambda^{\text{LTV}}$.

The production side is a small open economy: the real rate is pinned to the world
rate, $r_t = r^{\text{world}}_t$, the rental rate of capital satisfies
$r^K_t = r^{\text{world}}_t + \delta$, and net foreign assets
$\mathrm{NFA}_t = A^{hh}_t - K_t$ clear residually. Regional construction firms turn
investment into the urban/rural housing stocks, and a regulated rental sector links
rents $(f^u, f^r)$ to house prices $(q^u, q^r)$.

**General equilibrium** is a fixed point in the two house prices: the unknowns are the
regional housing stocks $(H^u, H^r)$ and the market-clearing targets are

$$\begin{pmatrix} \tilde q^{\,u}_t\big(H^{u,hh}_t(q^u_t,q^r_t)\big) - q^u_t \\
\tilde q^{\,r}_t\big(H^{r,hh}_t(q^u_t,q^r_t)\big) - q^r_t \end{pmatrix} = 0 .$$

### Solution method

- **Household problem:** DC-EGM (discrete-continuous endogenous grid method) following
  Iskhakov, Jørgensen, Rust & Schjerning (2017), with extreme-value taste shocks
  (smoothing parameter $\sigma$) over the discrete tenure choice.
- **Steady state:** `find_ss_prices()` solves the price fixed point above.
- **Transition & IRFs:** the dynamic equilibrium is solved along the DAG of blocks
  in [`blocks.py`](Code/Model/blocks.py); linearized impulse responses are built from
  sequence-space Jacobians.

The model is built on the [NumEconCopenhagen](https://github.com/NumEconCopenhagen)
`EconModelClass` / `GEModelTools` framework.

---

## Repository structure

```
Final thesis project/
├── Code/
│   ├── Model/                  # the HA-OLG housing model
│   │   ├── HousingModel.ipynb      # ← main entry point: solve, calibrate, run scenarios
│   │   ├── HANCHousingModel.py     # model class: parameters, grids, GE allocation
│   │   ├── household_problem.py    # DC-EGM household solver
│   │   ├── blocks.py               # DAG-ordered GE block equations
│   │   ├── steady_state.py         # steady-state solver & price fixed point
│   │   ├── transition.py           # transition path, Jacobians, linear IRFs
│   │   ├── simulation.py           # PE scenarios & wealth-distribution statistics
│   │   ├── calibration.py          # moment matching to 1992 targets
│   │   ├── build_cohort_weights.py # demographic cohort weights from Statistics Denmark
│   │   └── plots.py                # all thesis figures
│   └── Motivation/             # motivating empirical facts (Motivation.ipynb)
└── Data/
    ├── Moments/                # calibration targets (Statistics Denmark, ECB, MAKRO)
    ├── Figures/                # source data for thesis figures
    └── WID/                    # World Inequality Database series
```

---

## Getting started

### Requirements

- Python 3.12
- `numpy`, `scipy`, `numba`, `matplotlib`, `openpyxl`, `pandas`
- [`EconModel`](https://pypi.org/project/EconModel/) and
  [`GEModelTools`](https://github.com/NumEconCopenhagen/GEModelTools)
- `joblib` (optional — parallel scenario runs)
- `geopandas` (optional — only for the regional map figure)

```bash
pip install numpy scipy numba matplotlib openpyxl pandas EconModel GEModelTools
```

### Running

Open and run the main notebook top to bottom:

```bash
cd "Code/Model"
jupyter lab HousingModel.ipynb
```

It walks through, in order:

1. Demographic data and cohort weights (1992–2069 + padding).
2. Solving and calibrating the **1992 steady state** (Table 4.2 moments).
3. **Partial-equilibrium scenarios**, 1992–2070.
4. **Linearized IRFs** via sequence-space Jacobians.
5. Robustness checks and life-cycle appendices.

> **Note on data:** the large geospatial file `DK_INSPIRE_BBR.gpkg` (≈4.7 GB) used for
> the regional map in Figure 2.2 is intentionally excluded from the repository via `.gitignore`. The data is available on:
[`https://datafordeler.dk/dataoversigt/`](https://datafordeler.dk/dataoversigt/)

---

## Citation

If you use this code, please cite the thesis:

> Wohlert, F. (2026). *Housing Price Disparities and Wealth Inequality in Denmark.*
> Master's thesis, University of Copenhagen.

## References

- Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017).
  "The endogenous grid method for discrete-continuous dynamic choice models with
  (or without) taste shocks." *Quantitative Economics*, 8(2), 317–365.
