# Copilot Instructions for `subart2`

## Overview

`subart2` is an R package implementing **Multivariate Bayesian Additive Regression Trees (SuBART)** for multiple outcomes under the Seemingly Unrelated Regression (SUR) / Multivariate Normal (MVN) framework.  It is a high-performance C++ refactoring of the `subart` package.

## ⚠️ Canonical Methodology References

**All code changes, extensions, and agent work must be coherent with the methodology described in the two manuscripts located at:**

```
inst/research_papers/subart_submission/arXiv_main.tex   ← Main SuBART paper (AOAS)
inst/research_papers/subart_submission/supplementary.tex ← MCMC supplement
inst/research_papers/Nonparametric_CEA/main.tex          ← Applied CEA paper
```

A human-readable summary is in:
```
inst/research_papers/METHODOLOGY.md  ← Canonical methodology summary (read this first)
```

**Before making any change to the MCMC sampler, prior specifications, tree proposals, or model structure, read `METHODOLOGY.md` and the relevant LaTeX source. Do not deviate from the mathematics there without an explicit instruction from the user.**

---

## Repository Structure

```
R/
  subart.R              # Main wrapper: preprocessing, prior setup, C++ dispatch
  other_functions.R     # Helper metrics (RMSE, CRPS, ESS, coverage, etc.)
  simulation_functions.R# Simulation helpers (Friedman test functions)
src/
  subart.cpp            # MCMC sampler — general multivariate path (d > 2)
  subart_2d.cpp         # MCMC sampler — optimised bivariate path (d = 2)
  subart_missing.cpp    # MCMC sampler — missing Y (both d=2 and d>2 paths)
  subart_univariate.cpp # MCMC sampler — univariate fallback (d = 1)
  mcmc.cpp              # Shared MCMC utilities (tree moves, node sampling)
  tree.cpp              # Tree data structure
man/                    # Roxygen2-generated Rd documentation
tests/testthat/         # testthat tests
inst/research_papers/   # Overleaf manuscript sources (LaTeX) + METHODOLOGY.md
```

---

## Statistical Model

The model is: **Y_i = f(X_i) + ε_i**, where:

- **Y_i ∈ R^d** — vector of d responses for observation i
- **f(X_i) = Σ_{t=1}^{m} g(X_i; T_t, M_t)** — sum of m regression trees (per outcome)
- **ε_i ~ MVN(0, Σ)** — multivariate normal errors with d×d covariance matrix Σ

Each of the d outcomes shares the same tree structure but has its own set of leaf values.

### Key Priors (as in the paper)

| Component | Prior | R parameter |
|-----------|-------|-------------|
| Tree split probability at depth d | P = α · (1 + d)^(−β) | `alpha`, `beta` |
| Leaf node mean μ_{tℓ}^{(j)} | N(0, σ_μ²), σ_μ = 0.5/(κ√m) | `kappa`, `n_tree` |
| Σ (default) | Hierarchical half-t (Esser & Maia 2025) | `hier_prior_bool=TRUE` |
| Σ (alternative) | Inverse-Wishart(ν, A·I) | `hier_prior_bool=FALSE`, `nu`, `sigquant` |

### MCMC Tree Moves

Three Metropolis-Hastings proposals:
- **GROW** (prob ≈ 0.3): split a terminal node
- **PRUNE** (prob ≈ 0.3): collapse an internal node
- **CHANGE** (prob ≈ 0.4): change the splitting rule at an internal node

> **Critical note**: The verb threshold in `subart.cpp` must be `verb < 0.6` (not `6.0`) for CHANGE to be reachable. This bug was fixed in commit `0f01a41`.

### C++ Dispatch Logic (in `R/subart.R`)

| Condition | C++ function called |
|-----------|-------------------|
| d = 1 | `cppsubart_univariate()` |
| d = 2, no missing Y | `cppsubart_2d()` |
| d > 2, no missing Y | `cppsubart()` |
| d = 2, missing Y | `cppsubart_missing_2d()` |
| d > 2, missing Y | `cppsubart_missing()` |

---

## R Function API

```r
subart(
  x_train,            # data.frame: training covariates
  y_train,            # matrix: training responses (n × d); alias y_mat accepted
  x_test = NULL,      # data.frame: test covariates (optional)
  n_tree = 100,       # m: number of trees per outcome
  node_min_size = 5,  # minimum observations per leaf
  n_mcmc = 2000,      # total MCMC iterations
  n_burn = 500,       # burn-in iterations
  alpha = 0.95,       # tree prior base parameter α
  beta = 2,           # tree prior power parameter β
  nu = 3,             # degrees of freedom for Σ prior
  sigquant = 0.9,     # quantile for calibrating Σ prior scale
  kappa = 2,          # leaf mean prior width parameter κ
  numcut = 100L,      # number of candidate split points per predictor
  usequants = FALSE,  # use empirical quantiles for split points
  m = 20,             # correlation prior parameter (probit-MVN path)
  varimportance = TRUE,       # compute variable importance counts
  hier_prior_bool = TRUE,     # use hierarchical half-t prior for Σ
  specify_variables = NULL,   # restrict predictor subset (integer vector)
  diagnostic = TRUE           # compute ESS for Σ posterior samples
)
```

**Backward compatibility**: `y_mat` is accepted as an alias for `y_train`.

---

## Code Conventions

- **C++ style**: RcppArmadillo. Use `Rprintf()` not `printf()` for console output.
- **R documentation**: Roxygen2. All exported functions must have `@param` for every argument.
- **Tests**: `testthat` edition 3. Run via `devtools::test()` or `R CMD check`. 30 test assertions across 8 test cases covering univariate, bivariate, multivariate, x_test handling, missing Y, and input validation.
- **CI**: GitHub Actions at `.github/workflows/R-CMD-check.yml` — runs `R CMD check` (including tests) on ubuntu-latest + windows-latest on every PR to `main` and every push to `main`.
- **Branching**: feature branches → `main` via PR.

---

---

## Known Discrepancies between Papers and Current Code

These are documented in `inst/research_papers/METHODOLOGY.md §Known Discrepancies`. In brief:

1. **Probit suBART** is described in Paper 1 but the R wrapper currently returns `"Not available yet."` for classification outcomes.
2. **Default hyperparameters differ from paper recommendations**: papers use `nu=2`, `sigquant=0.95`; code defaults are `nu=3`, `sigquant=0.9`.
3. **Default chain length**: papers use `n_mcmc=5000`, `n_burn=1000`; code defaults are `n_mcmc=2000`, `n_burn=500`.
4. **`varimportance` is a stub**: papers define variable importance via split counts, but `R/subart.R` currently returns `NULL`.
5. **`specify_variables` not wired to C++**: the matrix is built but not passed through to any C++ routine yet.
6. **No LASSO fallback for residual-SD calibration**: Paper 1 recommends LASSO when `n < p`; code uses `naive_sigma()` only.

When any of these are resolved in code, remove the corresponding item from this list and from `METHODOLOGY.md`.

---

## Key References

1. **Maia & Esser et al. (2025)** — "SuBART: Multivariate Bayesian Additive Regression Trees". *Annals of Applied Statistics* (AOAS). → `inst/research_papers/subart_submission/arXiv_main.tex`
2. **Maia et al. (2025)** — "Nonparametric Cost-Effectiveness Analysis using SuBART". → `inst/research_papers/Nonparametric_CEA/main.tex`
3. Chipman, George & McCulloch (2010) — "BART: Bayesian Additive Regression Trees". *Annals of Applied Statistics*.
4. Esser & Maia (2025) — Hierarchical half-t prior for Σ (referenced in Paper 1).
