# Compatibility Comparisons: `subart` (original) vs `subart2` (refactored)

This folder contains standalone R scripts that compare the original
[`subart`](https://github.com/MateusMaiaDS/subart) package with the refactored
[`subart2`](https://github.com/MateusMaiaDS/subart2) package across several
simulation scenarios.

## Prerequisites

Both packages must be installed:

```r
remotes::install_github("MateusMaiaDS/subart")
remotes::install_github("MateusMaiaDS/subart2")
remotes::install_github("nicholasjclark/mvnfast")  # dependency
```

Each script calls `setup_packages()` (defined in `00_helpers.R`) which
installs missing packages automatically.

## File overview

| Script | Scenario | d | MCMC path |
|--------|----------|---|-----------|
| `01_univariate_regression.R`           | Friedman-1 continuous response      | 1 | `cppsubart_univariate` |
| `02_univariate_classification.R`       | Probit binary classification         | 1 | `cppsubart_univariate_CLASS` |
| `03_2d_classification.R`               | Bivariate probit classification      | 2 | `cppsubart_CLASS` |
| `04_3d_classification.R`               | Trivariate probit classification     | 3 | `cppsubart_CLASS` |
| `05_missing_univariate_regression.R`   | Missing Y, univariate regression     | 1+2 | complete-case + `cppsubart_missing_2d` |
| `06_missing_univariate_classification.R` | Missing labels, univariate probit  | 1 | complete-case vs NA→0 |
| `07_missing_2d_classification.R`       | Missing labels, bivariate probit    | 2 | `cppsubart_CLASS` (NA→0) |
| `08_missing_3d_classification.R`       | Missing labels, trivariate probit   | 3 | `cppsubart_CLASS` (NA→0) |

`00_helpers.R` — shared utility functions (metrics, plots, timing). Sourced
by every script.

## Comparison dimensions

Each script reports and/or plots:

- **Computational time** — `system.time()` for both models; speedup ratio.
- **y_hat plots** — Predicted vs true scatter (train and test) per outcome.
- **Traceplots** — Posterior traces of Σ (regression: variances/covariances;
  classification d>1: off-diagonal correlations R[i,j]).
- **Accuracy / RMSE** — Printed to console after fitting.
- **Variable importance** — Mean split count per predictor (if `varimportance=TRUE`).

## Notes on missing-data scenarios

Scripts 05–08 test behaviour when `y_train` contains `NA` values.

- **d=1 regression** (script 05): neither package has a specialised univariate
  missing-data sampler. Two approaches are compared: complete-case analysis
  and a bivariate augmentation workaround that activates `cppsubart_missing_2d`.
- **d=1 classification** (script 06): `NA` labels are handled as class 0 by
  both packages; no principled missing-label imputation is performed.
- **d≥2 classification** (scripts 07–08): `NA` labels are replaced with 0
  before the standard probit CLASS sampler runs. Neither package has a
  dedicated missing-label probit sampler.

## Running

Source any script directly in RStudio, or from an R console:

```r
source("inst/compatibility_og_subart/01_univariate_regression.R")
```

MCMC settings used across all scripts (fast defaults for comparison):

| Parameter | Value |
|-----------|-------|
| `n_tree`  | 50    |
| `n_mcmc`  | 1000  |
| `n_burn`  | 200   |
