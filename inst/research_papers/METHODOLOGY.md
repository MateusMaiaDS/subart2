**This file is the canonical methodology reference for subart2. All code changes, extensions, and agent instructions must be consistent with this document.**

# subart2 Methodology Reference

## Source documents

This reference synthesizes:

- **Paper 1:** `inst\research_papers\subart_submission\arXiv_main.tex`
- **Paper 1 Supplement:** `inst\research_papers\subart_submission\supplementary.tex`
- **Paper 2:** `inst\research_papers\Nonparametric_CEA\main.tex`
- **Implementation cross-check:** `R\subart.R`

Throughout, let `i = 1, ..., n` index observations and let `j = 1, ..., d` index outcomes.

---

## 1. Model Overview

**SuBART** (“seemingly unrelated BART”) is a multivariate extension of Bayesian additive regression trees under the **seemingly unrelated regression (SUR)** framework.

For each observation,

`Y_i = (Y_i^(1), ..., Y_i^(d))^T in R^d`

with continuous-outcome likelihood

`Y_i | X_i ~ MVN_d(f(X_i), Sigma)`.

Hence suBART is a **nonparametric SUR model**: it keeps the multivariate normal error structure of SUR, but replaces each linear predictor by a sum of regression trees. Its defining methodological feature is:

- each outcome gets its **own ensemble of trees**, and
- dependence across outcomes is handled through the residual covariance matrix `Sigma`.

This is the main distinction from mvBART-style models that force all outcomes to share the same tree structure.

---

## 2. The Statistical Model

For continuous outcomes, Paper 1 specifies

`Y_i = f(X_i) + epsilon_i`, with `epsilon_i ~ MVN_d(0, Sigma)`.

Equivalently,

`Y_i^(j) = E[Y_i^(j) | X_i] + epsilon_i^(j)`.

The regression function is outcome-specific:

`f(X_i) = (f_1(X_i), ..., f_d(X_i))^T`,

where

`f_j(X_i) = sum_(t=1)^m g(X_i, T_t^(j), M_t^(j))`.

Here:

- `T_t^(j)` is the `t`-th binary tree for outcome `j`,
- `M_t^(j) = (mu_(t1)^(j), ..., mu_(t h_t^(j))^(j))` are that tree’s leaf parameters,
- `g(X_i, T_t^(j), M_t^(j))` returns the leaf mean corresponding to the region of predictor space containing `X_i`.

Thus:

- each outcome `j` receives **`m` trees**,
- the full multivariate model contains **`m x d` trees**,
- tree structures are independent across outcomes a priori.

This yields a joint model that allows different outcomes to depend on different covariates and different interactions, while still sharing a common residual dependence structure through `Sigma`.

---

## 3. Prior Specifications

### 3.1 Tree-structure prior

For any node at depth `gamma`, the probability that the node splits is

`P(split at depth gamma) = alpha * (1 + gamma)^(-beta)`.

Default values recommended in Paper 1 are `alpha = 0.95` and `beta = 2`, favoring shallow trees.

### 3.2 Leaf-node prior

After scaling each continuous outcome to `[-0.5, 0.5]`, the leaf parameters satisfy

`mu_(t ell)^(j) | T_t^(j) ~ N(0, sigma_mu,j^2)`

with

`sigma_mu,j = 1 / (2 * kappa * sqrt(m)) = 0.5 / (kappa * sqrt(m))`.

Default `kappa = 2` gives approximately 95% prior mass to the event

`E[Y_i^(j) | X_i] in [-0.5, 0.5]`.

The `1 / sqrt(m)` scaling is the usual BART shrinkage that regularizes each tree’s contribution as the ensemble size grows.

### 3.3 Prior on `Sigma`: default hierarchical half-`t`

Paper 1’s default prior (`hier_prior_bool = TRUE`) is the hierarchical covariance prior of Esser & Maia (2025), adapted from Huang et al. (2013):

`Sigma | a_1, ..., a_d ~ Inv-Wishart_d(nu + d - 1, S_0)`

`a_j ~ Inv-Gamma(1/2, 1 / A_j^2)`

with

`S_0 = 2 * nu * diag(1 / a_1, ..., 1 / a_d)`.

Important implied marginals:

- `sigma_j ~ Half-t(nu, A_j)` for each marginal standard deviation,
- `pi(rho_jk) proportional to (1 - rho_jk^2)^(nu/2 - 1)` for each correlation.

Consequences:

- `nu = 2` implies a **uniform prior on correlations**,
- larger `nu` shrinks correlations toward 0,
- `nu = 1` heavily favors extreme correlations and was found numerically unstable in Paper 1.

### 3.4 Calibration of `A_j` via `nu` and `sigquant`

For each outcome, let `hat(sigma_j)` be a rough overestimate of the residual SD. Paper 1 proposes obtaining it by least-squares regression, or by LASSO if `n < p`.

Then `A_j` is chosen so that

`P(sigma_j < hat(sigma_j)) = alpha_sigma`,

where `alpha_sigma` is the user-level calibration quantile (`sigquant` in code). Under the half-`t` prior this means solving

`alpha_sigma = integral_0^(hat(sigma_j)) Half-t_pdf(x; nu, A_j) dx`.

The papers recommend `nu = 2` and typically `alpha_sigma = 0.95`.

### 3.5 Alternative prior on `Sigma`

An alternative path is an **inverse-Wishart prior** (`hier_prior_bool = FALSE`). Paper 1 discusses this as computationally convenient but methodologically less desirable because it couples variances and correlations too strongly and can bias correlations toward 0 when variances are small.

### 3.6 Probit suBART prior

For binary outcomes, latent variables `Z_i` are introduced and the same tree prior is used. The leaf prior becomes

`mu_(t ell)^(j) ~ N(0, sigma_mu,j^2)`, with `sigma_mu,j = q_z / (kappa * sqrt(m))`.

Paper 1 recommends `q_z = 3`, implying approximately 95% prior mass on

`P(Y_i^(j)=1 | X_i) in [Phi(-3), Phi(3)] = [0.0013, 0.9987]`.

---

## 4. MCMC Algorithm

### 4.1 Backfitting structure

Posterior inference uses a Metropolis-within-Gibbs sampler. For each outcome `j` and tree `t`, define the tree-specific partial residuals

`r_(ti)^(j) = Y_i^(j) - sum_(k != t) g(X_i, T_k^(j), M_k^(j))`.

Conditionally on all other outcomes, the Gaussian SUR structure gives

`Y_i^(j) | Y_i^(-j), Theta ~ N(m_i^(j), v^(j))`

with

`m_i^(j) = hat(Y)_i^(j) + Sigma_(j,-j) * Sigma_(-j,-j)^(-1) * (Y_i^(-j) - hat(Y)_i^(-j))`

`v^(j) = Sigma_jj - Sigma_(j,-j) * Sigma_(-j,-j)^(-1) * Sigma_(-j,j)`.

This reduces multivariate updating to BART-style univariate updates with outcome-specific offsets induced by cross-outcome dependence.

### 4.2 Tree moves

Tree structures are updated by Metropolis-Hastings using the integrated likelihood over leaf means. The move set is the standard BART trio:

- **GROW**: split a terminal node,
- **PRUNE**: collapse a pair of terminal children,
- **CHANGE**: change an existing split rule.

The manuscripts refer to standard BART MH updates; the current `subart2` implementation uses proposal probabilities:

- `P(GROW) = 0.3`
- `P(PRUNE) = 0.3`
- `P(CHANGE) = 0.4`

(C++ comments and branching in `src\subart.cpp` / `src\subart_2d.cpp`.)

### 4.3 Leaf updates

Given a fixed tree, each leaf mean has a conjugate Gaussian update. Paper 1 gives

`mu_(t ell)^(j) | Theta ~ N( mean_(t ell)^(j), var_(t ell)^(j) )`

where, for observations in that leaf,

`u_i^(j) = Sigma_(j,-j) * Sigma_(-j,-j)^(-1) * (Y_i^(-j) - hat(Y)_i^(-j))`,

`mean_(t ell)^(j) = [sigma_mu,j^2 / (v^(j) + n_(t ell)^(j) * sigma_mu,j^2)] * [sum_i r_i^(j) - sum_i u_i^(j)]`,

`var_(t ell)^(j) = [v^(j) * sigma_mu,j^2] / [v^(j) + n_(t ell)^(j) * sigma_mu,j^2]`.

### 4.4 Covariance updates for continuous suBART

Under the hierarchical half-`t` prior, Gibbs updates are available:

`a_j | Theta ~ Inv-Gamma((nu + d)/2, 1/A_j^2 + nu * (Sigma^(-1))_jj)`

and

`Sigma | Theta ~ Inv-Wishart_d(nu + d - 1 + n, S_0 + S)`

with

`S = sum_i (Y_i - hat(Y_i)) (Y_i - hat(Y_i))^T`.

### 4.5 Missing-outcome imputation

Under MAR, missing outcomes can be imputed inside the Gibbs sampler from their conditional Gaussian distributions implied by the SUR model. Paper 1 describes this as a natural Gibbs extension using the same conditional normal formula above; Paper 2 presents it as available functionality; current code implements continuous missing-outcome paths via dedicated C++ dispatchers.

### 4.6 Probit suBART sampler

For binary outcomes, introduce latent `Z_i` with

`Y_i^(j) = 1{ Z_i^(j) > 0 }`.

Then:

1. update trees/leaves using `Z` in place of `Y`,
2. sample each `Z_i^(j)` from a truncated normal,
3. update the correlation matrix using the parameter-expanded MH scheme of Zhang et al. (2020):
   `W = D^(1/2) Sigma D^(1/2)`,
   `W^(k+1) | W^(k) ~ Inv-Wishart_d(nu_prop, nu_prop * W^(k))`.

Paper 1 uses longer chains for probit suBART because this PX-MH step mixes more slowly.

---

## 5. Implementation Details

#### 5.1 Outcome scaling

For continuous outcomes, each response column is scaled to `[-0.5, 0.5]` before fitting. Posterior means and covariance draws are then transformed back to the original scale.

#### 5.2 Outcome-specific ensembles

The methodological design is always:

- `d = 1`: univariate BART fallback,
- `d = 2`: bivariate suBART with an optimized computational path,
- `d > 2`: general multivariate suBART.

#### 5.3 Variable importance

Variable importance is defined in the BART sense: count how often each predictor is used in splitting rules across the outcome-specific tree ensembles, then summarize those counts over posterior draws. In suBART this must be interpreted **per outcome**, not globally, because predictors may matter differently for costs and effects.

#### 5.4 Outcome-specific predictor sets

Paper 1 allows different subsets of predictors to be available to different outcomes. This is conceptually important: unlike mvBART, suBART does not require common predictors or common tree structures across outcomes.

#### 5.5 Typical MCMC settings in the papers

For continuous suBART, the papers typically use:

- `N_MCMC = 5000`
- `N_burn = 1000`

For probit suBART simulation work:

- `N_MCMC = 10000`
- `N_burn = 2000`
- `nu_prop = n_train / 10` for `d = 2`
- `nu_prop = n_train / 2` for `d = 3`

---

## 6. The CEA Application

The motivating application is **observational cost-effectiveness analysis (CEA)** with two joint outcomes:

- `c_i`: healthcare costs,
- `q_i`: health-related quality of life / QALYs.

The inferential problem is to estimate treatment effects on both outcomes while adjusting for confounding and preserving their joint uncertainty.

#### 6.1 Causal estimands

Under ignorability,

`Delta_c = (1/n) * sum_i [ E(c_i | t_i = 1, x_i) - E(c_i | t_i = 0, x_i) ]`

`Delta_q = (1/n) * sum_i [ E(q_i | t_i = 1, x_i) - E(q_i | t_i = 0, x_i) ]`.

These are the mixed average treatment effects (MATEs) used in Paper 1.

#### 6.2 Why joint modeling matters in CEA

CEA decisions depend on the **incremental net benefit**

`INB_lambda = lambda * Delta_q - Delta_c`.

Hence

`Var(INB_lambda) = lambda^2 Var(Delta_q) + Var(Delta_c) - 2 lambda Cov(Delta_q, Delta_c)`.

So the CE plane, credible intervals, and the probability of cost-effectiveness all depend on the joint distribution of `(Delta_c, Delta_q)`. Fitting separate models for costs and QALYs discards that covariance information.

#### 6.3 Propensity-score augmented suBART

For observational data, the recommended workflow is **ps-suBART**:

1. estimate propensity scores with probit BART,
2. append the estimated propensity score as an additional predictor,
3. fit suBART jointly to `(c_i, q_i)`.

This is an **S-learner / g-computation** workflow: duplicate each patient in the test set, once with `t = 0` and once with `t = 1`, predict both potential outcomes, and average the resulting contrasts.

#### 6.4 CE plane, CEAC, and EVPI

From posterior draws `s = 1, ..., S` of `(Delta_c^(s), Delta_q^(s))`, compute

`INB_lambda^(s) = lambda * Delta_q^(s) - Delta_c^(s)`.

Then:

- **CE plane:** plot draws of `(Delta_q^(s), Delta_c^(s))`.
- **CEAC:**
  `P(INB_lambda > 0 | data) approx (1/S) * sum_s 1{INB_lambda^(s) > 0}`.
- **EVPI (two-option incremental form):** although not written out explicitly in the manuscripts, the standard Bayesian CEA quantity compatible with this framework is
  `EVPI(lambda) = E[max(INB_lambda, 0)] - max(E[INB_lambda], 0)`
  which is estimated from posterior INB draws by Monte Carlo averaging.

Paper 2 also gives a normal-approximation CEAC formula when only pooled mean/covariance summaries of `(Delta_c, Delta_q)` are available.

---

## 7. Key Parameters Reference Table

| R parameter | Symbol | Description | Paper reference |
|---|---|---|---|
| `x_train` | `X` | Training covariate matrix / data frame | Paper 1, SUR and suBART model sections; Paper 2 §Observational studies and model specification |
| `y_train` | `Y` | Training response matrix with rows `Y_i in R^d` | Paper 1 §suBART for continuous outcomes |
| `n_tree` | `m` | Number of trees **per outcome** | Paper 1 §suBART for continuous outcomes; Paper 2 §BART / §suBART |
| `n_mcmc` | `N_MCMC` | Total MCMC iterations | Paper 1 §Simulation settings / convergence; Paper 2 §Bayesian estimation and MCMC |
| `n_burn` | `N_burn` | Burn-in iterations discarded | Paper 1 §Simulation settings / convergence; Paper 2 §Bayesian estimation and MCMC |
| `alpha` | `alpha` | Tree-depth split prior parameter | Paper 1 §review of BART / §suBART for continuous outcomes |
| `beta` | `beta` | Tree-depth decay parameter | Paper 1 §review of BART / §suBART for continuous outcomes |
| `nu` | `nu` | Covariance-prior degrees of freedom; controls correlation prior concentration | Paper 1 §suBART for continuous outcomes / §Probit suBART |
| `sigquant` | `alpha_sigma` | Prior calibration target `P(sigma_j < hat(sigma_j))` | Paper 1 §suBART for continuous outcomes |
| `kappa` | `kappa` | Leaf-shrinkage hyperparameter in `sigma_mu,j = 0.5/(kappa sqrt(m))` | Paper 1 Eq. for leaf prior |
| `numcut` | — | Number of candidate cutpoints per predictor | BART implementation detail; consistent with standard BART software practice |
| `usequants` | — | Whether cutpoints are quantile-based instead of equally spaced | Implementation detail |
| `m` | — (legacy / probit-specific code hyperparameter) | Legacy probit-related parameter in current R API; **not** the tree count | Code-level API only; not part of continuous suBART notation |
| `varimportance` | — | Request split-count variable-importance summaries | Paper 1 Appendix on variable importance |
| `hier_prior_bool` | — | Toggle hierarchical half-`t` prior (`TRUE`) vs inverse-Wishart path (`FALSE`) for `Sigma` | Paper 1 covariance-prior discussion |
| `specify_variables` | outcome-specific predictor sets | Restrict which predictors each outcome-specific ensemble may use | Paper 1 §suBART for continuous outcomes |
| `diagnostic` | — | Compute ESS-based diagnostics for posterior covariance draws | Implementation / Paper 2 traceplot discussion |

---

## 8. Code-Paper Mapping

| Function | Meaning |
|---|---|
| `cppsubart_univariate` | Univariate BART fallback (`d = 1`) |
| `cppsubart_2d` | Optimized bivariate continuous suBART (`d = 2`) |
| `cppsubart` | General continuous multivariate suBART (`d > 2`) |
| `cppsubart_missing_2d` | Bivariate continuous suBART with missing outcomes |
| `cppsubart_missing` | General continuous multivariate suBART with missing outcomes |

Operationally, these implement the same methodology described in Paper 1, but with separate dispatch paths for computational efficiency and missing-data handling.

---

## Known Discrepancies / Implementation Notes

1. **Binary/probit suBART is described in both papers, but the current `R/subart.R` wrapper stops with `"Not available yet."` for classification outcomes.** The papers describe a full probit suBART model; the current top-level R interface does not expose it.
2. **Default prior hyperparameters differ from the papers’ recommendations.** The papers recommend `nu = 2` and typically `sigquant = 0.95`; current code defaults are `nu = 3` and `sigquant = 0.9`.
3. **Default chain lengths differ from the paper defaults.** The papers typically use `n_mcmc = 5000`, `n_burn = 1000` for continuous analyses, whereas current code defaults are `2000` and `500`.
4. **Residual-SD calibration differs from Paper 1.** Paper 1 recommends least-squares `hat(sigma_j)` with LASSO fallback when `n < p`; current code uses `naive_sigma()` only when `n > d` and otherwise falls back to sample SD, with no LASSO path.
5. **`varimportance` is currently a stub.** The papers define variable importance through split counts, but `R/subart.R` currently sets `var_importance <- NULL` and returns no computed importance values.
6. **`specify_variables` is validated but not wired through to C++.** `R/subart.R` builds `sv_matrix`, but the current dispatch calls do not pass it to any C++ routine, so outcome-specific predictor restrictions are not presently enforced.
7. **Missing-outcome handling is more advanced in code than in the main AOAS application text.** Paper 1’s TTCM application used a single externally imputed dataset and discussed in-sampler missing-outcome imputation as a natural extension; current code includes dedicated continuous missing-outcome dispatchers.
8. **Current API naming differs from manuscript/tutorial notation.** The papers and older examples often use `y_mat`; current function signature uses `y_train` and internally aliases `y_mat <- y_train` for backward compatibility.
9. **Current implementation additionally scales predictors to `[0, 1]`.** The papers emphasize response scaling to `[-0.5, 0.5]`; the code also normalizes covariates as an implementation detail.
