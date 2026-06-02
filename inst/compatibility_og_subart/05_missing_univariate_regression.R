# =============================================================================
# 05_missing_univariate_regression.R
# Comparison: subart (original) vs subart2 — Missing Data, Univariate Regression
# =============================================================================
# Scenario: Friedman-1 single response with ~15% of y_train set to NA.
#
# Important note on missing data and d=1:
#   Neither subart nor subart2 implements a specialised Gibbs imputation step
#   for the univariate case (d=1). Both packages skip the missing-data C++
#   path and the NA entries are simply ignored by the C++ sampler (or produce
#   NaN), meaning the fit is effectively a complete-case analysis.
#
#   For a principled treatment of missing Y in a single-response model,
#   two approaches are demonstrated here:
#     (A) Complete-case analysis — drop rows with NA before fitting.
#     (B) Multivariate workaround — augment Y to d=2, introducing a noise
#         auxiliary outcome for the non-NA rows; the missing-data C++ path
#         (cppsubart_missing_2d) then imputes the NA entries properly.
#
# Plots:   Scatter (predicted vs true) on observed and held-out NA positions
#          sigma^2 traceplot (approach A)
#          Imputation quality for approach B
# Metrics: RMSE on observed, RMSE on imputed (approach B only), timing
# =============================================================================

HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n    <- 250
p    <- 5
n_tr <- 200
na_frac <- 0.15

x_all      <- data.frame(matrix(stats::runif(n * p), ncol = p,
                                 dimnames = list(NULL, paste0("X", 1:p))))
y_true_all <- with(x_all,
  10 * sin(pi * X1 * X2) + 20 * (X3 - 0.5)^2 + 10 * X4 + 5 * X5)
y_all      <- matrix(y_true_all + stats::rnorm(n, 0, 1), ncol = 1)

x_train      <- x_all[1:n_tr, ]
x_test       <- x_all[(n_tr + 1):n, ]
y_train_full <- y_all[1:n_tr, , drop = FALSE]
y_test       <- y_all[(n_tr + 1):n, , drop = FALSE]
y_true_train <- y_true_all[1:n_tr]
y_true_test  <- y_true_all[(n_tr + 1):n]

# Introduce NA
set.seed(123)
na_idx        <- sample(n_tr, floor(na_frac * n_tr))
y_train_miss  <- y_train_full
y_train_miss[na_idx, ] <- NA
obs_idx       <- setdiff(seq_len(n_tr), na_idx)

cat("=== 05 Missing Univariate Regression ===\n")
cat(sprintf("n_train=%d  NA=%d (%.0f%%)  n_test=%d  p=%d\n",
            n_tr, length(na_idx), 100 * na_frac, n - n_tr, p))
cat("NOTE: d=1 missing not natively supported; running complete-case (A)\n")
cat("      and bivariate augmentation (B) for comparison.\n\n")

# ---- Approach A: complete-case analysis -------------------------------------

x_train_cc <- x_train[obs_idx, ]
y_train_cc <- y_train_miss[obs_idx, , drop = FALSE]

MCMC_A <- list(n_tree = 50, n_mcmc = 1000, n_burn = 200,
               varimportance = TRUE, diagnostic = FALSE)

cat("--- Approach A: Complete-case analysis ---\n")

time_og_A <- system.time({
  fit_og_A <- do.call(subart::subart,
                      c(list(x_train = x_train_cc, y_train = y_train_cc,
                             x_test  = x_test), MCMC_A))
})
time_new_A <- system.time({
  fit_new_A <- do.call(subart2::subart,
                       c(list(x_train = x_train_cc, y_train = y_train_cc,
                              x_test  = x_test), MCMC_A))
})

print_timing_table(time_og_A, time_new_A, scenario = "Approach A (complete-case)")
print_rmse_table(fit_og_A, fit_new_A, y_true_train[obs_idx], y_true_test)

# ---- Approach B: bivariate augmentation with missing-data C++ path ----------
# Augment Y with a dummy second outcome (fully observed) so that the
# cppsubart_missing_2d sampler handles the NAs in column 1 properly.
# The dummy column is a noisy version of the training mean to be uninformative.

cat("--- Approach B: Bivariate augmentation (proper imputation via d=2) ---\n")

y_aux        <- matrix(stats::rnorm(n_tr, mean(y_train_full, na.rm = TRUE),
                                    stats::sd(y_train_full, na.rm = TRUE)),
                       ncol = 1)
y_train_aug  <- cbind(y_train_miss, y_aux)   # col 1 has NAs, col 2 is observed

time_og_B <- system.time({
  fit_og_B <- do.call(subart::subart,
                      c(list(x_train = x_train, y_train = y_train_aug,
                             x_test  = x_test), MCMC_A))
})
time_new_B <- system.time({
  fit_new_B <- do.call(subart2::subart,
                       c(list(x_train = x_train, y_train = y_train_aug,
                              x_test  = x_test), MCMC_A))
})

print_timing_table(time_og_B, time_new_B, scenario = "Approach B (bivariate augmentation)")

# RMSE for outcome 1 only (the one with NAs)
cat("RMSE outcome 1:\n")
cat(sprintf("  Train (obs)  subart: %.4f  subart2: %.4f\n",
            rmse(fit_og_B$y_hat_mean[obs_idx, 1], y_true_train[obs_idx]),
            rmse(fit_new_B$y_hat_mean[obs_idx, 1], y_true_train[obs_idx])))
cat(sprintf("  Train (NA)   subart: %.4f  subart2: %.4f\n",
            rmse(fit_og_B$y_hat_mean[na_idx, 1], y_true_train[na_idx]),
            rmse(fit_new_B$y_hat_mean[na_idx, 1], y_true_train[na_idx])))
cat(sprintf("  Test         subart: %.4f  subart2: %.4f\n",
            rmse(fit_og_B$y_hat_test_mean[, 1], y_true_test),
            rmse(fit_new_B$y_hat_test_mean[, 1], y_true_test)))
cat("\n")

# ---- Plots ------------------------------------------------------------------

op <- par(no.readonly = TRUE)
on.exit(par(op))

# -- Panel 1: Approach A scatter — train observed, test
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
scatter_pred_vs_true(fit_og_A$y_hat_mean,       y_true_train[obs_idx],
                     "subart A – Train (obs)",  "steelblue")
scatter_pred_vs_true(fit_new_A$y_hat_mean,      y_true_train[obs_idx],
                     "subart2 A – Train (obs)", "darkorange")
scatter_pred_vs_true(fit_og_A$y_hat_test_mean,  y_true_test,
                     "subart A – Test",         "steelblue")
scatter_pred_vs_true(fit_new_A$y_hat_test_mean, y_true_test,
                     "subart2 A – Test",        "darkorange")

# -- Panel 2: Approach B scatter — train observed, NA imputed, test (outcome 1)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
scatter_pred_vs_true(fit_og_B$y_hat_mean[obs_idx, 1],  y_true_train[obs_idx],
                     "subart B – Y1 Train (obs)",   "steelblue")
scatter_pred_vs_true(fit_new_B$y_hat_mean[obs_idx, 1], y_true_train[obs_idx],
                     "subart2 B – Y1 Train (obs)",  "darkorange")
scatter_pred_vs_true(fit_og_B$y_hat_mean[na_idx, 1],   y_true_train[na_idx],
                     "subart B – Y1 Imputed (NA)",  "steelblue")
scatter_pred_vs_true(fit_new_B$y_hat_mean[na_idx, 1],  y_true_train[na_idx],
                     "subart2 B – Y1 Imputed (NA)", "darkorange")
scatter_pred_vs_true(fit_og_B$y_hat_test_mean[, 1],    y_true_test,
                     "subart B – Y1 Test",          "steelblue")
scatter_pred_vs_true(fit_new_B$y_hat_test_mean[, 1],   y_true_test,
                     "subart2 B – Y1 Test",         "darkorange")

# -- Panel 3: sigma^2 traces (Approach A)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
traceplot_sigma_1d(fit_og_A$Sigma_post,  "subart A  – sigma^2",  "steelblue",  n_burn = 200)
traceplot_sigma_1d(fit_new_A$Sigma_post, "subart2 A – sigma^2", "darkorange", n_burn = 200)

# -- Panel 4: Sigma[1,1] traces (Approach B, outcome 1 variance)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
if (!is.null(fit_og_B$Sigma_post))
  traceplot_sigma_nd(fit_og_B$Sigma_post, d = 2, main_prefix = "subart B",
                     col = "steelblue",  n_burn = 200, off_diag_only = FALSE)
if (!is.null(fit_new_B$Sigma_post))
  traceplot_sigma_nd(fit_new_B$Sigma_post, d = 2, main_prefix = "subart2 B",
                     col = "darkorange", n_burn = 200, off_diag_only = FALSE)

par(op)
