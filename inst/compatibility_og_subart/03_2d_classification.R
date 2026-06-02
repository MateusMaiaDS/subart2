# =============================================================================
# 03_2d_classification.R
# Comparison: subart (original) vs subart2 — Bivariate Probit Classification
# =============================================================================
# Scenario: d=2 binary responses, sim_class_mvn_friedman1 (Friedman functions)
#   n_train=250, n_test=50, p=5
#
# Plots:   Latent z boxplots by class for each outcome (train/test)
#          Predicted probability vs true probability (per outcome)
#          R[1,2] (off-diagonal correlation) traceplot
#          Variable importance
# Metrics: Per-outcome accuracy + mean accuracy, timing speedup
# =============================================================================

HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n    <- 300
n_tr <- 250
p    <- 5
d    <- 2

sim  <- subart2::sim_class_mvn_friedman1(n = n, p = p, mvn_dim = d)

x_train    <- sim$x[1:n_tr, ]
x_test     <- sim$x[(n_tr + 1):n, ]
y_train    <- sim$y[1:n_tr, ]
y_test     <- sim$y[(n_tr + 1):n, ]
z_true_tr  <- sim$z_true[1:n_tr, ]
z_true_te  <- sim$z_true[(n_tr + 1):n, ]
p_true_tr  <- stats::pnorm(z_true_tr)
p_true_te  <- stats::pnorm(z_true_te)

cat("=== 03 Bivariate Classification (d=2) ===\n")
cat(sprintf("n_train=%d  n_test=%d  p=%d  d=%d\n", n_tr, n - n_tr, p, d))
cat(sprintf("Prevalence train: Y1=%.3f  Y2=%.3f\n",
            mean(y_train[, 1]), mean(y_train[, 2])))

# ---- Fit models -------------------------------------------------------------

MCMC_SETTINGS <- list(n_tree = 50, n_mcmc = 1000, n_burn = 200,
                      varimportance = TRUE, diagnostic = FALSE)

time_og <- system.time({
  fit_og <- do.call(subart::subart,
                    c(list(x_train = x_train, y_train = y_train,
                           x_test  = x_test),
                      MCMC_SETTINGS))
})

time_new <- system.time({
  fit_new <- do.call(subart2::subart,
                     c(list(x_train = x_train, y_train = y_train,
                            x_test  = x_test),
                       MCMC_SETTINGS))
})

# ---- Console output ---------------------------------------------------------

print_timing_table(time_og, time_new, scenario = "Bivariate Classification d=2")
print_accuracy_table(fit_og, fit_new, y_train, y_test)
cat("Per-outcome accuracy (subart / subart2):\n")
for (j in 1:d) {
  cat(sprintf("  Y%d  Train: %.4f / %.4f   Test: %.4f / %.4f\n", j,
              accuracy(fit_og$y_hat_mean_class[, j], y_train[, j]),
              accuracy(fit_new$y_hat_mean_class[, j], y_train[, j]),
              accuracy(fit_og$y_hat_test_mean_class[, j], y_test[, j]),
              accuracy(fit_new$y_hat_test_mean_class[, j], y_test[, j])))
}
cat("\n")

# ---- Plots ------------------------------------------------------------------

op <- par(no.readonly = TRUE)
on.exit(par(op))

# -- Panel 1: Latent z boxplots — train
par(mfrow = c(2, d), mar = c(4, 4, 3, 1))
for (j in 1:d) {
  plot_latent_by_class(fit_og$y_hat_mean,  y_train, j,
                       sprintf("subart  – Train Y%d", j),  "steelblue")
  plot_latent_by_class(fit_new$y_hat_mean, y_train, j,
                       sprintf("subart2 – Train Y%d", j), "darkorange")
}

# -- Panel 2: Latent z boxplots — test
par(mfrow = c(2, d), mar = c(4, 4, 3, 1))
for (j in 1:d) {
  plot_latent_by_class(fit_og$y_hat_test_mean,  y_test, j,
                       sprintf("subart  – Test Y%d", j),  "steelblue")
  plot_latent_by_class(fit_new$y_hat_test_mean, y_test, j,
                       sprintf("subart2 – Test Y%d", j), "darkorange")
}

# -- Panel 3: Predicted probability vs true probability (train)
par(mfrow = c(2, d), mar = c(4, 4, 3, 1))
for (j in 1:d) {
  scatter_pred_vs_true(stats::pnorm(fit_og$y_hat_mean[, j]),  p_true_tr[, j],
                       sprintf("subart  – P(Y%d=1) Train", j),  "steelblue")
  scatter_pred_vs_true(stats::pnorm(fit_new$y_hat_mean[, j]), p_true_tr[, j],
                       sprintf("subart2 – P(Y%d=1) Train", j), "darkorange")
}

# -- Panel 4: R[1,2] traceplot (off-diagonal correlation)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
traceplot_sigma_nd(fit_og$Sigma_post,  d = d, main_prefix = "subart",
                   col = "steelblue",  n_burn = 200, off_diag_only = TRUE)
traceplot_sigma_nd(fit_new$Sigma_post, d = d, main_prefix = "subart2",
                   col = "darkorange", n_burn = 200, off_diag_only = TRUE)

# -- Panel 5: Variable importance
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og,  "subart – Var. Importance", d = d)
plot_varimportance(fit_new, "subart2 – Var. Importance", d = d)

par(op)
