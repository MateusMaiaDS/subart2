# =============================================================================
# 04_3d_classification.R
# Comparison: subart (original) vs subart2 — Trivariate Probit Classification
# =============================================================================
# Scenario: d=3 binary responses, sim_class_mvn_friedman1 (Friedman functions)
#   n_train=300, n_test=50, p=5
#
# Plots:   Latent z boxplots by class for each outcome (train/test)
#          Predicted probability vs true probability (per outcome)
#          R[1,2], R[1,3], R[2,3] traceplots
#          Variable importance
# Metrics: Per-outcome accuracy + mean accuracy, timing speedup
# =============================================================================

HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n    <- 350
n_tr <- 300
p    <- 5
d    <- 3

sim  <- subart2::sim_class_mvn_friedman1(n = n, p = p, mvn_dim = d)

x_train    <- sim$x[1:n_tr, ]
x_test     <- sim$x[(n_tr + 1):n, ]
y_train    <- sim$y[1:n_tr, ]
y_test     <- sim$y[(n_tr + 1):n, ]
z_true_tr  <- sim$z_true[1:n_tr, ]
z_true_te  <- sim$z_true[(n_tr + 1):n, ]
p_true_tr  <- stats::pnorm(z_true_tr)
p_true_te  <- stats::pnorm(z_true_te)

cat("=== 04 Trivariate Classification (d=3) ===\n")
cat(sprintf("n_train=%d  n_test=%d  p=%d  d=%d\n", n_tr, n - n_tr, p, d))
cat(sprintf("Prevalence train: Y1=%.3f  Y2=%.3f  Y3=%.3f\n",
            mean(y_train[, 1]), mean(y_train[, 2]), mean(y_train[, 3])))

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

print_timing_table(time_og, time_new, scenario = "Trivariate Classification d=3")
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

# -- Panel 1: Latent z boxplots — train (2 rows × d cols)
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

# -- Panel 3: P(y=1) predicted vs true (train, per outcome)
par(mfrow = c(2, d), mar = c(4, 4, 3, 1))
for (j in 1:d) {
  scatter_pred_vs_true(stats::pnorm(fit_og$y_hat_mean[, j]),  p_true_tr[, j],
                       sprintf("subart  – P(Y%d=1)", j),  "steelblue")
  scatter_pred_vs_true(stats::pnorm(fit_new$y_hat_mean[, j]), p_true_tr[, j],
                       sprintf("subart2 – P(Y%d=1)", j), "darkorange")
}

# -- Panel 4: Correlation traceplots R[i,j] for i<j
#    d=3 has pairs (1,2),(1,3),(2,3) → 3 rows × 2 models = 6 panels
n_pairs <- d * (d - 1) / 2   # 3
par(mfrow = c(n_pairs, 2), mar = c(3, 4, 2.5, 1))
for (i in 1:(d - 1)) {
  for (j in (i + 1):d) {
    tr_og  <- fit_og$Sigma_post[i, j, ]
    tr_new <- fit_new$Sigma_post[i, j, ]
    lbl <- sprintf("R[%d,%d]", i, j)
    plot(tr_og,  type = "l", col = "steelblue",
         main = sprintf("subart  %s", lbl), xlab = "Iter", ylab = lbl, cex.main = 0.85)
    abline(v = 200, col = "firebrick", lty = 2)
    abline(h = mean(tr_og,  na.rm = TRUE), col = "navy", lty = 3)
    plot(tr_new, type = "l", col = "darkorange",
         main = sprintf("subart2 %s", lbl), xlab = "Iter", ylab = lbl, cex.main = 0.85)
    abline(v = 200, col = "firebrick", lty = 2)
    abline(h = mean(tr_new, na.rm = TRUE), col = "navy", lty = 3)
  }
}

# -- Panel 5: Variable importance
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og,  "subart – Var. Importance", d = d)
plot_varimportance(fit_new, "subart2 – Var. Importance", d = d)

par(op)
