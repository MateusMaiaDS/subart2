# =============================================================================
# 08_missing_3d_classification.R
# Comparison: subart (original) vs subart2 — Missing Labels, d=3 Probit
# =============================================================================
# Scenario: d=3 binary responses with ~15% of y_train labels set to NA
#   (independently per outcome), using sim_class_mvn_friedman1.
#   n_train=300, n_test=50, p=5
#
# Both packages replace NA labels with class 0 (no principled imputation).
# Accuracy is evaluated exclusively on the non-NA training observations.
#
# Plots:   Latent z boxplots per outcome (train observed subset, test)
#          Predicted probability vs true probability per outcome
#          R[1,2], R[1,3], R[2,3] traceplots
#          Accuracy bar chart
# Metrics: Per-outcome accuracy on observed labels, timing speedup
# =============================================================================

HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n       <- 350
n_tr    <- 300
p       <- 5
d       <- 3
na_frac <- 0.15

sim      <- subart2::sim_class_mvn_friedman1(n = n, p = p, mvn_dim = d)
x_all    <- sim$x
y_all    <- sim$y
z_true   <- sim$z_true
p_true   <- stats::pnorm(z_true)

x_train      <- x_all[1:n_tr, ]
x_test       <- x_all[(n_tr + 1):n, ]
y_train_full <- y_all[1:n_tr, ]
y_test       <- y_all[(n_tr + 1):n, ]
p_true_tr    <- p_true[1:n_tr, ]
p_true_te    <- p_true[(n_tr + 1):n, ]

set.seed(123)
y_train_miss <- y_train_full
obs_mask     <- matrix(TRUE, nrow = n_tr, ncol = d)
for (j in 1:d) {
  na_j  <- sample(n_tr, floor(na_frac * n_tr))
  y_train_miss[na_j, j] <- NA
  obs_mask[na_j, j]     <- FALSE
}

cat("=== 08 Missing Trivariate Classification (d=3) ===\n")
cat(sprintf("n_train=%d  NA per col ~%.0f%%  n_test=%d  p=%d\n",
            n_tr, 100 * na_frac, n - n_tr, p))
cat(sprintf("Observed labels: Y1=%d  Y2=%d  Y3=%d (of %d)\n",
            sum(obs_mask[, 1]), sum(obs_mask[, 2]), sum(obs_mask[, 3]), n_tr))

# ---- Fit models -------------------------------------------------------------

MCMC <- list(n_tree = 50, n_mcmc = 1000, n_burn = 200,
             varimportance = TRUE, diagnostic = FALSE)

time_og <- system.time({
  fit_og <- do.call(subart::subart,
                    c(list(x_train = x_train, y_train = y_train_miss,
                           x_test  = x_test), MCMC))
})
time_new <- system.time({
  fit_new <- do.call(subart2::subart,
                     c(list(x_train = x_train, y_train = y_train_miss,
                            x_test  = x_test), MCMC))
})

# ---- Console output ---------------------------------------------------------

print_timing_table(time_og, time_new, scenario = "Missing d=3 Classification")

cat("Accuracy on OBSERVED labels only (subart / subart2):\n")
for (j in 1:d) {
  obs_j <- obs_mask[, j]
  cat(sprintf("  Y%d  Train: %.4f / %.4f   Test: %.4f / %.4f\n", j,
              accuracy(fit_og$y_hat_mean_class[obs_j, j],  y_train_full[obs_j, j]),
              accuracy(fit_new$y_hat_mean_class[obs_j, j], y_train_full[obs_j, j]),
              accuracy(fit_og$y_hat_test_mean_class[, j],  y_test[, j]),
              accuracy(fit_new$y_hat_test_mean_class[, j], y_test[, j])))
}
cat("\n")

# ---- Plots ------------------------------------------------------------------

op <- par(no.readonly = TRUE)
on.exit(par(op))

# -- Panel 1: Latent z boxplots — train (observed labels per outcome)
par(mfrow = c(2, d), mar = c(4, 4, 3, 1))
for (j in 1:d) {
  obs_j <- obs_mask[, j]
  plot_latent_by_class(fit_og$y_hat_mean[obs_j, j], y_train_full[obs_j, j],
                       j = 1, sprintf("subart  – Train Y%d (obs)", j), "steelblue")
  plot_latent_by_class(fit_new$y_hat_mean[obs_j, j], y_train_full[obs_j, j],
                       j = 1, sprintf("subart2 – Train Y%d (obs)", j), "darkorange")
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
                       sprintf("subart  – P(Y%d=1) Train", j),  "steelblue")
  scatter_pred_vs_true(stats::pnorm(fit_new$y_hat_mean[, j]), p_true_tr[, j],
                       sprintf("subart2 – P(Y%d=1) Train", j), "darkorange")
}

# -- Panel 4: Correlation traceplots R[i,j] for all i<j pairs
n_pairs <- d * (d - 1) / 2  # 3
par(mfrow = c(n_pairs, 2), mar = c(3, 4, 2.5, 1))
for (i in 1:(d - 1)) {
  for (j in (i + 1):d) {
    tr_og  <- fit_og$Sigma_post[i, j, ]
    tr_new <- fit_new$Sigma_post[i, j, ]
    lbl    <- sprintf("R[%d,%d]", i, j)
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

# -- Panel 5: Accuracy bar chart (test set)
par(mfrow = c(1, 1), mar = c(5, 5, 3, 1))
acc_te <- numeric(d * 2)
nms    <- character(d * 2)
for (j in 1:d) {
  acc_te[2*j-1] <- accuracy(fit_og$y_hat_test_mean_class[, j],  y_test[, j])
  acc_te[2*j]   <- accuracy(fit_new$y_hat_test_mean_class[, j], y_test[, j])
  nms[2*j-1] <- sprintf("subart\nY%d", j)
  nms[2*j]   <- sprintf("subart2\nY%d", j)
}
bp <- barplot(acc_te, col = rep(c("steelblue","darkorange"), d),
              ylim = c(0, 1), ylab = "Test Accuracy",
              main = "Test Accuracy – Missing d=3 Classification",
              las = 2, names.arg = nms)
text(bp, acc_te + 0.02, sprintf("%.3f", acc_te), cex = 0.8)
abline(h = 0.5, lty = 2, col = "gray50")

# -- Panel 6: Variable importance
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og,  "subart – Var. Importance", d = d)
plot_varimportance(fit_new, "subart2 – Var. Importance", d = d)

par(op)
