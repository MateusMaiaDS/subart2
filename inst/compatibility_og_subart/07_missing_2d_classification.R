# =============================================================================
# 07_missing_2d_classification.R
# Comparison: subart (original) vs subart2 — Missing Labels, d=2 Probit
# =============================================================================
# Scenario: d=2 binary responses with ~15% of y_train labels set to NA
#   (independently per outcome), using sim_class_mvn_friedman1.
#   n_train=250, n_test=50, p=5
#
# How each package handles missing labels for d>1 classification:
#   subart (original): replaces NA with -1 before cppbart_CLASS
#   subart2:           replaces NA with 0  before cppsubart_CLASS
#   In both cases missing labels are treated as class 0/negative; this is
#   NOT a principled missing-data model but is the packages' current behavior.
#
# Plots:   Latent z boxplots by true class per outcome (train/test)
#          Predicted probability vs true probability
#          R[1,2] traceplot
#          Accuracy on observed subset
# Metrics: Accuracy on observed labels only, timing speedup
# =============================================================================

HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n       <- 300
n_tr    <- 250
p       <- 5
d       <- 2
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

# Introduce NA independently per outcome column
set.seed(123)
y_train_miss <- y_train_full
obs_mask     <- matrix(TRUE, nrow = n_tr, ncol = d)
for (j in 1:d) {
  na_j  <- sample(n_tr, floor(na_frac * n_tr))
  y_train_miss[na_j, j] <- NA
  obs_mask[na_j, j]     <- FALSE
}

cat("=== 07 Missing Bivariate Classification (d=2) ===\n")
cat(sprintf("n_train=%d  NA per col ~%.0f%%  n_test=%d  p=%d\n",
            n_tr, 100 * na_frac, n - n_tr, p))
cat(sprintf("Observed labels: Y1=%d  Y2=%d (of %d)\n",
            sum(obs_mask[, 1]), sum(obs_mask[, 2]), n_tr))

# ---- Fit models (using packages' default NA handling) -----------------------

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

print_timing_table(time_og, time_new, scenario = "Missing d=2 Classification")

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

# -- Panel 1: Latent z boxplots — train (observed labels only)
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

# -- Panel 3: Predicted probability vs true (train, per outcome)
par(mfrow = c(2, d), mar = c(4, 4, 3, 1))
for (j in 1:d) {
  scatter_pred_vs_true(stats::pnorm(fit_og$y_hat_mean[, j]),  p_true_tr[, j],
                       sprintf("subart  – P(Y%d=1) Train", j),  "steelblue")
  scatter_pred_vs_true(stats::pnorm(fit_new$y_hat_mean[, j]), p_true_tr[, j],
                       sprintf("subart2 – P(Y%d=1) Train", j), "darkorange")
}

# -- Panel 4: R[1,2] traceplot
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
traceplot_sigma_nd(fit_og$Sigma_post,  d = d, main_prefix = "subart",
                   col = "steelblue",  n_burn = 200, off_diag_only = TRUE)
traceplot_sigma_nd(fit_new$Sigma_post, d = d, main_prefix = "subart2",
                   col = "darkorange", n_burn = 200, off_diag_only = TRUE)

# -- Panel 5: Accuracy bar chart (test set)
par(mfrow = c(1, 1), mar = c(5, 5, 3, 1))
acc_te <- c(
  `subart\nY1`  = accuracy(fit_og$y_hat_test_mean_class[, 1],  y_test[, 1]),
  `subart2\nY1` = accuracy(fit_new$y_hat_test_mean_class[, 1], y_test[, 1]),
  `subart\nY2`  = accuracy(fit_og$y_hat_test_mean_class[, 2],  y_test[, 2]),
  `subart2\nY2` = accuracy(fit_new$y_hat_test_mean_class[, 2], y_test[, 2])
)
bp <- barplot(acc_te, col = c("steelblue","darkorange","steelblue","darkorange"),
              ylim = c(0, 1), ylab = "Test Accuracy",
              main = "Test Accuracy – Missing d=2 Classification", las = 2)
text(bp, acc_te + 0.02, sprintf("%.3f", acc_te), cex = 0.85)
abline(h = 0.5, lty = 2, col = "gray50")

# -- Panel 6: Variable importance
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og,  "subart – Var. Importance", d = d)
plot_varimportance(fit_new, "subart2 – Var. Importance", d = d)

par(op)
