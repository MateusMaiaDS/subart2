# =============================================================================
# 06_missing_univariate_classification.R
# Comparison: subart (original) vs subart2 — Missing Labels, Univariate Probit
# =============================================================================
# Scenario: Single binary response with ~15% of y_train set to NA.
#
# Important note on missing labels and d=1 classification:
#   Both packages treat missing binary labels (NA in y_train) as class 0 in
#   the d=1 probit BART. This is NOT a proper missing-label model; it is
#   equivalent to adding pseudo-negatives. The comparison demonstrates:
#     (A) Complete-case analysis — fit on rows with observed y only.
#     (B) NA-as-zero — pass y with NAs (both packages replace NA with 0).
#   The accuracy is evaluated exclusively on the non-NA training observations.
#
# Plots:   Latent z boxplots by true class (approaches A & B)
#          Accuracy bar chart
# Metrics: Accuracy on observed subset, timing speedup
# =============================================================================

HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n       <- 250
p       <- 5
n_tr    <- 200
na_frac <- 0.15

x_all <- data.frame(matrix(stats::runif(n * p, min = -1, max = 1), ncol = p,
                            dimnames = list(NULL, paste0("X", 1:p))))
z_true_all <- with(x_all, sin(pi * X1 * X2) + X3^3)
z_all      <- z_true_all + stats::rnorm(n, 0, 1)
y_all      <- matrix(as.integer(z_all > 0), ncol = 1)

x_train <- x_all[1:n_tr, ]
x_test  <- x_all[(n_tr + 1):n, ]
y_train_full <- y_all[1:n_tr, , drop = FALSE]
y_test       <- y_all[(n_tr + 1):n, , drop = FALSE]

set.seed(123)
na_idx        <- sample(n_tr, floor(na_frac * n_tr))
y_train_miss  <- y_train_full
y_train_miss[na_idx, ] <- NA
obs_idx <- setdiff(seq_len(n_tr), na_idx)

cat("=== 06 Missing Univariate Classification ===\n")
cat(sprintf("n_train=%d  NA=%d (%.0f%%)  n_test=%d  p=%d\n",
            n_tr, length(na_idx), 100 * na_frac, n - n_tr, p))
cat("NOTE: d=1 missing labels not specially handled; NAs replaced with 0\n\n")

# ---- Approach A: complete-case analysis -------------------------------------

x_train_cc <- x_train[obs_idx, ]
y_train_cc <- y_train_miss[obs_idx, , drop = FALSE]

MCMC <- list(n_tree = 50, n_mcmc = 1000, n_burn = 200,
             varimportance = TRUE, diagnostic = FALSE)

cat("--- Approach A: Complete-case ---\n")
time_og_A <- system.time({
  fit_og_A <- do.call(subart::subart,
                      c(list(x_train = x_train_cc, y_train = y_train_cc,
                             x_test  = x_test), MCMC))
})
time_new_A <- system.time({
  fit_new_A <- do.call(subart2::subart,
                       c(list(x_train = x_train_cc, y_train = y_train_cc,
                              x_test  = x_test), MCMC))
})
print_timing_table(time_og_A, time_new_A, scenario = "Approach A (complete-case)")
print_accuracy_table(fit_og_A, fit_new_A,
                     y_train_full[obs_idx, , drop = FALSE], y_test)

# ---- Approach B: NA replaced with 0 (both packages behave identically) -----

cat("--- Approach B: NA replaced with 0 (pseudo-negative) ---\n")
y_train_B <- y_train_miss
y_train_B[na_idx, ] <- 0L

time_og_B <- system.time({
  fit_og_B <- do.call(subart::subart,
                      c(list(x_train = x_train, y_train = y_train_B,
                             x_test  = x_test), MCMC))
})
time_new_B <- system.time({
  fit_new_B <- do.call(subart2::subart,
                       c(list(x_train = x_train, y_train = y_train_B,
                              x_test  = x_test), MCMC))
})
print_timing_table(time_og_B, time_new_B, scenario = "Approach B (NA→0)")

# Accuracy evaluated only on observed subset (fair comparison)
cat("Accuracy on observed training indices only:\n")
cat(sprintf("  subart  A: train=%.4f  B: train=%.4f\n",
            accuracy(fit_og_A$y_hat_mean_class,
                     y_train_full[obs_idx, , drop = FALSE]),
            accuracy(fit_og_B$y_hat_mean_class[obs_idx],
                     y_train_full[obs_idx, , drop = FALSE])))
cat(sprintf("  subart2 A: train=%.4f  B: train=%.4f\n",
            accuracy(fit_new_A$y_hat_mean_class,
                     y_train_full[obs_idx, , drop = FALSE]),
            accuracy(fit_new_B$y_hat_mean_class[obs_idx],
                     y_train_full[obs_idx, , drop = FALSE])))
print_accuracy_table(fit_og_A, fit_new_A,
                     y_train_full[obs_idx, , drop = FALSE], y_test)

# ---- Plots ------------------------------------------------------------------

op <- par(no.readonly = TRUE)
on.exit(par(op))

# -- Panel 1: Latent z boxplots — Approach A
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot_latent_by_class(fit_og_A$y_hat_mean, y_train_full[obs_idx, , drop = FALSE],
                     j = 1, "subart A – Train (CC)", "steelblue")
plot_latent_by_class(fit_new_A$y_hat_mean, y_train_full[obs_idx, , drop = FALSE],
                     j = 1, "subart2 A – Train (CC)", "darkorange")
plot_latent_by_class(fit_og_A$y_hat_test_mean, y_test,
                     j = 1, "subart A – Test", "steelblue")
plot_latent_by_class(fit_new_A$y_hat_test_mean, y_test,
                     j = 1, "subart2 A – Test", "darkorange")

# -- Panel 2: Latent z boxplots — Approach B (obs subset highlighted)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot_latent_by_class(fit_og_B$y_hat_mean[obs_idx], y_train_full[obs_idx, ],
                     j = 1, "subart B – Train obs only (NA→0)", "steelblue")
plot_latent_by_class(fit_new_B$y_hat_mean[obs_idx], y_train_full[obs_idx, ],
                     j = 1, "subart2 B – Train obs only (NA→0)", "darkorange")
plot_latent_by_class(fit_og_B$y_hat_test_mean, y_test,
                     j = 1, "subart B – Test", "steelblue")
plot_latent_by_class(fit_new_B$y_hat_test_mean, y_test,
                     j = 1, "subart2 B – Test", "darkorange")

# -- Panel 3: Accuracy summary bar chart
par(mfrow = c(1, 1), mar = c(6, 5, 3, 1))
acc_vals <- c(
  `og\nA-train`  = accuracy(fit_og_A$y_hat_mean_class,
                             y_train_full[obs_idx, , drop = FALSE]),
  `new\nA-train` = accuracy(fit_new_A$y_hat_mean_class,
                             y_train_full[obs_idx, , drop = FALSE]),
  `og\nB-train`  = accuracy(fit_og_B$y_hat_mean_class[obs_idx],
                             y_train_full[obs_idx, , drop = FALSE]),
  `new\nB-train` = accuracy(fit_new_B$y_hat_mean_class[obs_idx],
                             y_train_full[obs_idx, , drop = FALSE]),
  `og\ntest`     = accuracy(fit_og_A$y_hat_test_mean_class, y_test),
  `new\ntest`    = accuracy(fit_new_A$y_hat_test_mean_class, y_test)
)
bp <- barplot(acc_vals, col = rep(c("steelblue","darkorange"), 3),
              ylim = c(0, 1), ylab = "Accuracy",
              main = "Accuracy – Missing Label Comparison",
              las = 2, names.arg = names(acc_vals))
text(bp, acc_vals + 0.02, sprintf("%.3f", acc_vals), cex = 0.8)
abline(h = 0.5, lty = 2, col = "gray50")

# -- Panel 4: Variable importance (Approach A)
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og_A,  "subart A – Var. Importance")
plot_varimportance(fit_new_A, "subart2 A – Var. Importance")

par(op)
