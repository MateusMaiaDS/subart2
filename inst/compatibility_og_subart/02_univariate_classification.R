# =============================================================================
# 02_univariate_classification.R
# Comparison: subart (original) vs subart2 — Univariate Probit Classification
# =============================================================================
# Scenario: Single binary response via probit link
#   latent z = sin(pi*x1*x2) + x3^3 + N(0,1),  y = I(z > 0)
#   n_train=200, n_test=50, p=5
#
# Plots:   Boxplot of predicted latent z by true class (train/test)
#          Accuracy comparison bar chart
# Metrics: Accuracy (train, test), timing speedup
# Note:    d=1 classification has no Sigma_post (no traceplot for Sigma)
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

x_all <- data.frame(matrix(stats::runif(n * p, min = -1, max = 1), ncol = p,
                            dimnames = list(NULL, paste0("X", 1:p))))
z_true_all <- with(x_all, sin(pi * X1 * X2) + X3^3)
z_all      <- z_true_all + stats::rnorm(n, 0, 1)
y_all      <- matrix(as.integer(z_all > 0), ncol = 1)
p_true_all <- stats::pnorm(z_true_all)

x_train    <- x_all[1:n_tr, ]
x_test     <- x_all[(n_tr + 1):n, ]
y_train    <- y_all[1:n_tr, , drop = FALSE]
y_test     <- y_all[(n_tr + 1):n, , drop = FALSE]
p_true_tr  <- p_true_all[1:n_tr]
p_true_te  <- p_true_all[(n_tr + 1):n]

cat("=== 02 Univariate Classification ===\n")
cat(sprintf("n_train=%d  n_test=%d  p=%d  prevalence_train=%.3f\n",
            n_tr, n - n_tr, p, mean(y_train)))

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

print_timing_table(time_og, time_new, scenario = "Univariate Classification")
print_accuracy_table(fit_og, fit_new, y_train, y_test)

# ---- Plots ------------------------------------------------------------------

op <- par(no.readonly = TRUE)
on.exit(par(op))

# -- Panel 1: Latent z boxplots by true class (train)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot_latent_by_class(fit_og$y_hat_mean,  y_train, j = 1,
                     "subart – Train",  "steelblue")
plot_latent_by_class(fit_new$y_hat_mean, y_train, j = 1,
                     "subart2 – Train", "darkorange")
plot_latent_by_class(fit_og$y_hat_test_mean,  y_test, j = 1,
                     "subart – Test",  "steelblue")
plot_latent_by_class(fit_new$y_hat_test_mean, y_test, j = 1,
                     "subart2 – Test", "darkorange")

# -- Panel 2: Predicted probability (latent mean converted to P via pnorm)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
prob_og_tr  <- stats::pnorm(fit_og$y_hat_mean)
prob_new_tr <- stats::pnorm(fit_new$y_hat_mean)
prob_og_te  <- stats::pnorm(fit_og$y_hat_test_mean)
prob_new_te <- stats::pnorm(fit_new$y_hat_test_mean)

scatter_pred_vs_true(prob_og_tr,  p_true_tr, "subart – P(y=1) Train",  "steelblue")
scatter_pred_vs_true(prob_new_tr, p_true_tr, "subart2 – P(y=1) Train", "darkorange")
scatter_pred_vs_true(prob_og_te,  p_true_te, "subart – P(y=1) Test",   "steelblue")
scatter_pred_vs_true(prob_new_te, p_true_te, "subart2 – P(y=1) Test",  "darkorange")

# -- Panel 3: Accuracy bar chart
par(mfrow = c(1, 1), mar = c(5, 5, 3, 1))
acc_vals <- c(
  `subart\ntrain`  = accuracy(fit_og$y_hat_mean_class,       y_train),
  `subart2\ntrain` = accuracy(fit_new$y_hat_mean_class,      y_train),
  `subart\ntest`   = accuracy(fit_og$y_hat_test_mean_class,  y_test),
  `subart2\ntest`  = accuracy(fit_new$y_hat_test_mean_class, y_test)
)
bp <- barplot(acc_vals, col = c("steelblue", "darkorange", "steelblue", "darkorange"),
              ylim = c(0, 1), ylab = "Accuracy", main = "Accuracy Comparison",
              las = 2, names.arg = c("subart\ntrain","subart2\ntrain",
                                     "subart\ntest","subart2\ntest"))
text(bp, acc_vals + 0.02, sprintf("%.3f", acc_vals), cex = 0.85)
abline(h = 0.5, lty = 2, col = "gray50")

# -- Panel 4: Variable importance
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og,  "subart – Var. Importance")
plot_varimportance(fit_new, "subart2 – Var. Importance")

par(op)
