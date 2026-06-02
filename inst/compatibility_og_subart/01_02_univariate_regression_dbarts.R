# =============================================================================
# 01_univariate_regression.R
# Comparison: subart (original) vs subart2 — Univariate Regression
# =============================================================================
# Scenario: Friedman-1 single-response regression
#   y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + N(0,1)
#   n_train=200, n_test=50, p=5
#
# Plots:   Train/test scatter (predicted vs true) for each model
#          sigma^2 posterior traceplots
#          Variable importance
# Metrics: RMSE (train, test), timing speedup
# =============================================================================

# Source helpers (adjust path as needed when running interactively)
HERE <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path),
                 error = function(e) "inst/compatibility_og_subart")
source(file.path(HERE, "00_helpers.R"))

setup_packages()

# ---- Simulation -------------------------------------------------------------

set.seed(42)
n     <- 250
p     <- 5
n_tr  <- 200
n_te  <- 200

x_all   <- data.frame(matrix(stats::runif(n * p), ncol = p,
                             dimnames = list(NULL, paste0("X", 1:p))))
y_true_all <- with(x_all,
                   10 * sin(pi * X1 * X2) + 20 * (X3 - 0.5)^2 + 10 * X4 + 5 * X5)
y_all <- matrix(y_true_all + stats::rnorm(n, 0, 1), ncol = 1)

x_train      <- x_all[1:n_tr, ]
x_test       <- x_all[(n_tr + 1):n, ]
y_train      <- y_all[1:n_tr, , drop = FALSE]
y_test       <- y_all[(n_tr + 1):n, , drop = FALSE]
y_true_train <- y_true_all[1:n_tr]
y_true_test  <- y_true_all[(n_tr + 1):n]

cat("=== 01 Univariate Regression ===\n")
cat(sprintf("n_train=%d  n_test=%d  p=%d\n", n_tr, n_te, p))

# ---- Fit models -------------------------------------------------------------

MCMC_SETTINGS <- list(n_tree = 50, n_mcmc = 1000, n_burn = 200,
                      varimportance = TRUE, diagnostic = FALSE)

time_og <- system.time({
  fit_og <- do.call(dbarts::bart,
                    c(list(x.train = x_train, y.train = y_train,
                           x.test  = x_test,ntree=50,nskip = 200, ndpost = 1000)))
})

time_new <- system.time({
  fit_new <- do.call(subart2::subart,
                     c(list(x_train = x_train, y_train = y_train,
                            x_test  = x_test),
                       MCMC_SETTINGS))
})

# ---- Console output ---------------------------------------------------------

print_timing_table(time_og, time_new, scenario = "Univariate Regression")
plot(fit_og$yhat.train.mean,fit_new$y_hat_mean,xlab = "dbarts",ylab = "subart")


# ---- Plots ------------------------------------------------------------------

op <- par(no.readonly = TRUE)
on.exit(par(op))

# -- Panel 1: Scatter plots (train + test)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
scatter_pred_vs_true(fit_og$y_hat_mean,       y_true_train,
                     "subart – Train",       col = "steelblue")
scatter_pred_vs_true(fit_new$y_hat_mean,      y_true_train,
                     "subart2 – Train",      col = "darkorange")
scatter_pred_vs_true(fit_og$y_hat_test_mean,  y_true_test,
                     "subart – Test",        col = "steelblue")
scatter_pred_vs_true(fit_new$y_hat_test_mean, y_true_test,
                     "subart2 – Test",       col = "darkorange")

# -- Panel 2: sigma^2 traceplots
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
traceplot_sigma_1d(fit_og$Sigma_post,  "subart  – sigma^2",  "steelblue",  n_burn = 200)
traceplot_sigma_1d(fit_new$Sigma_post, "subart2 – sigma^2",  "darkorange", n_burn = 200)

# -- Panel 3: Variable importance
par(mfrow = c(1, 2), mar = c(5, 4, 3, 1))
plot_varimportance(fit_og,  "subart – Var. Importance")
plot_varimportance(fit_new, "subart2 – Var. Importance")

par(op)
