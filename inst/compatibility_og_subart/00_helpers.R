# =============================================================================
# 00_helpers.R  --  Shared utilities for subart vs subart2 comparisons
# =============================================================================
# Source this at the top of each comparison script:
#   source(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "00_helpers.R"))
# or from the package root:
#   source("inst/compatibility_og_subart/00_helpers.R")

# ---- Package checks ---------------------------------------------------------

check_and_install <- function(pkg, repo) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  if (pkg == "subart2") {
    # Always install from the phase2-cpp branch to ensure C++14 build and
    # latest varimportance / classification features are present.
    message("Installing subart2 from feat/phase2-cpp branch ...")
    remotes::install_github("MateusMaiaDS/subart2", ref = "feat/phase2-cpp",
                            upgrade = "never", quiet = FALSE)
  } else if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Package '%s' not found. Installing from GitHub (%s) ...", pkg, repo))
    remotes::install_github(repo)
  }
}

setup_packages <- function() {
  check_and_install("subart",  "MateusMaiaDS/subart")
  check_and_install("subart2", "MateusMaiaDS/subart2")
  check_and_install("mvnfast", "nicholasjclark/mvnfast")
  invisible(NULL)
}

# ---- Metrics ----------------------------------------------------------------

rmse <- function(pred, true) sqrt(mean((pred - true)^2, na.rm = TRUE))

accuracy <- function(pred, true) mean(pred == true, na.rm = TRUE)

rmse_multivariate <- function(pred_mat, true_mat) {
  mean(sapply(seq_len(ncol(pred_mat)), function(j)
    rmse(pred_mat[, j], true_mat[, j])))
}

accuracy_multivariate <- function(pred_mat, true_mat) {
  mean(sapply(seq_len(ncol(pred_mat)), function(j)
    accuracy(pred_mat[, j], true_mat[, j])))
}

# ---- Timing printout --------------------------------------------------------

print_timing_table <- function(time_og, time_new, scenario = "") {
  cat("\n")
  if (nchar(scenario) > 0) cat(sprintf("=== Scenario: %s ===\n", scenario))
  cat("--- Timing Comparison ---\n")
  cat(sprintf("  subart (original) : %.2f s\n", time_og["elapsed"]))
  cat(sprintf("  subart2 (new)     : %.2f s\n", time_new["elapsed"]))
  ratio <- time_og["elapsed"] / time_new["elapsed"]
  if (ratio > 1.0) {
    cat(sprintf("  --> subart2 is %.2fx FASTER\n\n", ratio))
  } else if (ratio < 1.0) {
    cat(sprintf("  --> subart2 is %.2fx SLOWER\n\n", 1 / ratio))
  } else {
    cat("  --> Equivalent speed\n\n")
  }
}

print_rmse_table <- function(fit_og, fit_new, y_train_true, y_test_true = NULL) {
  cat("--- RMSE ---\n")
  get_rmse <- function(fit, y_true, slot) {
    yh <- fit[[slot]]
    if (is.null(yh)) return(NA_real_)
    if (is.matrix(yh) && is.matrix(y_true)) rmse_multivariate(yh, y_true)
    else rmse(as.numeric(yh), as.numeric(y_true))
  }
  cat(sprintf("  Train  subart: %.4f  |  subart2: %.4f\n",
              get_rmse(fit_og,  y_train_true, "y_hat_mean"),
              get_rmse(fit_new, y_train_true, "y_hat_mean")))
  if (!is.null(y_test_true)) {
    cat(sprintf("  Test   subart: %.4f  |  subart2: %.4f\n",
                get_rmse(fit_og,  y_test_true, "y_hat_test_mean"),
                get_rmse(fit_new, y_test_true, "y_hat_test_mean")))
  }
  cat("\n")
}

print_accuracy_table <- function(fit_og, fit_new, y_train_true, y_test_true = NULL) {
  cat("--- Accuracy ---\n")
  get_acc <- function(fit, y_true, slot) {
    yh <- fit[[slot]]
    if (is.null(yh)) return(NA_real_)
    if (is.matrix(yh) && is.matrix(as.matrix(y_true)))
      accuracy_multivariate(yh, as.matrix(y_true))
    else accuracy(as.integer(yh), as.integer(y_true))
  }
  cat(sprintf("  Train  subart: %.4f  |  subart2: %.4f\n",
              get_acc(fit_og,  y_train_true, "y_hat_mean_class"),
              get_acc(fit_new, y_train_true, "y_hat_mean_class")))
  if (!is.null(y_test_true)) {
    cat(sprintf("  Test   subart: %.4f  |  subart2: %.4f\n",
                get_acc(fit_og,  y_test_true, "y_hat_test_mean_class"),
                get_acc(fit_new, y_test_true, "y_hat_test_mean_class")))
  }
  cat("\n")
}

# ---- Regression: scatter predicted vs true ----------------------------------

scatter_pred_vs_true <- function(pred, true, main = "", col = "steelblue",
                                  add_rmse = TRUE, cex_pts = 0.45) {
  lims <- range(c(as.numeric(pred), as.numeric(true)), na.rm = TRUE)
  plot(as.numeric(true), as.numeric(pred),
       pch = 19, cex = cex_pts, col = adjustcolor(col, 0.65),
       xlim = lims, ylim = lims,
       main = main, xlab = "True", ylab = "Predicted", cex.main = 0.9)
  abline(0, 1, col = "firebrick", lwd = 1.5, lty = 2)
  if (add_rmse)
    legend("topleft", bty = "n", cex = 0.75,
           legend = sprintf("RMSE = %.4f", rmse(pred, true)))
}

# Plot train + test scatter for one outcome j (side by side: subart vs subart2)
scatter_comparison_reg <- function(fit_og, fit_new,
                                    y_train_true, y_test_true,
                                    j = 1, d = 1) {
  extract <- function(fit, slot, j, d) {
    x <- fit[[slot]]
    if (is.null(x)) return(NULL)
    if (d == 1) as.numeric(x) else x[, j]
  }
  og_tr  <- extract(fit_og,  "y_hat_mean",      j, d)
  new_tr <- extract(fit_new, "y_hat_mean",      j, d)
  og_te  <- extract(fit_og,  "y_hat_test_mean", j, d)
  new_te <- extract(fit_new, "y_hat_test_mean", j, d)

  tr_true <- if (d == 1) as.numeric(y_train_true) else y_train_true[, j]
  te_true <- if (d == 1) as.numeric(y_test_true)  else y_test_true[, j]

  scatter_pred_vs_true(og_tr,  tr_true, sprintf("subart – Train (Y%d)",  j), "steelblue")
  scatter_pred_vs_true(new_tr, tr_true, sprintf("subart2 – Train (Y%d)", j), "darkorange")
  if (!is.null(og_te) && !is.null(te_true)) {
    scatter_pred_vs_true(og_te,  te_true, sprintf("subart – Test (Y%d)",  j), "steelblue")
    scatter_pred_vs_true(new_te, te_true, sprintf("subart2 – Test (Y%d)", j), "darkorange")
  }
}

# ---- Traceplots -------------------------------------------------------------

traceplot_sigma_1d <- function(sigma_post, main = "", col = "steelblue",
                                n_burn = NULL) {
  if (is.null(sigma_post)) {
    plot.new(); title(paste(main, "(no samples)"), cex.main = 0.8)
    return(invisible(NULL))
  }
  chain <- as.numeric(sigma_post)
  plot(chain, type = "l", col = col,
       main = main, xlab = "Iteration", ylab = expression(sigma^2), cex.main = 0.9)
  if (!is.null(n_burn)) abline(v = n_burn, col = "firebrick", lty = 2, lwd = 1.2)
  abline(h = mean(chain, na.rm = TRUE), col = "navy", lty = 3, lwd = 1.2)
  legend("topright", bty = "n", cex = 0.7,
         legend = sprintf("mean = %.4f", mean(chain, na.rm = TRUE)))
}

traceplot_sigma_nd <- function(Sigma_post, d, main_prefix = "",
                                col = "steelblue", n_burn = NULL,
                                off_diag_only = FALSE) {
  if (is.null(Sigma_post)) return(invisible(NULL))
  for (i in seq_len(d)) {
    jstart <- if (off_diag_only) i + 1L else i
    if (jstart > d) next
    for (j in jstart:d) {
      tr  <- Sigma_post[i, j, ]
      lbl <- sprintf("Sigma[%d,%d]", i, j)
      plot(tr, type = "l", col = col,
           main = sprintf("%s  %s", main_prefix, lbl),
           xlab = "Iteration", ylab = lbl, cex.main = 0.85)
      if (!is.null(n_burn)) abline(v = n_burn, col = "firebrick", lty = 2, lwd = 1.2)
      abline(h = mean(tr, na.rm = TRUE), col = "navy", lty = 3, lwd = 1.2)
      legend("topright", bty = "n", cex = 0.7,
             legend = sprintf("mean = %.4f", mean(tr, na.rm = TRUE)))
    }
  }
}

# ---- Classification: boxplot of latent z by true class ---------------------

plot_latent_by_class <- function(z_hat, y_true, j = 1, main = "",
                                   col = "steelblue") {
  zj <- if (is.matrix(z_hat)) z_hat[, j] else as.numeric(z_hat)
  yj <- if (is.matrix(y_true)) y_true[, j] else as.integer(y_true)
  boxplot(list(`y=0` = zj[yj == 0], `y=1` = zj[yj == 1]),
          col = c(adjustcolor("salmon", 0.7), adjustcolor(col, 0.7)),
          main = main, ylab = "Predicted latent z",
          outline = FALSE, cex.main = 0.85)
  abline(h = 0, lty = 2, col = "gray40", lwd = 1.2)
}

# ---- Variable importance bar chart ------------------------------------------

plot_varimportance <- function(fit, main = "Variable Importance", d = 1) {
  vi <- fit$var_importance
  if (is.null(vi)) {
    plot.new(); title(paste(main, "(not available)"), cex.main = 0.8)
    return(invisible(NULL))
  }
  if (is.numeric(vi) && is.vector(vi)) {
    barplot(vi, las = 2, main = main, ylab = "Mean split count",
            col = "steelblue", cex.names = 0.75, cex.main = 0.9)
  } else if (is.matrix(vi)) {
    # Rows = predictors, cols = outcomes (or vice versa; take column means)
    barplot(colMeans(vi), las = 2, main = main,
            ylab = "Mean split count", col = "steelblue",
            cex.names = 0.75, cex.main = 0.9)
  } else if (is.array(vi) && length(dim(vi)) == 3) {
    vi_mean <- apply(vi, 2, mean)
    barplot(vi_mean, las = 2, main = main,
            ylab = "Mean split count", col = "steelblue",
            cex.names = 0.75, cex.main = 0.9)
  }
}

# ---- Figure heading banner --------------------------------------------------

fig_banner <- function(title, col = "gray20") {
  plot.new()
  text(0.5, 0.5, title, cex = 1.2, font = 2, col = col, adj = 0.5)
}
