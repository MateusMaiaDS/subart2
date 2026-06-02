source("inst/compatibility_og_subart/00_helpers.R")
setup_packages()

set.seed(42)
n <- 250; p <- 5; n_tr <- 200
x_all <- data.frame(matrix(stats::runif(n*p), ncol=p,
                            dimnames=list(NULL, paste0("X",1:p))))
y_true_all <- with(x_all, 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5)
y_all <- matrix(y_true_all + stats::rnorm(n, 0, 1), ncol=1)
x_train <- x_all[1:n_tr, ]; x_test <- x_all[(n_tr+1):n, ]
y_train <- y_all[1:n_tr, , drop=FALSE]
y_true_train <- y_true_all[1:n_tr]; y_true_test <- y_true_all[(n_tr+1):n]

MCMC <- list(n_tree=50, n_mcmc=1000, n_burn=200, varimportance=TRUE, diagnostic=FALSE)

fit_og  <- do.call(subart::subart,  c(list(x_train=x_train, y_train=y_train, x_test=x_test), MCMC))
fit_new <- do.call(subart2::subart, c(list(x_train=x_train, y_train=y_train, x_test=x_test), MCMC))

cat("subart  train RMSE:", sqrt(mean((fit_og$y_hat_mean  - y_true_train)^2)), "\n")
cat("subart2 train RMSE:", sqrt(mean((fit_new$y_hat_mean - y_true_train)^2)), "\n")
cat("subart  test  RMSE:", sqrt(mean((fit_og$y_hat_test_mean  - y_true_test)^2)), "\n")
cat("subart2 test  RMSE:", sqrt(mean((fit_new$y_hat_test_mean - y_true_test)^2)), "\n")
cat("y_true_train range:", range(y_true_train), "\n")
cat("subart  y_hat_mean range:", range(fit_og$y_hat_mean), "\n")
cat("subart2 y_hat_mean range:", range(fit_new$y_hat_mean), "\n")
cat("subart  y_hat_test_mean range:", range(fit_og$y_hat_test_mean), "\n")
cat("subart2 y_hat_test_mean range:", range(fit_new$y_hat_test_mean), "\n")
# Check if predictions look scaled differently
cat("\nsubart  y_hat_mean  head:", head(fit_og$y_hat_mean), "\n")
cat("subart2 y_hat_mean  head:", head(fit_new$y_hat_mean), "\n")
cat("y_true_train        head:", head(y_true_train), "\n")
