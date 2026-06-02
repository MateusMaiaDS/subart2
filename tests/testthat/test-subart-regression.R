# Unit tests for subart() regression paths
# Covers: 1D univariate, 2D bivariate, general multivariate (d>2), and NA handling.
# All tests use small n_mcmc/n_burn/n_tree to keep runtime fast.

library(testthat)
library(subart2)

# Shared small dataset (30 obs, 5 predictors)
set.seed(42)
n  <- 30
p  <- 5
X  <- as.data.frame(matrix(rnorm(n * p), nrow = n))

# Helpers ----------------------------------------------------------------

make_y <- function(d) {
  matrix(rnorm(n * d), nrow = n, ncol = d)
}

fast_args <- list(
  n_tree  = 5,
  n_mcmc  = 50,
  n_burn  = 10,
  numcut  = 20L
)

run_subart <- function(y, x_test = NULL) {
  do.call(subart, c(
    list(x_train = X, y_train = y, x_test = x_test),
    fast_args
  ))
}

n_post <- fast_args$n_mcmc - fast_args$n_burn  # 40 posterior draws

# -----------------------------------------------------------------------
# 1.  1D univariate regression  (cppsubart_univariate path)
# -----------------------------------------------------------------------
test_that("1D univariate regression runs and returns expected structure", {
  y1 <- make_y(1)
  fit <- run_subart(y1)

  expect_s3_class(fit, "subart")
  # y_hat: matrix [n_train x n_post]
  expect_true(is.matrix(fit$y_hat))
  expect_equal(nrow(fit$y_hat), n)
  expect_equal(ncol(fit$y_hat), n_post)
  # y_hat_mean: vector of length n_train
  expect_equal(length(fit$y_hat_mean), n)
  # no x_test → test predictions stripped from output
  expect_null(fit$y_hat_test)
  expect_null(fit$y_hat_test_mean)
})

# -----------------------------------------------------------------------
# 2.  2D bivariate regression  (cppsubart_2d path)
# -----------------------------------------------------------------------
test_that("2D bivariate regression runs and returns expected structure", {
  y2 <- make_y(2)
  fit <- run_subart(y2)

  expect_s3_class(fit, "subart")
  # y_hat: 3D array [n_train x d x n_post]
  expect_equal(dim(fit$y_hat), c(n, 2L, n_post))
  # y_hat_mean: matrix [n_train x d]
  expect_true(is.matrix(fit$y_hat_mean))
  expect_equal(dim(fit$y_hat_mean), c(n, 2L))
  # no x_test → test predictions stripped
  expect_null(fit$y_hat_test)
  expect_null(fit$y_hat_test_mean)
})

# -----------------------------------------------------------------------
# 3.  General multivariate regression d = 3  (cppsubart path)
# -----------------------------------------------------------------------
test_that("3D multivariate regression runs and returns expected structure", {
  y3 <- make_y(3)
  fit <- run_subart(y3)

  expect_s3_class(fit, "subart")
  expect_equal(dim(fit$y_hat), c(n, 3L, n_post))
  expect_equal(dim(fit$y_hat_mean), c(n, 3L))
  expect_null(fit$y_hat_test)
})

# -----------------------------------------------------------------------
# 4.  x_test provided – test predictions are returned
# -----------------------------------------------------------------------
test_that("Providing x_test returns 2D test-set predictions", {
  y2 <- make_y(2)
  n_test <- 5L
  X_test <- as.data.frame(matrix(rnorm(n_test * p), nrow = n_test))
  fit <- run_subart(y2, x_test = X_test)

  expect_false(is.null(fit$y_hat_test))
  # 3D array [n_test x d x n_post]
  expect_equal(dim(fit$y_hat_test), c(n_test, 2L, n_post))
  expect_equal(dim(fit$y_hat_test_mean), c(n_test, 2L))
})

test_that("Providing x_test returns 1D test-set predictions", {
  y1 <- make_y(1)
  n_test <- 5L
  X_test <- as.data.frame(matrix(rnorm(n_test * p), nrow = n_test))
  fit <- run_subart(y1, x_test = X_test)

  expect_false(is.null(fit$y_hat_test))
  # matrix [n_post x n_test] or similar — just check not NULL and has test rows
  expect_equal(length(fit$y_hat_test_mean), n_test)
})

# -----------------------------------------------------------------------
# 5.  Missing data (NA) in y  (cppsubart_missing / cppsubart_missing_2d)
# -----------------------------------------------------------------------
test_that("2D regression with NA in y_train runs without error", {
  y2_na <- make_y(2)
  y2_na[sample(seq_len(n), 3), 1] <- NA

  expect_no_error(fit <- run_subart(y2_na))
  expect_s3_class(fit, "subart")
  expect_equal(dim(fit$y_hat_mean), c(n, 2L))
})

test_that("3D regression with NA in y_train runs without error", {
  y3_na <- make_y(3)
  y3_na[sample(seq_len(n), 3), 2] <- NA

  expect_no_error(fit <- run_subart(y3_na))
  expect_s3_class(fit, "subart")
  expect_equal(dim(fit$y_hat_mean), c(n, 3L))
})

# -----------------------------------------------------------------------
# 6.  Input validation errors
# -----------------------------------------------------------------------
test_that("n_mcmc <= n_burn raises an error", {
  y1 <- make_y(1)
  expect_error(
    subart(x_train = X, y_train = y1, n_mcmc = 20, n_burn = 20),
    "Number of MCMC iterations must be greater"
  )
})

test_that("specify_variables with wrong length raises an error", {
  y2 <- make_y(2)
  expect_error(
    do.call(subart, c(
      list(x_train = X, y_train = y2,
           specify_variables = list(1:3)),  # length 1, not 2
      fast_args
    )),
    "specify_variables must be a list with one element per response column"
  )
})
