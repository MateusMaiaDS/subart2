#' Multivariate Normal Bayesian Additive Regression trees.
#' @useDynLib subart2
#' @importFrom Rcpp sourceCpp
#'
#' @description
#' \code{subart()} function models a Bayesian Additive Regression trees model considering the Multivariate Normal (MVN) distribution
#' from the target dependent variable \eqn{Y_{i} \in \mathbb{R}^{d}}.
#'
#' @details
#'
#' Add more details of the function description
#'
#' @returns
#' In case of a continuous response the model returns
#' \code{subart} object with the model predictions and parameters for the standard MVN. For the binary-classification problems the probit
#' Probit-MVN approach is used, and returns the \code{subart} object.
#'
#'
#' @param x_train A \code{data.frame} of the training data covariates.
#' @param y_mat A numeric matrix of the data responses.
#' @param x_test A \code{data.frame} of the test data covariates.
#' @param n_tree The number of trees used in each set of trees for the respective \eqn{j} entry. The total number of trees is given by \eqn{j \times p}.
#' @param node_min_size The minimun number of observations within a terminal node.
#' @param n_mcmc The total number of MCMC iterations.
#' @param n_burn The number of MCMC iterations to be tretead as burn in samples
#' @param alpha The power parameter used in tree prior.
#' @param beta The base parameter used in tree prior.
#' @param nu The \eqn{\nu} parameter associated with the degree of freedom from the variance prior.
#' @param sigquant Quantile used to define the residuals variance. Remind that the prior definition is based on \eqn{P(\sigma^{(j)} < \hat{\sigma}^{(j)})} where \eqn{\hat{\sigma}^{(j)}} is a "rough data-based estimation" for example, the sample variance of the observed \eqn{y^{(j)}} values.
#' @param kappa Hyper-parameter from the \eqn{\mu_{t\ell}^{(j)}} (Chipman et. al 2010). Such that \eqn{\sigma} = 0.25
#' @param numcut The maximum number of possible values used in the tree decision rules. The uniform approximation for choose a decision rule over \eqn{X^{(j)}} is given a grid of size \code{numcut}.
#' @param usequants Boolean; if true the quantiles are going to be used to define the grid of cutpoints.
#' @param m Hyperparameter used in the definition of the prior setting of the correlation matrix for the Probit-Multivariate approach.
#' @param varimportance Boolean; if \code{TRUE} returns a matrix with \code{n_mcmc} rows and \eqn{d} columns corresponding to the total number of times each variable was used across all trees in a MCMC iteration.
#' @param hier_prior_bool Boolean; if true the prior for Sigma is defined using the hierarchical prior as defined by Esser & Maia et. al 2025. See details for reference.
#' @param specify_variables A list of numeric vectors where each element contains the indices of the covariates allowed to be selected for the trees of the respective response \eqn{Y_j}. Default is \code{NULL}, which allows all covariates for all trees.
#' @param diagnostic a boolean to compute or not the ESS for the posterior samples of the \eqn{\boldsymbol{\Sigma}}
#'
#' @export
subart <- function(x_train,
                   y_train,
                   x_test = NULL,
                   n_tree = 100,
                   node_min_size = 5,
                   n_mcmc = 2000,
                   n_burn = 500,
                   alpha = 0.95,
                   beta = 2,
                   nu = 3,
                   sigquant = 0.9,
                   kappa = 2,
                   numcut = 100L,
                   usequants = FALSE,
                   m = 20,
                   varimportance = TRUE,
                   hier_prior_bool = TRUE,
                   specify_variables = NULL,
                   diagnostic = TRUE
) {

  # Alias: accept y_mat for backward compatibility
  y_mat <- y_train

  # Convert matrix inputs to data.frame
  if(is.matrix(x_train)) x_train <- as.data.frame(x_train)
  if(is.data.frame(y_mat)) y_mat <- as.matrix(y_mat)

  # Handling error heading
  if(n_mcmc<=n_burn){
    stop("Number of MCMC iterations must be greater than the number of burn-in samples.")
  }

  if(nrow(x_train)<numcut){
    warning("numcut is smaller than the number of rows of x_train, numcut was re-defined as the nrow(x_train)")
    numcut <- nrow(x_train)
  }

  if(varimportance && (NCOL(x_train)==1)){
    warning("varimportance is set to FALSE as there is only one predictor.")
    varimportance <- FALSE
  }

  if(!is.null(specify_variables) && (length(specify_variables) != NCOL(y_mat))){
    stop("specify_variables must be a list with one element per response column.")
  }

  # Build sv_bool / sv_matrix for specify_variables
  if(is.null(specify_variables)){
    sv_bool <- FALSE
    sv_matrix <- matrix(1L, nrow = NCOL(y_mat), ncol = NCOL(x_train))
  } else {
    sv_bool <- TRUE
    sv_matrix <- matrix(0L, nrow = NCOL(y_mat), ncol = NCOL(x_train))
    for(i in seq_len(NCOL(y_mat))){
      sv_matrix[i, specify_variables[[i]]] <- 1L
    }
  }

  # Handle NULL x_test: internally use first 2 rows of x_train and strip at the end
  if(is.null(x_test)){
    x_test <- x_train[1:2, , drop = FALSE]
    null_x_test <- TRUE
    fit_test <- FALSE
  } else {
    null_x_test <- FALSE
    if(is.matrix(x_test)) x_test <- as.data.frame(x_test)
    fit_test <- TRUE
  }

  if(is.vector(x_train) || is.vector(x_test)){
    stop("x_train and x_test must be either a matrix or data.frame.")
  }

  if(!is.data.frame(x_train) || !is.data.frame(x_test)){
    stop("Insert valid data.frame for both data and xnew.")
  }

  # Scale y_set as true as default
  scale_y <- TRUE

  # Changing to a classification model
  if(length(unique(c(y_mat[complete.cases(y_mat)])))==2){
    class_model <- TRUE
    scale_y <- FALSE
  } else {
    class_model <- FALSE
  }

  if(class_model){
    if(!identical(sort(round(unique(c(y_mat)))),c(0,1))){
      stop(" Use the y as c(0,1) vector for the classification model.")
    }
  }

  if(class_model & scale_y){
    stop("Classification model should not scale y.")
  }

  # Getting the valid
  dummy_x <- base_dummyVars(x_train)


  # Create a data.frame aux
  initial_rank <- FALSE

  # Create a list
  if(length(dummy_x$facVars)!=0 & initial_rank){

    # Selected rank_var categorical
    rank_var <- 1

    for(i in 1:length(dummy_x$facVars)){
      # See if the levels of the test and train matches
      if(fit_test){
        if(!all(levels(x_train[[dummy_x$facVars[i]]])==levels(x_test[[dummy_x$facVars[i]]]))){
          levels(x_test[[dummy_x$facVars[[i]]]]) <- levels(x_train[[dummy_x$facVars[[i]]]])
        }
      }

      df_aux <- data.frame( x = x_train[,dummy_x$facVars[i]], y = y_mat[,rank_var])
      formula_aux <- stats::aggregate(y~x,df_aux,mean)
      formula_aux$y <- rank(formula_aux$y)
      x_train[[dummy_x$facVars[i]]] <- as.numeric(factor(x_train[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

      # Doing the same for the test set
      if(fit_test){
        x_test[[dummy_x$facVars[i]]] <- as.numeric(factor(x_test[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1
      }

      categorical_indicators <- numeric(ncol(x_train))

    }

  } else if ((length(dummy_x$facVars)!=0) && isFALSE(initial_rank)) {

    for(i in 1:length(dummy_x$facVars)){

      x_train[[dummy_x$facVars[i]]] <- as.numeric(x_train[[dummy_x$facVars[i]]])
      x_test[[dummy_x$facVars[i]]] <- as.numeric(x_test[[dummy_x$facVars[i]]])

      categorical_indicators <- numeric(ncol(x_train))
      categorical_indicators[which(colnames(x_train) %in% dummy_x$facVars)] <- 1

    }

  } else {

    categorical_indicators <- numeric(ncol(x_train))

  }

  # Getting the train and test set
  x_train_scale <- as.matrix(x_train)

  # Scaling x
  x_min <- apply(as.matrix(x_train_scale),2,min)
  x_max <- apply(as.matrix(x_train_scale),2,max)

  # Storing the original
  x_train_original <- x_train

  # Always scale x_test (even when fit_test = FALSE the dummy slice is still passed to C++)
  x_test_scale <- as.matrix(x_test)
  if(fit_test){
    x_test_original <- x_test
  }

  # Normalising all the columns
  for(i in 1:ncol(x_train)){
    x_train_scale[,i] <- normalize_covariates_bart(y = x_train_scale[,i],a = x_min[i], b = x_max[i])
    x_test_scale[,i]  <- normalize_covariates_bart(y = x_test_scale[,i], a = x_min[i], b = x_max[i])
  }

  # Creating the numcuts matrix of splitting rules
  xcut_m <- matrix(NA,nrow = numcut,ncol = ncol(x_train_scale))
  for(i in 1:ncol(x_train_scale)){

    if(nrow(x_train_scale)<numcut){
      xcut_m[,i] <- sort(x_train_scale[,i])
    } else {
      xcut_m[,i] <- seq(min(x_train_scale[,i]),
                        max(x_train_scale[,i]),
                        length.out = numcut+2)[-c(1,numcut+2)]
    }
  }


  # Scaling the y
  min_y <- apply(y_mat,2,min,na.rm = TRUE)
  max_y <- apply(y_mat,2,max,na.rm = TRUE)

  # Scaling the data
  if(scale_y){
    y_mat_scale <- y_mat
    for(n_col in 1:NCOL(y_mat)){
      y_mat_scale[,n_col] <- normalize_bart(y = y_mat[,n_col],a = min_y[n_col],b = max_y[n_col])
    }
  } else {
    y_mat_scale <- y_mat
  }
  # Getting the min and max for each column
  min_x <- apply(x_train_scale,2,min)
  max_x <- apply(x_train_scale, 2, max)


  # Defining tau_mu_j
  if(class_model){
    tau_mu_j <- rep((n_tree*(kappa^2))/9.0,NCOL(y_mat))
  } else {
    if(scale_y){
      tau_mu_j <- rep((4*n_tree*(kappa^2)),NCOL(y_mat))
    } else {
      tau_mu_j <- (4*n_tree*(kappa^2))/((max_y-min_y)^2)
    }

  }

  # Getting sigma
  sigma_mu_j <- tau_mu_j^(-1/2)

  # =========
  # Calculating prior for the \tau in case of regression and skipping it
  #in terms of classification
  # =========

  if(class_model){
    # For probit suBART, correlation matrix prior starts at identity (diag = 1)
    Sigma_init <- diag(1, nrow = NCOL(y_mat))
    mu_init <- apply(y_mat,2,mean,na.rm = TRUE)
    # S_0_wish: identity-scaled IW prior for correlation matrix
    S_0_wish <- diag(1, nrow = NCOL(y_mat))
    A_j <- rep(0, NCOL(y_mat))

    df <- nu + ncol(y_mat_scale) - 1
    # No extra parameters are need to calculate for the class model
  } else {
    # Getting the naive sigma value
    if(nrow(x_train_scale) > ncol(y_mat_scale)){
      nsigma <- apply(y_mat_scale, 2, function(Y){naive_sigma(x = x_train_scale,y = Y)})
    } else {
      nsigma <- apply(y_mat_scale, 2, function(Y){stats::sd(Y,na.rm = TRUE)})
    }
    # Define the ensity function
    phalft <- function(x, A, nu){
      return(2 * stats::pt(x/A, nu) - 1)
    }

    # Define parameters
    df <- nu + ncol(y_mat_scale) - 1

    # Selecting hypera-parmeters for the t-distribution case
    if(hier_prior_bool){
      A_j <- numeric()

      for(i in 1:length(nsigma)){
        # Calculating lambda
        A_j[i] <- stats::optim(par = 0.01, f = function(A){(sigquant - phalft(nsigma[i], A, nu))^2},
                               method = "Brent",lower = 0.00001,upper = 100)$par
      }

      # Calculating lambda
      qchi <- stats::qchisq(p = 1-sigquant,df = df,lower.tail = 1,ncp = 0)
      lambda <- (nsigma*nsigma*qchi)/df
      rate_tau <- (lambda*df)/2

      S_0_wish <- if(ncol(y_mat)!=1){
        2*df*diag(c(rate_tau))
      } else {
        matrix(2*df*c(rate_tau),ncol = 1,nrow = 1)
      }



    } else {
      A_j <- numeric()
      for(i in 1:length(nsigma)){
        A_j[i] <- stats::optim(par = 0.01, f = function(A){(sigquant - stats::pgamma(q = 1/(nsigma[i]^2),
                                                                                     shape = nu/2, rate = A/2,
                                                                                     lower.tail = FALSE))^2},
                               method = "Brent",lower = 0.00001,upper = 100)$par
      }

      S_0_wish <- if(ncol(y_mat)!=1){
        diag(A_j)
      } else {
        matrix(A_j,ncol = 1,nrow = 1)
      }
    }


    # Call the bart function
    Sigma_init <- if(ncol(y_mat)!=1){
      diag(nsigma^2)
    } else {
      matrix(nsigma^2,ncol = 1,nrow = 1)
    }

    mu_init <- apply(y_mat_scale,2,mean,na.rm = TRUE)
  }



  # Generating the BART obj
  if(class_model){

    if(ncol(y_mat_scale)==1 ){ # For the univariate case
      na_boolean <- FALSE
      y_bin_vec <- as.integer(c(y_mat_scale))
      bart_obj <- cppsubart_univariate_CLASS(x_train_scale,
                                             y_bin_vec,
                                             x_test_scale,
                                             xcut_m,
                                             n_tree,
                                             node_min_size,
                                             n_mcmc,
                                             n_burn,
                                             mu_init,
                                             sigma_mu_j,
                                             alpha, beta,
                                             categorical_indicators,
                                             fit_test,
                                             varimportance,
                                             sv_bool,
                                             sv_matrix)
    } else {

      if(any(is.na(y_mat_scale))){
        y_mat_scale[is.na(y_mat_scale)] <- 0L
        na_boolean <- TRUE
      } else {
        na_boolean <- FALSE
      }

      y_bin_mat <- matrix(as.integer(y_mat_scale), nrow = nrow(y_mat_scale), ncol = ncol(y_mat_scale))
      bart_obj <- cppsubart_CLASS(x_train_scale,
                                  y_bin_mat,
                                  x_test_scale,
                                  xcut_m,
                                  n_tree,
                                  node_min_size,
                                  n_mcmc,
                                  n_burn,
                                  Sigma_init,
                                  mu_init,
                                  sigma_mu_j,
                                  alpha, beta, nu,
                                  S_0_wish,
                                  A_j,
                                  m,
                                  categorical_indicators,
                                  fit_test,
                                  varimportance,
                                  sv_bool,
                                  sv_matrix)
    }

  } else {

    if(ncol(y_mat_scale)==1){ # For the univariate case

      na_boolean <- FALSE

      bart_obj <- cppsubart_univariate(x_train_scale,
                                       c(y_mat_scale),
                                     x_test_scale,
                                     xcut_m,
                                     n_tree,
                                     node_min_size,
                                     n_mcmc,
                                     n_burn,
                                     Sigma_init,
                                     mu_init,
                                     sigma_mu_j,
                                     alpha,beta,nu,
                                     S_0_wish,
                                     A_j,
                                     hier_prior_bool,
                                     categorical_indicators,
                                     fit_test,
                                     varimportance,
                                     sv_bool,
                                     sv_matrix)
    } else {


      if(any(is.na(y_mat_scale))){

        number_na <- apply(y_mat_scale,2,function(x){sum(is.na(x),na.rm = TRUE)})
        na_indicators <- ifelse(is.na(y_mat_scale),1,0)
        y_mat_scale[is.na(y_mat_scale)] <- 0

        na_boolean <- TRUE

        bart_obj <- if(ncol(y_mat_scale)==2){
          cppsubart_missing_2d(x_train_scale,
                                    y_mat_scale,
                                    number_na,
                                    na_indicators,
                                    x_test_scale,
                                    xcut_m,
                                    n_tree,
                                    node_min_size,
                                    n_mcmc,
                                    n_burn,
                                    Sigma_init,
                                    mu_init,
                                    sigma_mu_j,
                                    alpha,beta,nu,
                                    S_0_wish,
                                    A_j,
                                    hier_prior_bool,
                                    categorical_indicators,
                                    fit_test,
                                    varimportance,
                                    sv_bool,
                                    sv_matrix)
        } else {
          cppsubart_missing(x_train_scale,
                            y_mat_scale,
                            number_na,
                            na_indicators,
                            x_test_scale,
                            xcut_m,
                            n_tree,
                            node_min_size,
                            n_mcmc,
                            n_burn,
                            Sigma_init,
                            mu_init,
                            sigma_mu_j,
                            alpha,beta,nu,
                            S_0_wish,
                            A_j,
                            hier_prior_bool,
                            categorical_indicators,
                            fit_test,
                            varimportance,
                            sv_bool,
                            sv_matrix)
        }

      } else {
        na_boolean <- FALSE

        bart_obj <-if(ncol(y_mat_scale)==2){
          cppsubart_2d(x_train_scale,
                       y_mat_scale,
                       x_test_scale,
                       xcut_m,
                       n_tree,
                       node_min_size,
                       n_mcmc,
                       n_burn,
                       Sigma_init,
                       mu_init,
                       sigma_mu_j,
                       alpha,beta,nu,
                       S_0_wish,
                       A_j,
                       hier_prior_bool,
                       categorical_indicators,
                       fit_test,
                       varimportance,
                       sv_bool,
                       sv_matrix)
        } else {
          cppsubart(x_train_scale,
                    y_mat_scale,
                    x_test_scale,
                    xcut_m,
                    n_tree,
                    node_min_size,
                    n_mcmc,
                    n_burn,
                    Sigma_init,
                    mu_init,
                    sigma_mu_j,
                    alpha,beta,nu,
                    S_0_wish,
                    A_j,
                    hier_prior_bool,
                    categorical_indicators,
                    fit_test,
                    varimportance,
                    sv_bool,
                    sv_matrix)
        }

      }
    }

  }


  # Returning the main components from the model
  y_train_post <- bart_obj[[1]]
  y_test_post <- bart_obj[[2]]

  # For classification, extract Sigma_post only for multivariate case
  if (class_model) {
    if (ncol(y_mat) == 1) {
      # cppsubart_univariate_CLASS: [[1]] y_train, [[2]] y_test, [[3]] var_imp
      Sigma_post     <- NULL
      all_Sigma_post <- NULL
      y_mat_post     <- NULL
      var_importance_raw <- if (varimportance) bart_obj[[3]] else NULL
    } else {
      # cppsubart_CLASS: [[1]] y_train, [[2]] y_test, [[3]] Sigma, [[4]] all_Sigma, [[5]] var_imp
      Sigma_post     <- bart_obj[[3]]
      all_Sigma_post <- bart_obj[[4]]
      y_mat_post     <- NULL
      var_importance_raw <- if (varimportance) bart_obj[[5]] else NULL
    }
    var_importance <- if (varimportance && !is.null(var_importance_raw)) {
      if (is.matrix(var_importance_raw)) colMeans(var_importance_raw) else apply(var_importance_raw, c(1, 2), mean)
    } else NULL
  } else {
    Sigma_post     <- bart_obj[[3]]
    all_Sigma_post <- bart_obj[[4]]
    if (na_boolean) {
      y_mat_post         <- bart_obj[[5]]
      var_importance_raw <- if (varimportance) bart_obj[[6]] else NULL
    } else {
      y_mat_post         <- NULL
      var_importance_raw <- if (varimportance) bart_obj[[5]] else NULL
    }
    var_importance <- if (varimportance && !is.null(var_importance_raw)) {
      if (is.matrix(var_importance_raw)) colMeans(var_importance_raw) else apply(var_importance_raw, c(1, 2), mean)
    } else NULL
  }


  Sigma_scale <- if(ncol(y_mat)!=1){
    diag((max_y-min_y))
  } else {
    matrix((max_y-min_y),ncol=1,nrow=1)
  }

  # The outcomes are different depending on the
  if(ncol(y_mat)==1){

    # Getting the mean values for the Sigma and \y_hat and \y_hat_test
    y_train_for <- numeric(nrow(y_mat))

    if(fit_test){
      y_test_for <- numeric(nrow(x_test))
    }


    Sigma_scale <- c(Sigma_scale)

    if(scale_y && !class_model){

      # Re-scaling Sigma_all, important to cover convergence issues.
      Sigma_post <- Sigma_post * (Sigma_scale)^2

      y_train_post <- unnormalize_bart_matrix(y_train_post,min_y,max_y)
      y_test_post <- unnormalize_bart(y_test_post,min_y,max_y)

    } else if(scale_y && class_model){

      y_train_post <- unnormalize_bart_matrix(y_train_post,min_y,max_y)
      y_test_post <- unnormalize_bart(y_test_post,min_y,max_y)

    }

    Sigma_post_mean <- if(!class_model) mean(Sigma_post) else NULL
    y_hat_mean <- apply(y_train_post,1,mean)

    y_test_mean <- apply(y_test_post,1,mean)


    sigmas_mean <- if(!class_model) sqrt(Sigma_post_mean) else NULL

    # Transforming to classification context

    # Getting the list of outcomes
    if(class_model){

      # No Sigma samples in probit-BART; skip ESS on Sigma
      ESS_val  <- NULL
      ESS_warn <- FALSE

      list_obj_ <- list(y_hat = y_train_post,
                        y_hat_test = y_test_post,
                        y_hat_mean = y_hat_mean,
                        y_hat_test_mean = y_test_mean,
                        y_hat_mean_class = ifelse(y_hat_mean > 0, 1L, 0L),
                        y_hat_test_mean_class = ifelse(apply(y_test_post, 1, mean) > 0, 1L, 0L),
                        var_importance = var_importance,
                        var_importance_raw = var_importance_raw,
                        prior = list(n_tree = n_tree,
                                     alpha = alpha,
                                     beta = beta,
                                     tau_mu_j = tau_mu_j,
                                     mu_init = mu_init),
                        mcmc = list(n_mcmc = n_mcmc,
                                    n_burn = n_burn),
                        data = list(x_train = x_train,
                                    y_mat = y_mat,
                                    x_test = x_test),
                        ESS = ESS_val)

      class(list_obj_) <- "subart-probit"

    } else {


      # Calculate the ESS for all parameters throw a warning if any of them is smaller than half of the MCMC samples
      if(diagnostic){

        diagnostic_bool = FALSE
        ESS_val <- ESS(x = Sigma_post)
        ESS_warn <- FALSE

        if(ESS_val<round((n_mcmc-n_burn)/2,digits = 0)){
          ESS_warn <- TRUE
        }

      } else {
        ESS_val <- NULL
      }

      if(ESS_warn){
        warning(paste0("A ESS less than ",round((n_mcmc-n_burn)/2,digits = 0)," was obtanied. Verify the traceplots and adjust the priors to improve the sampling."))
      }


      # Returning the data list
      data_list <- if(na_boolean){
        list(x_train = x_train,
             y_mat = y_mat,
             x_test = x_test,
             y_mat_post = y_mat_post)
      } else {
        list(x_train = x_train,
             y_mat = y_mat,
             x_test = x_test)
      }

      list_obj_ <- list(y_hat = y_train_post,
                        y_hat_test = y_test_post,
                        y_hat_mean = y_hat_mean,
                        y_hat_test_mean = y_test_mean,
                        Sigma_post = Sigma_post,
                        Sigma_post_mean = Sigma_post_mean,
                        sigmas_mean = sigmas_mean,
                        all_Sigma_post = all_Sigma_post,
                        var_importance = var_importance,
                        prior = list(n_tree = n_tree,
                                     alpha = alpha,
                                     beta = beta,
                                     tau_mu_j = tau_mu_j,
                                     df = df,
                                     A_j = A_j,
                                     mu_init = mu_init),
                        mcmc = list(n_mcmc = n_mcmc,
                                    n_burn = n_burn),
                        data = data_list,
                        ESS = ESS_val)

      class(list_obj_) <- "subart"
    }

  } else { ## ELSE FOR THE MULTI-DIMENSIONAL OUTCOME

    # Getting the mean values for the Sigma and \y_hat and \y_hat_test
    Sigma_for <- matrix(0,nrow = nrow(Sigma_post), ncol = ncol(Sigma_post))
    y_train_for <- matrix(0,nrow = nrow(y_mat),ncol = ncol(y_mat))

    if(fit_test){
      y_test_for <- matrix(0,nrow = nrow(x_test),ncol = ncol(y_mat))
    }

    if(scale_y){

      # Re-scaling Sigma_all, important to cover convergence issues.
      for(k in 1:(dim(all_Sigma_post)[3])){
        all_Sigma_post[,,k] <- crossprod(Sigma_scale,tcrossprod(all_Sigma_post[,,k],Sigma_scale))

      }

      for(i in 1:(dim(Sigma_post)[3])){
        Sigma_post[,,i] <- crossprod(Sigma_scale,tcrossprod(Sigma_post[,,i],Sigma_scale))
        Sigma_for <- Sigma_for +  Sigma_post[,,i]
        for( jj in 1:NCOL(y_mat)){
          y_train_for[,jj] <- y_train_for[,jj] + unnormalize_bart(z = y_train_post[,jj,i],a = min_y[jj],b = max_y[jj])

          if(fit_test){
            y_test_for[,jj] <- y_test_for[,jj] +  unnormalize_bart(z = y_test_post[,jj,i],a = min_y[jj],b = max_y[jj])
            y_test_post[,jj,i] <-  unnormalize_bart(z = y_test_post[,jj,i],a = min_y[jj],b = max_y[jj])
          }

          y_train_post[,jj,i] <- unnormalize_bart(z = y_train_post[,jj,i],a = min_y[jj],b = max_y[jj])
          if(na_boolean){
            y_mat_post[,jj,i] <- unnormalize_bart(z = y_mat_post[,jj,i],a = min_y[jj],b = max_y[jj])
          }
        }
      }
    } else {
      for(i in 1:(dim(Sigma_post)[3])){
        Sigma_for <- Sigma_for + Sigma_post[,,i]
        y_train_for <- y_train_for +  y_train_post[,,i]
        if(fit_test){
          y_test_for <- y_test_for +  y_test_post[,,i]
        }

      }
    }


    Sigma_post_mean <- Sigma_for/dim(Sigma_post)[3]
    y_mat_mean <- y_train_for/dim(y_train_post)[3]

    y_mat_test_mean <- if(fit_test){
      y_test_for/dim(y_test_post)[3]
    } else {
      NULL
    }

    sigmas_mean <- sqrt(diag(Sigma_post_mean))

    # Transforming to classification context

    # Getting the list of outcomes
    if(class_model){

      # ESS on correlation matrix (Sigma already normalised to R in cppsubart_CLASS)
      if(diagnostic && !is.null(Sigma_post)){
        ESS_val <- matrix(NA, nrow = dim(Sigma_post)[1], ncol = dim(Sigma_post)[2])
        ESS_warn <- FALSE
        d_ <- dim(Sigma_post)[1]
        for(i in 1:d_){
          j = i
          while(j < d_){
            j = j + 1
            ESS_val[i,j] <- ESS_val[j,i] <- ESS(x = Sigma_post[i,j,])
            if(ESS_val[i,j] < round((n_mcmc-n_burn)/2, digits=0)){
              ESS_warn <- TRUE
            }
          }
        }
        if(ESS_warn){
          warning(paste0("A ESS less than ",round((n_mcmc-n_burn)/2,digits=0)," was obtained."))
        }
      } else {
        ESS_val <- NULL
      }

      list_obj_ <- list(y_hat = y_train_post,
                        y_hat_test = y_test_post,
                        y_hat_mean = y_mat_mean,
                        y_hat_test_mean = y_mat_test_mean,
                        y_hat_mean_class = apply(y_mat_mean,2,function(x){ifelse(x>0,1,0)}),
                        y_hat_test_mean_class = if(!is.null(y_mat_test_mean)) apply(y_mat_test_mean,2,function(x){ifelse(x>0,1,0)}) else NULL,
                        Sigma_post = Sigma_post,
                        Sigma_post_mean = Sigma_post_mean,
                        all_Sigma_post = all_Sigma_post,
                        var_importance = var_importance,
                        var_importance_raw = var_importance_raw,
                        prior = list(n_tree = n_tree,
                                     alpha = alpha,
                                     beta = beta,
                                     tau_mu_j = tau_mu_j,
                                     mu_init = mu_init),
                        mcmc = list(n_mcmc = n_mcmc,
                                    n_burn = n_burn),
                        data = list(x_train = x_train,
                                    y_mat = y_mat,
                                    x_test = x_test),
                        ESS = ESS_val)

      class(list_obj_) <- "subart-probit"

    } else {


      # Calculate the ESS for all parameters throw a warning if any of them is smaller than half of the MCMC samples
      if(diagnostic){

        diagnostic_bool = FALSE
        ESS_val <- matrix(NA, nrow = nrow(Sigma_post), ncol = ncol(Sigma_post))
        ESS_warn <- FALSE
        for(i in 1:nrow(Sigma_post)){
          ESS_val[i,i] <- ESS(x = sqrt(Sigma_post[i,i,]))
          j = i
          while(j < nrow(Sigma_post)){
            j = j+1
            ESS_val[i,j] <- ESS_val[j,i] <- ESS(x = Sigma_post[i,j,]/(sqrt(Sigma_post[i,i,])*sqrt(Sigma_post[j,j,])))
            if(ESS_val[i,j]<round((n_mcmc-n_burn)/2,digits = 0)){
              ESS_warn <- TRUE
            }
          }
        }

      } else {
        ESS_val <- NULL
      }

      if(ESS_warn){
        warning(paste0("A ESS less than ",round((n_mcmc-n_burn)/2,digits = 0)," was obtanied. Verify the traceplots and adjust the priors to improve the sampling."))
      }


      # Returning the data list
      data_list <- if(na_boolean){
        list(x_train = x_train,
             y_mat = y_mat,
             x_test = x_test,
             y_mat_post = y_mat_post)
      } else {
        list(x_train = x_train,
             y_mat = y_mat,
             x_test = x_test)
      }

      list_obj_ <- list(y_hat = y_train_post,
                        y_hat_test = y_test_post,
                        y_hat_mean = y_mat_mean,
                        y_hat_test_mean = y_mat_test_mean,
                        Sigma_post = Sigma_post,
                        Sigma_post_mean = Sigma_post_mean,
                        sigmas_mean = sigmas_mean,
                        all_Sigma_post = all_Sigma_post,
                        var_importance = var_importance,
                        prior = list(n_tree = n_tree,
                                     alpha = alpha,
                                     beta = beta,
                                     tau_mu_j = tau_mu_j,
                                     df = df,
                                     A_j = A_j,
                                     mu_init = mu_init),
                        mcmc = list(n_mcmc = n_mcmc,
                                    n_burn = n_burn),
                        data = data_list,
                        ESS = ESS_val)

      class(list_obj_) <- "subart"
    }

  }

  # Strip test predictions when x_test was originally NULL
  if(null_x_test){
    list_obj_$y_hat_test <- NULL
    list_obj_$y_hat_test_mean <- NULL
    if(!is.null(list_obj_$y_hat_test_mean_class)) list_obj_$y_hat_test_mean_class <- NULL
    list_obj_$data$x_test <- NULL
  }

  # Return the list with all objects and parameters
  return(list_obj_)
}
