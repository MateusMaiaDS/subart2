source("R/other_functions.R")
Rcpp::sourceCpp("src/subart.cpp")
n <- 1000
d <- 2
set.seed(42)
x_test <- x_train <- matrix(runif(n = n*d,min = -pi,max = pi),ncol=d)
y_mat <- matrix(rnorm(n = n*d),ncol=d)
y_mat[,1] <- 2*sin(x_train[,1]) + rnorm(n = n,sd = 0.25)
y_mat[,2] <- 2*cos(x_train[,2]) + rnorm(n = n,sd = 0.1)

Sigma_init <- diag(ncol = d,nrow=d)
n_tree <- 100
node_min_size <- 5
n_mcmc <- 2000
n_burn <- 500


alpha <- 0.95
beta <- 2.0
kappa <- 2
nu <- 2.0
sigquant <- 0.9
S_0_wish <- Sigma_init
numcut <- 100

update_Sigma <- TRUE
var_selection_bool <- TRUE
sv_bool <- TRUE
hier_prior_bool <- TRUE

# Handling error heading
if(n_mcmc<=n_burn){
  stop("Number of MCMC iterations must be greater than the number of burn-in samples.")
}

if(is.vector(x_train)|| is.vector(x_test)){
  stop("x_train and x_test must be either a matrix or data.frame.")
}

if(nrow(x_train)<numcut){
  warning("numcut is smaller than the number of rows of x_train, numcut was re-defined as the nrow(x_train)")
  numcut <- nrow(x_train)
}


## End error handling

# Defining initial paramters that are no longer up to the user to define
scale_y <- TRUE # If the target variable Y gonna be scaled or not
Sigma_init <- NULL # The initial value for the \Sigma matrix initialisation
update_Sigma <- TRUE # It always TRUE was only useful to test exeperiments
conditional_bool <- TRUE # Again is always true
tn_sampler <- FALSE # Define if the truncated-normal sampler gonna be used or not


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


# Avoiding error of this kind
if(class_model & scale_y){
  stop("Classificaton model should not scale y.")
}

# # Verifying if it's been using a y_mat matrix
# if(NCOL(y_mat)<2 & class_model){
#      stop("Insert a valid multivariate response for a classification task. ")
# }


# Getting the valid

# Getting the train and test set
x_train_scale <- as.matrix(x_train)
x_test_scale <- as.matrix(x_test)

# Scaling x
x_min <- apply(as.matrix(x_train_scale),2,min)
x_max <- apply(as.matrix(x_train_scale),2,max)

# Storing the original
x_train_original <- x_train
x_test_original <- x_test


# Normalising all the columns
for(i in 1:ncol(x_train)){
  x_train_scale[,i] <- normalize_covariates_bart(y = x_train_scale[,i],a = x_min[i], b = x_max[i])
  x_test_scale[,i] <- normalize_covariates_bart(y = x_test_scale[,i],a = x_min[i], b = x_max[i])
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
sigma_mu <- sigma_mu_j
# =========
# Calculating prior for the \tau in case of regression and skipping it
#in terms of classification
# =========

if(class_model){
  # Call the bart function
  if(is.null(Sigma_init) || NCOL(y_mat)==1){
    Sigma_init <- diag(1,nrow = NCOL(y_mat))
  }
  mu_init <- apply(y_mat,2,mean,na.rm = TRUE)

  df <- nu + ncol(y_mat_scale) - 1
  # No extra parameters are need to calculate for the class model
} else {
  # Getting the naive sigma value
  # if(nrow(x_train_scale) > ncol(y_mat_scale)){
    # nsigma <- apply(y_mat_scale, 2, function(Y){naive_sigma(x = x_train_scale,y = Y)})
  # } else {
    nsigma <- apply(y_mat_scale, 2, function(Y){stats::sd(Y,na.rm = TRUE)})
  # }
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
  if(is.null(Sigma_init)){
    Sigma_init <- if(ncol(y_mat)!=1){
      diag(nsigma^2)
    } else {
      matrix(nsigma^2,ncol = 1,nrow = 1)
    }
  }

  mu_init <- apply(y_mat_scale,2,mean,na.rm = TRUE)
}

A_j_vec <- A_j
sv_matrix <- matrix(1,rep(ncol(y_mat)))
categorical_indicators <- rep(1,ncol(y_mat))
init <- Sys.time()

subart_cpp <- cppsubart(x_train,
          y_mat,
          x_test,
          x_train,
          n_tree,
          node_min_size,
          n_mcmc,
          n_burn,
          Sigma_init,
          mu_init,
          sigma_mu,
          alpha,
          beta,
          nu,
          S_0_wish,
          A_j_vec,
          update_Sigma,
          var_selection_bool,
          sv_bool,
          hier_prior_bool,
          sv_matrix,
          categorical_indicators,
          FALSE)
end <- Sys.time()-init
end

par(mfrow=c(2,1))
plot(sqrt(subart_cpp[[3]][1,1,]), type = 'l',xlab = "MCMC", ylab = expression(sigma[1,1]))
plot(sqrt(subart_cpp[[3]][2,2,]), type = 'l',xlab = "MCMC", ylab = expression(sigma[2,2]))
