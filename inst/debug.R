# devtools::load_all()
Rcpp::sourceCpp("src/subart.cpp")
n <- 100
d <- 3

x_test <- x_train <- matrix(rnorm(n = n*d),ncol=d)
y_mat <- matrix(rnorm(n = n*d),ncol=d)
Sigma_init <- matrix(1:(d^2),ncol = d,nrow=d)
n_tree <- 10
node_min_size <- 5
n_mcmc <- 100
n_burn <- 50

mu_init <- numeric(n)
sigma_mu <- rep(1,d)

alpha <- 0.95
beta <- 2.0
nu <- 2.0
S_0_wish <- Sigma_init
A_j_vec <- sigma_mu

update_Sigma <- TRUE
var_selection_bool <- TRUE
sv_bool <- TRUE
hier_prior_bool <- TRUE

sv_matrix <- Sigma_init

categorical_indicators <- rep(1,d)

cppsubart(x_train,
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
          categorical_indicators)
