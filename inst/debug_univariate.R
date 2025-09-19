rm(list=ls())
devtools::load_all()
set.seed(42)

n <- 500
d <- 2

# # Simulate from a function
data <- list()
data$x <- runif(n,-pi,pi)
x_train <- as.matrix(data$x)
x_test <- x_train
y_mat <- as.matrix(sin(data$x) + rnorm(n = n,sd = 0.1))
# y_mat <- apply(y_mat,2,scale)

# x_test <- x_train <- matrix(runif(n = n*d,min = -pi,max = pi),ncol=d)
# y_mat <- matrix(rnorm(n = n*d),ncol=d)
# y_mat[,1] <- sin(x_train[,1]) #+ rnorm(n = n,sd = 0.25)
# y_mat[,2] <- cos(x_train[,2]) #+ rnorm(n = n,sd = 0.1)
# Sigma_true <- matrix(c(1.0,7.5,7.5,100),nrow=2,ncol = 2)
# err <- mvtnorm::rmvnorm(n = n,sigma = Sigma_true)
# y_mat[,1] <- y_mat[,1] + err[1,1]
# y_mat[,2] <- y_mat[,2] + err[2,2]

n_tree = 200
node_min_size = 2
n_mcmc = 2000
n_burn = 500
alpha = 0.95
beta = 2
nu = 3
sigquant = 0.9
kappa = 2
numcut = 100L # Defining the grid of split rules
usequants = TRUE
m = 20 # Degrees of freed for the classification setting.
hier_prior_sigma <- FALSE
diagnostic = TRUE

y_mat = y_mat[,1, drop = FALSE]
x_train <- data.frame(x_train)



init_time <- Sys.time()
subart_mod <- subart(x_train = x_train,
                     y_mat = y_mat,
                     x_test = x_train,
                     n_tree = n_tree,
                     n_mcmc = n_mcmc,
                     n_burn = n_burn)
end_time <- Sys.time() - init_time

plot(x_train$x_train,apply(subart_mod$y_hat,1,mean))
points(x_train$x_train,sin(x_train$x_train),col= 'blue')

library(dbarts)

dbarts_mod <- bart(x.train = x_train,y.train = y_mat,x.test = x_train,
                   nskip = n_burn,ndpost = n_mcmc-n_burn,ntree = n_tree)

plot(dbarts_mod$sigma, type = 'l', col = 'red')
lines(sqrt(subart_mod$Sigma_post), type = 'l', col = 'blue')
