rm(list=ls())
devtools::load_all()
set.seed(42)

n <- 250
d <- 2

# # Simulate from a function
data <- sim_mvn_friedman1(n = n,
                          p = 5,mvn_dim = d,Sigma = diag(nrow = 2))
x_train <- data$x
x_test <- x_train
y_mat <- data$y
# y_mat <- apply(y_mat,2,scale)

# x_test <- x_train <- matrix(runif(n = n*d,min = -pi,max = pi),ncol=d)
# y_mat <- matrix(rnorm(n = n*d),ncol=d)
# y_mat[,1] <- sin(x_train[,1]) #+ rnorm(n = n,sd = 0.25)
# y_mat[,2] <- cos(x_train[,2]) #+ rnorm(n = n,sd = 0.1)
# Sigma_true <- matrix(c(1.0,7.5,7.5,100),nrow=2,ncol = 2)
# err <- mvtnorm::rmvnorm(n = n,sigma = Sigma_true)
# y_mat[,1] <- y_mat[,1] + err[1,1]
# y_mat[,2] <- y_mat[,2] + err[2,2]

n_tree = 100
node_min_size = 2
n_mcmc = 2000
n_burn = 1000
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


x_train <- data.frame(x_train)
# subart_mod <- subart2::subart(x_train = x_train,
#                               y_mat = y_mat,
#                               x_test = x_train,
#                               n_tree = n_tree,
#                               n_mcmc = n_mcmc,
#                               n_burn = n_burn)

par(mfrow=c(2,1))

plot(subart_mod$y_hat_mean[,1],y_mat[,1],
     xlab = "y.1.pred",ylab = "y.1.obs")
plot(subart_mod$y_hat_mean[,2],y_mat[,2],
     xlab = "y.2.pred",ylab = "y.2.obs")

par(mfrow=c(2,1))
plot(sqrt(subart_mod$Sigma_post[1,1,]), type = "l",xlab = "mcmc", ylab = expression(sigma[11]))
# abline(h = sqrt(data$Sigma[1,1]), col = "blue",lty = 2)
plot(sqrt(subart_mod$Sigma_post[2,2,]), type = "l",xlab = "mcmc", ylab = expression(sigma[22]))
# abline(h = sqrt(data$Sigma[2,2]), col = "blue",lty = 2)

# original_subart <- subart::subart(x_train = x_train,
#                                   y_mat = y_mat,x_test = x_train,
#                                   n_tree = 100,n_mcmc = 1000,
#                                   n_burn = 500)
# #
# #
# subart_mod <- original_subart
plot(subart_mod$y_hat_mean[,1],data$y[,1],
     xlab = "y.1.pred",ylab = "y.1.obs")
plot(subart_mod$y_hat_mean[,2],data$y[,2],
     xlab = "y.2.pred",ylab = "y.2.obs")

par(mfrow=c(2,1))
plot(sqrt(subart_mod$Sigma_post[1,1,]), type = "l",xlab = "mcmc", ylab = expression(sigma[11]))
plot(sqrt(subart_mod$Sigma_post[2,2,]), type = "l",xlab = "mcmc", ylab = expression(sigma[11]))


dbart_mod <- dbarts::bart(x.train = x_train,ntree = n_tree,
                          y.train = y_mat[,1],proposalprobs = c("birth_death" = 0.99,
                                                                                  "change" = 0.01,
                                                                                  "swap" = 0.0,
                                                                                  "birth" = 0.5))
dbart_mod_two <- dbarts::bart(x.train = x_train,ntree = n_tree,
                              y.train = y_mat[,2],proposalprobs = c("birth_death" = 0.99,
                                                                                      "change" = 0.01,
                                                                                      "swap" = 0.0,
                                                                                      "birth" = 0.5))
par(mfrow=c(2,2))
plot(dbart_mod$yhat.train.mean,y_mat[,1],xlab = "dbart_pred",ylab = "y_obs")
plot(dbart_mod$yhat.train.mean,subart_mod$y_hat_mean[,1],xlab = "dbart_pred",ylab = "subart")


plot(dbart_mod_two$yhat.train.mean,y_mat[,2],xlab = "dbart_pred",ylab = "y_obs")
plot(dbart_mod_two$yhat.train.mean,subart_mod$y_hat_mean[,2],xlab = "dbart_pred",ylab = "subart")


bart_mod <- BART::gbart(x.train = x_train,y.train = y_mat[,1],ntree = n_tree)
plot(bart_mod$yhat.train.mean,y_mat[,1])
plot(bart_mod$yhat.train.mean,
     dbart_mod$yhat.train.mean)

plot(bart_mod$yhat.train.mean,
     subart_mod$y_hat_mean[,1])

# plot(x_train$X1,dbart_mod$yhat.train.mean,main = "dbart")
# plot(x_train$X1,subart_mod$y_hat_mean[,1],main = "subart")

# plot(x_train$X2,dbart_mod_two$yhat.train.mean,main = "dbart")
# plot(x_train$X2,subart_mod$y_hat_mean[,2],main = "subart")

# par(mfrow=c(1,1))
# plot(dbart_mod$sigma, type = 'l')
# lines(sqrt(subart_mod$Sigma_post[1,1,]), type = 'l', col = 'blue')


partial_dependance_plot(variable_index = 3,n_points = 10,use_quantiles = TRUE,x_train = x_train,y_train = y_mat)
