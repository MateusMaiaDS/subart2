#include "subart_classes.h"
#include <RcppArmadillo.h>

// Initialising a node
Node::Node(){
  isLeaf = true;
  isRoot = true;
  left = NULL;
  right = NULL;
  parent = NULL;

  lower = 0.0;
  upper = 1.0;
  mu = 0.0;
  n_leaf = 0;
  n_leaf_test = 0;
  log_likelihood = 0.0;
  depth = 0;
}

bool Node::isLeft() {
  if(parent == NULL) return true;
  return (this == parent->left);
}

Node::~Node() {
  if(!isLeaf){
    delete left;
    delete right;
  }
}

// Initialising the model Param
modelParam::modelParam(arma::mat x_train_,
                       arma::mat y_mat_,
                       arma::mat x_test_,
                       arma::mat x_cut_,
                       unsigned int n_tree_,
                       unsigned int node_min_size_,
                       double alpha_,
                       double beta_,
                       double nu_,
                       arma::vec sigma_mu_,
                       arma::mat Sigma_,
                       arma::mat S_0_wish_,
                       arma::vec A_j_vec_,
                       double n_mcmc_,
                       double n_burn_,
                       arma::uvec categorical_indicators_,
                       bool fit_test_){


  // Assign the variables
  x_train = x_train_;
  y_mat = y_mat_;
  x_test = x_test_;
  xcut = x_cut_;

  n = x_train_.n_rows; // Converting uword to unsigned int; see if isn't a problme in the future
  n_test = x_test_.n_rows; // Converting uword to unsigned int; see if isn't a problme in the future
  d = y_mat.n_cols;

  p = x_train.n_cols;

  init_train_index = arma::uvec(n);
  init_test_index = arma::uvec(n_test);

  for(unsigned int i = 0; i<n; i++){
    init_train_index[i] = i;
  }

  for(unsigned int i = 0; i<n_test; i++){
    init_test_index[i] = i;
  }

  n_tree = n_tree_;
  node_min_size = node_min_size_;
  alpha = alpha_;
  beta = beta_;
  nu = nu_;
  sigma_mu = sigma_mu_;

  Sigma = Sigma_;
  S_0_wish = S_0_wish_;
  A_j_vec = A_j_vec_;
  a_j_vec = arma::vec(d);
  n_mcmc = n_mcmc_;
  n_burn = n_burn_;

  sigma_mu_j = arma::vec(d,arma::fill::none);
  sigma_mu_j_sq = arma::vec(d,arma::fill::none);


  // Generating the elements for the correlation matrix
  R = Sigma_;
  D = arma::mat(d,d); // There is more efficient way to declare the diagonal matrix without using arma::eye?

  for(arma::uword i = 0; i<d;i++){
    D.at(i,i) = 1.0;
    sigma_mu_j[i] = sigma_mu_[i];
    sigma_mu_j_sq[i] = sigma_mu_j[i]*sigma_mu_j[i]; // This operation can be done when constructing it
  }
  // Grow acceptation ratio
  for(unsigned int i = 0; i<3;i++){
    move_proposal[i] = 0;
    move_acceptance[i] = 0;
  }

  categorical_indicators = categorical_indicators_;

  unsigned int count_cat_indicator = 0;
  for(unsigned int i=0; i < categorical_indicators.n_elem; i++){
    count_cat_indicator++;
  }

  categorical_indicators_bool = count_cat_indicator==0 ? false: true;

  // Decide to wether update the fit_test or not;
  fit_test = fit_test_;

}

// Initialising the model Param
modelParam_uni::modelParam_uni(arma::mat x_train_,
                       arma::vec y_,
                       arma::mat x_test_,
                       arma::mat x_cut_,
                       unsigned int n_tree_,
                       unsigned int node_min_size_,
                       double alpha_,
                       double beta_,
                       double nu_,
                       double sigma_mu_,
                       double sigma_,
                       double S_0_,
                       double A_j_,
                       double n_mcmc_,
                       double n_burn_,
                       arma::uvec categorical_indicators_,
                       bool fit_test_){


  // Assign the variables
  x_train = x_train_;
  y = y_;
  x_test = x_test_;
  xcut = x_cut_;

  n = x_train_.n_rows; // Converting uword to unsigned int; see if isn't a problem in the future
  n_test = x_test_.n_rows; // Converting uword to unsigned int; see if isn't a problem in the future

  p = x_train.n_cols;

  p = x_train.n_cols;

  init_train_index = arma::uvec(n);
  init_test_index = arma::uvec(n_test);

  for(unsigned int i = 0; i<n; i++){
    init_train_index[i] = i;
  }

  for(unsigned int i = 0; i<n_test; i++){
    init_test_index[i] = i;
  }


  n_tree = n_tree_;
  node_min_size = node_min_size_;
  alpha = alpha_;
  beta = beta_;
  nu = nu_;
  sigma_mu = sigma_mu_;
  sigma_mu_sq = sigma_mu*sigma_mu;

  sigma_sq = sigma_*sigma_;
  S_0 = S_0_;
  A_j = A_j_;
  a_j = 0.0;
  n_mcmc = n_mcmc_;
  n_burn = n_burn_;

  // Grow acceptation ratio
  move_proposal = arma::vec(3);
  move_acceptance = arma::vec(3);

  categorical_indicators = categorical_indicators_;

  unsigned int count_cat_indicator = 0;
  for(unsigned int i=0; i < categorical_indicators.n_elem; i++){
    count_cat_indicator++;
  }

  categorical_indicators_bool = count_cat_indicator==0 ? false: true;

}
