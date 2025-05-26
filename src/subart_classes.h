#ifndef SUBART_CLASSES_H
#define SUBART_CLASSES_H

#include <RcppArmadillo.h>

// Creating the struct
struct Node;
struct modelParam;
struct modelParam_uni;


struct modelParam {

  arma::mat x_train;
  arma::mat y_mat;
  arma::mat x_test;
  arma::mat xcut;

  unsigned int n;
  unsigned int n_test;
  unsigned int d;

  arma::uvec init_train_index;
  arma::uvec init_test_index;

  // BART prior param specification
  unsigned int n_tree;
  unsigned int d_var; // Dimension of variables in my base
  double alpha;
  double beta;
  arma::vec sigma_mu;
  arma::mat Sigma;
  arma::mat S_0_wish;
  arma::vec a_j_vec;
  arma::vec A_j_vec;
  arma::mat W;
  arma::mat R;
  arma::mat D;


  double nu;
  unsigned int node_min_size;

  // MCMC spec.
  unsigned int n_mcmc;
  unsigned int n_burn;

  // Create an indicator of accepted grown trees
  arma::uvec::fixed<3> move_proposal;
  arma::uvec::fixed<3> move_acceptance;

  // Creating new objects for new rules for categorical indicators
  arma::uvec categorical_indicators;
  bool categorical_indicators_bool;

  // Elements to be used in the loglikelihood update and mu update
  double v_j;
  arma::vec sigma_mu_j;
  arma::vec sigma_mu_j_sq;

  // A boolean to update or not the test
  bool fit_test;

  // Defining the constructor for the model param
  modelParam(arma::mat x_train_,
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
             bool fit_test_);

};


struct modelParam_uni {

  arma::mat x_train;
  arma::mat y_mat;
  arma::mat x_test;
  arma::mat xcut;

  unsigned int n;
  unsigned int n_test;

  arma::uvec init_train_index;
  arma::uvec init_test_index;

  unsigned int d;
  // BART prior param specification
  unsigned int n_tree;
  unsigned int d_var; // Dimension of variables in my base
  double alpha;
  double beta;
  arma::vec sigma_mu;
  arma::mat Sigma;
  arma::mat S_0_wish;
  arma::vec a_j_vec;
  arma::vec A_j_vec;
  arma::mat W;
  arma::mat R;
  arma::mat D;

  double nu;
  int node_min_size;

  // MCMC spec.
  unsigned int n_mcmc;
  unsigned int n_burn;

  // Create an indicator of accepted grown trees
  arma::vec move_proposal;
  arma::vec move_acceptance;

  // Creating new objects for new rules for categorical indicators
  arma::uvec categorical_indicators;
  bool categorical_indicators_bool;

  // Elements to be used in the loglikelihood update and mu update
  double v_j;
  double sigma_mu_j;
  double sigma_mu_j_sq;

  // Defining the constructor for the model param
  modelParam_uni(arma::mat x_train_,
                 arma::vec y_mat_,
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
                 arma::uvec categorical_indicators_);

};

// Creating the node struct
struct Node {

  bool isRoot;
  bool isLeaf;
  Node* left;
  Node* right;
  Node* parent;
  arma::uvec train_index;
  arma::uvec test_index;


  // Branch parameters
  unsigned int var_split;
  double var_split_rule;
  double lower;
  double upper;
  double curr_weight; // indicates if the observation is within terminal node or not
  unsigned int depth = 0;


  // Leaf parameters
  double mu;

  // Storing sufficient statistics over the nodes
  double log_likelihood = 0.0;
  double r_sum = 0.0;
  double u_sum = 0.0;
  double Gamma_j;
  double S_j;


  unsigned int n_leaf = 0;
  unsigned int n_leaf_test = 0;

  // Creating the methods
  void addingLeaves();
  void deletingLeaves();
  void Stump(modelParam& data);
  void updateWeight(const arma::mat X, int i);
  void getLimits(unsigned int split_var_candidate,
                 double &lower_candidate,
                 double &upper_candidate); // This function will get previous limit for the current var
  bool isLeft();
  bool isRight();
  // void grow(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec &curr_u,unsigned int &j);
  // void prune(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u,unsigned int &j);
  // void change(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u,unsigned int &j);
  void nodeLogLike(modelParam &data, unsigned int &j);
  void updateResiduals(modelParam& data, arma::vec &curr_res, arma::vec &curr_u, unsigned int &j);
  void updateResiduals_uni(modelParam& data, arma::vec &curr_res, unsigned int &j);
  void displayCurrNode();

  Node();
  ~Node();
};

#endif
