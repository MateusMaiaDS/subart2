#include "subart_classes.h"
#include "tree.h"
#include <RcppArmadillo.h>



// [[Rcpp::export]]
void cppsubart(arma::mat x_train,
                   arma::mat y_mat,
                   arma::mat x_test,
                   arma::mat x_cut,
                   unsigned int n_tree,
                   unsigned int node_min_size,
                   unsigned int n_mcmc,
                   unsigned int n_burn,
                   arma::mat Sigma_init,
                   arma::vec mu_init,
                   arma::vec sigma_mu,
                   double alpha, double beta, double nu,
                   arma::mat S_0_wish,
                   arma::vec A_j_vec,
                   bool update_Sigma,
                   bool var_selection_bool,
                   bool sv_bool,
                   bool hier_prior_bool,
                   arma::mat sv_matrix,
                   arma::uvec categorical_indicators){


      // Posterior counter
      unsigned int curr = 0;

      // Initializing the model definition
      modelParam data(x_train,
                      y_mat,
                      x_test,
                      x_cut,
                      n_tree,
                      node_min_size,
                      alpha,
                      beta,
                      nu,
                      sigma_mu,
                      Sigma_init,
                      S_0_wish,
                      A_j_vec,
                      n_mcmc,
                      n_burn,
                      sv_bool,
                      sv_matrix,
                      categorical_indicators);

      // Getting n_post
      unsigned int n_post = n_mcmc - n_burn;

      // Creating the posterior elements
      arma::cube y_train_hat_post(data.n,data.d,n_post);
      arma::cube y_test_hat_post(data.n_test,data.d,n_post);
      arma::cube Sigma_post(data.d,data.d,n_post);
      arma::cube all_Sigma_post(data.d,data.d,n_mcmc);

      arma::vec partial_residuals(data.n);
      arma::cube tree_fits_store(data.n,data.n_tree,data.d);
      arma::cube tree_fits_store_test(data.n_test,data.n_tree,data.d);


      // Setting a vector to store the variables used in the split
      ///  ....
      /// [PLACE HOLDER - built it later]
      ///

      double verb;

      // Define a progress bar here in the future

      std::vector<Node*> all_trees(data.n_tree*data.d);

      for(auto trees= all_trees.begin(); trees!=all_trees.end(); trees++){
        *trees = new Node();
      }

      for(auto nodes:all_trees){
        nodes->Stump(data);
      }

      return;
}
