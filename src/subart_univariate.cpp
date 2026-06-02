#include "subart_classes.h"
#include "tree.h"
#include "mcmc.h"
#include <RcppArmadillo.h>
#include "tree_univariate.h"


// [[Rcpp::export]]
Rcpp::List cppsubart_univariate(arma::mat x_train,
                        arma::vec y,
                        arma::mat x_test,
                        arma::mat x_cut,
                        unsigned int n_tree,
                        unsigned int node_min_size,
                        unsigned int n_mcmc,
                        unsigned int n_burn,
                        double sigma_init,
                        double mu_init,
                        double sigma_mu,
                        double alpha, double beta, double nu,
                        double S_0,
                        double A_j,
                        bool hier_prior_sigma,
                        arma::uvec categorical_indicators,
                        bool fit_test){


  // Posterior counter
  unsigned int curr = 0;


  // Initializing the model definition
  modelParam_uni data(x_train,
                  y,
                  x_test,
                  x_cut,
                  n_tree,
                  node_min_size,
                  alpha,
                  beta,
                  nu,
                  sigma_mu,
                  sigma_init,
                  S_0,
                  A_j,
                  n_mcmc,
                  n_burn,
                  categorical_indicators,
                  fit_test);


  // Getting n_post
  unsigned int n_post = n_mcmc - n_burn;

  // Creating the posterior elements
  arma::mat y_train_hat_post(data.n,n_post,arma::fill::none);
  arma::mat y_test_hat_post(data.n_test,n_post,arma::fill::none);
  arma::vec sigma_post(n_post,arma::fill::none);
  arma::vec all_sigma_post(n_mcmc,arma::fill::none);

  arma::vec partial_residuals(data.n,arma::fill::none);
  arma::mat trees_fit_store(data.n,data.n_tree,arma::fill::zeros);
  arma::mat trees_fit_store_test(data.n_test,data.n_tree,arma::fill::zeros);


  // Setting a vector to store the variables used in the split
  ///  ....
  /// [PLACE HOLDER - built it later]
  ///

  double verb;

  // Define a progress bar here in the future

  std::vector<Node*> all_trees(data.n_tree);

  for(auto trees= all_trees.begin(); trees!=all_trees.end(); trees++){
    *trees = new Node();
  }


  for(auto nodes:all_trees){
    nodes->Stump(data);
  }



  // Matrix to store all predictors for all y
  arma::vec y_mat_hat(data.n,arma::fill::none);
  arma::vec y_mat_test_hat(data.n_test,arma::fill::none);

  // Creating the auxiliar predictor to update the predictions for a single tree
  arma::vec f_j_hat(data.n,arma::fill::zeros); // Prediction for the tree t

  // Avoid to always create a vector and fill it with zeros
  arma::vec zeros_x_test(data.n_test,arma::fill::zeros);

  // Creating a matrix to store the sum for each j
  arma::vec f_sum_trees(data.n,arma::fill::zeros);

  arma::vec f_sum_j_excluding_tree_t(data.n,arma::fill::zeros);


  // Creating a matrix for f_sum for the test
  arma::vec f_sum_trees_test(data.n_test,arma::fill::zeros);


  // Initializing the messages:
  Rprintf("\nRunning subart with numeric y\n\n");
  Rprintf("\nParameters: \n");
  Rprintf("\tnumber of trees: %u \n", data.n_tree);
  Rprintf("\talpha and beta for tree prior: %f %f\n", data.alpha, data.beta);
  Rprintf("\tnumber of training observations: %u\n", data.n);
  if(fit_test){
    Rprintf("\tnumber of test observations : %u\n", data.n_test);
  }
  Rprintf("\tnumber of explanatory variables: %u \n", data.p);
  Rprintf("\nMCMC \n");
  Rprintf("\tnumber of mcmc iter: %u \n", data.n_mcmc);
  Rprintf("\tnumber of n_burn iter: %u \n", data.n_burn);
  Rprintf("\nMCMC run: \n");
  unsigned int printevery = 100;

  for(unsigned int i = 0; i < data.n_mcmc; i ++){

    if(i%printevery==0) Rprintf("done %u (out of %u)\n",i,data.n_mcmc);

      // Initializing the the column of f_sum_trees_test as zero
      if(data.fit_test){
        f_sum_trees_test = zeros_x_test;
      }

      // Updating the tree
      for(unsigned int t = 0; t < data.n_tree;t++){



        // Updating partial residuals
        if(data.n_tree>1){
          f_sum_j_excluding_tree_t = f_sum_trees - trees_fit_store.unsafe_col(t);
          partial_residuals = data.y - f_sum_j_excluding_tree_t;
        } else {
          partial_residuals = y;
        }



        // Sampling the verb
        verb = arma::randu(arma::distr_param(0.0,1.0));

        if(all_trees[t]->isLeaf){ // Is pointing to the root of the current tree
          verb = 0.1;
        }

        // Selecting each verb -- Here I considering the probability of Grow:0.3, Prune: 0.3, and Change = 0.4 -- May need to reavulate those
        if(verb < 0.5) {
          data.move_proposal.at(0)++;
          grow_uni(all_trees[t],data,partial_residuals);
        } else if((verb >= 0.5) & (verb < 1.0)){
          data.move_proposal.at(1)++;
          prune_uni(all_trees[t],data,partial_residuals);
        } else {
          data.move_proposal(2)++;
          change_uni(all_trees[t],data,partial_residuals);

        }
        // std::cout << "Tree fit one:" << trees_fit_store.at(1,t,j) << std::endl;

        // Updating Mu and Predictions
        update_mu_and_predictions_uni(all_trees[t],data,trees_fit_store,trees_fit_store_test,t);

        // At the end of the iteration we will have
        f_sum_trees = f_sum_j_excluding_tree_t + trees_fit_store.unsafe_col(t); // REMEMBER TO UPDATE trees_fit_store (specifically the col(t)) BEFORE!

        // std::cout << "Obs one (iter2): " << f_sum_trees.at(1,j) << std::endl;

        // Only if fitting test
        if(data.fit_test){
          f_sum_trees_test = f_sum_trees_test + trees_fit_store_test.unsafe_col(t);
        }

        // Add latter : add the variable selection step. I.e: which variables are used and which are not

      } // End of iteration in the trees


    // Updating the covariance matrix
    update_a_j_uni(data);


    updateSigma_uni(f_sum_trees,data);


    // Storing all_Sigma
    all_sigma_post[i] = data.sigma_sq;

    // Storing MCMC iterations
    if(i >= n_burn){

      y_train_hat_post.unsafe_col(curr) = f_sum_trees;


      if(data.fit_test){
        y_test_hat_post.unsafe_col(curr) = f_sum_trees_test;
      }

      sigma_post[curr] = data.sigma_sq;
      curr++;

    }


  } // End of the MCMC iteration


  Rprintf("\nDONE SUBART\n\n");

  return Rcpp::List::create(y_train_hat_post, //[1]
                            y_test_hat_post, //[2]
                            sigma_post, //[3]
                            all_sigma_post); // [4];
}
