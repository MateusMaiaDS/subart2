#include "subart_classes.h"
#include "tree.h"
#include "mcmc.h"
#include <RcppArmadillo.h>



// [[Rcpp::export]]
Rcpp::List cppsubart_2d(arma::mat x_train,
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
                     bool hier_prior_sigma,
                     arma::uvec categorical_indicators,
                     bool fit_test,
                     bool varimportance,
                     bool sv_bool,
                     arma::umat sv_matrix){


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
                  categorical_indicators,
                  fit_test);

  // Only for d>2
  if(data.d<2){
    throw std::invalid_argument(" The y response should cannot be unidimensional for subart call");
  }

  // Wire specify_variables into data struct
  if (sv_bool) {
    data.sv_bool   = true;
    data.sv_matrix = sv_matrix;
  }

  // Getting n_post
  unsigned int n_post = n_mcmc - n_burn;

  // Creating the posterior elements
  arma::cube y_train_hat_post(data.n,data.d,n_post,arma::fill::none);
  arma::cube y_test_hat_post(data.n_test,data.d,n_post,arma::fill::none);
  arma::cube Sigma_post(data.d,data.d,n_post,arma::fill::none);
  arma::cube all_Sigma_post(data.d,data.d,n_mcmc,arma::fill::none);

  arma::vec partial_residuals(data.n,arma::fill::none);
  arma::cube trees_fit_store(data.n,data.n_tree,data.d,arma::fill::zeros);
  arma::cube trees_fit_store_test(data.n_test,data.n_tree,data.d,arma::fill::zeros);


  // Setting a vector to store the variables used in the split
  // var_imp_post: d × p × n_post (accumulated per posterior draw per response)
  arma::cube var_imp_post(data.d, data.p, n_post, arma::fill::zeros);

  double verb;

  // Define a progress bar here in the future

  std::vector<Node*> all_trees(data.n_tree*data.d);

  for(auto trees= all_trees.begin(); trees!=all_trees.end(); trees++){
    *trees = new Node();
  }


  for(auto nodes:all_trees){
    nodes->Stump(data);
  }




  // Creating variable to help to define which set of tree we are
  unsigned int curr_tree_counter = 0;

  // Matrix to store all predictors for all y
  arma::mat y_mat_hat(data.n,data.d,arma::fill::none);
  arma::mat y_mat_test_hat(data.n_test,data.d,arma::fill::none);



  // Declaring elements necessary to perform operations across the trees
  arma::mat prediction_train_sum(data.n,data.d,arma::fill::none);
  arma::mat prediction_test_sum(data.n_test,data.d,arma::fill::none);

  arma::vec partial_u(data.n,arma::fill::none);
  arma::mat y_mj(data.n,data.d-1,arma::fill::none);
  arma::mat y_hat_mj(data.n,data.d-1,arma::fill::none);

  // Avoiding transposing inside a loop
  arma::mat y_mj_t(data.d-1,data.n,arma::fill::none);
  arma::mat y_hat_mj_t(data.d-1,data.n,arma::fill::none);

  // Initializing the Sigma auxiliary objects
  double Sigma_j_mj;
  double Sigma_mj_j;
  double Sigma_mj_mj; // 2d--change

  // Declaring Sigmas and auxiliarys int/doubles
  unsigned int ij;

  // Creating the auxiliar predictor to update the predictions for a single tree
  arma::vec f_j_hat(data.n,arma::fill::zeros); // Prediction for the tree t

  // Avoid to always createa vector and fill it with zeros
  arma::vec zeros_x_test(data.n_test,arma::fill::zeros);

  // Creating a matrix to store the sum for each j
  arma::mat f_sum_trees(data.n,data.d,arma::fill::zeros);


  arma::vec f_sum_j_excluding_tree_t(data.n,arma::fill::zeros);


  // Creating a matrix for f_sum for the test
  arma::mat f_sum_trees_test(data.n_test,data.d,arma::fill::zeros);


  // Initializing the messages:
  Rprintf("\nRunning subart with numeric y\n\n");
  Rprintf("\nParameters: \n");
  Rprintf("\tnumber of trees: %u \n", data.n_tree);
  Rprintf("\talpha and beta for tree prior: %f %f\n", data.alpha, data.beta);
  Rprintf("\tnumber of responses: %u \n", data.d);
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

    // Do I need to initialise prediction train_test and so on as zero before?

    for(unsigned int j = 0; j < data.d;j++){

      if(j == 0){
        ij = 1;
      } else if(j == 1){
        ij = 0;
      } else {
        // FINISH THE PROGRAM
      }


      Sigma_j_mj = data.Sigma.at(j,ij);
      Sigma_mj_j = data.Sigma.at(ij,j);
      Sigma_mj_mj = data.Sigma.at(ij,ij);

      double scale_mean_aux = Sigma_j_mj/Sigma_mj_mj;

      // ============================================
      // This step does not iterate over the trees!!!
      // ============================================


      for(unsigned int id_train = 0; id_train <data.n; id_train++){
        partial_u.at(id_train) = scale_mean_aux*(data.y_mat.at(id_train,ij)-f_sum_trees.at(id_train,ij));
      }

      double v = data.Sigma.at(j,j) - scale_mean_aux*Sigma_mj_j;

      data.v_j = v;


      // Initializing the the column of f_sum_trees_test as zero
      if(data.fit_test){
        f_sum_trees_test.unsafe_col(j) = zeros_x_test;
      }

      // Updating the tree
      for(unsigned int t = 0; t < data.n_tree;t++){


        // Current tree counter
        curr_tree_counter = t + j*data.n_tree;

        // Updating partial residuals
        if(data.n_tree>1){
          f_sum_j_excluding_tree_t = f_sum_trees.unsafe_col(j) - trees_fit_store.slice(j).unsafe_col(t);
          partial_residuals = y_mat.unsafe_col(j) - f_sum_j_excluding_tree_t;
        } else {
          partial_residuals = y_mat.unsafe_col(j);
        }



        // Sampling the verb
        verb = arma::randu(arma::distr_param(0.0,1.0));

        if(all_trees[curr_tree_counter]->isLeaf){ // Is pointing to the root of the current tree
          verb = 0.1;
        }

        // Selecting each verb -- Here I considering the probability of Grow:0.3, Prune: 0.3, and Change = 0.4 -- May need to reavulate those
        if(verb < 0.5) {
          data.move_proposal.at(0)++;
          grow(all_trees[curr_tree_counter],data,partial_residuals,partial_u,j);
        } else if((verb >= 0.5) & (verb < 1.0)){
          data.move_proposal.at(1)++;
          prune(all_trees[curr_tree_counter],data,partial_residuals,partial_u,j);
        } else {
          data.move_proposal(2)++;
          change(all_trees[curr_tree_counter],data,partial_residuals,partial_u,j);

        }
        // std::cout << "Tree fit one:" << trees_fit_store.at(1,t,j) << std::endl;

        // Updating Mu and Predictions
        update_mu_and_predictions(all_trees[curr_tree_counter],data,trees_fit_store,trees_fit_store_test,t,j);

        // std::cout << "Obs one:" << f_sum_trees.at(1,j) << std::endl;
        // std::cout << "Tree fit two:" << trees_fit_store.at(1,t,j) << std::endl;

        // At the end of the iteration we will have
        f_sum_trees.unsafe_col(j) = f_sum_j_excluding_tree_t + trees_fit_store.slice(j).unsafe_col(t); // REMEMBER TO UPDATE trees_fit_store (specifically the col(t)) BEFORE!

        // std::cout << "Obs one (iter2): " << f_sum_trees.at(1,j) << std::endl;

        // Only if fitting test
        if(data.fit_test){
          f_sum_trees_test.unsafe_col(j) = f_sum_trees_test.unsafe_col(j) + trees_fit_store_test.slice(j).unsafe_col(t);
        }

        // Add latter : add the variable selection step. I.e: which variables are used and which are not

      } // End of iteration in the trees
    } // End of the iterations of response (j)


    // Updating the covariance matrix
    if(hier_prior_sigma){
      update_a_j(data);
    }

    updateSigma(f_sum_trees,data);


    // Storing all_Sigma
    all_Sigma_post.slice(i) = data.Sigma;

    // Storing MCMC iterations
    if(i >= n_burn){

      y_train_hat_post.slice(curr) = f_sum_trees;

      if(data.fit_test){
        y_test_hat_post.slice(curr) = f_sum_trees_test;
      }

      Sigma_post.slice(curr) = data.Sigma;

      if (varimportance) {
        for (unsigned int jj = 0; jj < data.d; jj++) {
          arma::uvec vcounts(data.p, arma::fill::zeros);
          for (unsigned int tt = 0; tt < data.n_tree; tt++) {
            get_var_counts(all_trees[tt + jj * data.n_tree], vcounts);
          }
          var_imp_post.slice(curr).row(jj) = arma::conv_to<arma::rowvec>::from(vcounts);
        }
      }

      curr++;

    }


  } // End of the MCMC iteration


  Rprintf("\nDONE SUBART\n\n");

  return Rcpp::List::create(y_train_hat_post, //[1]
                            y_test_hat_post, //[2]
                            Sigma_post, //[3]
                            all_Sigma_post, // [4]
                            var_imp_post); // [5]
}
