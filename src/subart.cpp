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

      // Only for d>2
      if(data.d<2){
        throw std::invalid_argument(" The y response should cannot be unidimensional for subart call");
      }
      // Getting n_post
      unsigned int n_post = n_mcmc - n_burn;

      // Creating the posterior elements
      arma::cube y_train_hat_post(data.n,data.d,n_post,arma::fill::none);
      arma::cube y_test_hat_post(data.n_test,data.d,n_post,arma::fill::none);
      arma::cube Sigma_post(data.d,data.d,n_post,arma::fill::none);
      arma::cube all_Sigma_post(data.d,data.d,n_mcmc,arma::fill::none);

      arma::vec partial_residuals(data.n,arma::fill::none);
      arma::cube tree_fits_store(data.n,data.n_tree,data.d,arma::fill::zeros);
      arma::cube tree_fits_store_test(data.n_test,data.n_tree,data.d,arma::fill::zeros);


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
      arma::rowvec Sigma_j_mj((data.d-1),arma::fill::none); // 2d--change -- TODO: change this to a vector and make things simpler later.
      arma::colvec Sigma_mj_j((data.d-1),arma::fill::none); // 2d--change -- TODO: change this to a vector and make things simpler later.
      arma::mat Sigma_mj_mj((data.d-1),(data.d-1),arma::fill::none); // 2d--change
      arma::mat Sigma_mj_mj_inv((data.d-1),(data.d-1),arma::fill::none);

      // Declaring Sigmas and auxiliarys int/doubles
      double Sigma_j_j;
      unsigned int aux_j_counter = 0;
      unsigned int extra_aux_j_counter = 0;


      // Creating the auxiliar predictor to update the predictions for a single tree
      arma::vec f_j_hat(data.y_mat.n,arma::fill::zeros); // Prediction for the tree t
      arma::vec f_j_test_hat(data.x_test.n,arma::fill::none);


      // Creating a matrix to store the sum for each j
      arma::mat f_sum_trees(data.n,data.d,arma::fill::zeros);
      arma::vec f_sum_excluding_tree_j(data.n,data.d,arma::fill::zeros);


      for(unsigned int i = 0; i < data.n_mcmc; i ++){


        // Do I need to initialise prediction train_test and so on as zero before?

        for(unsigned int j = 0; j < data.d;j++){

            // Popoulating Sigma;
            Sigma_j_j = data.Sigma.at(j,j);
            aux_j_counter = 0;

            for(unsigned int id_col = 0; id_col < data.d; id_col++){

              if(id_col!=j){
                Sigma_j_mj[aux_j_counter] = data.Sigma.at(j,id_col);
                Sigma_mj_j[aux_j_counter] = data.Sigma.at(j,id_col);
                y_mj.unsafe_col(aux_j_counter) = data.y_mat.unsafe_col(id_col);
                y_hat_mj.unsafe_col(aux_j_counter) = data.y_mat.unsafe_col(id_col);
                extra_aux_j_counter = 0;

                for(unsigned int extra_id_col = 0; extra_id_col < data.d; extra_id_col++){

                  if(extra_id_col!=j){
                    Sigma_mj_mj.at(aux_j_counter,extra_aux_j_counter) = data.Sigma.at(id_col,extra_id_col);
                    extra_aux_j_counter++;
                  }

                }

              aux_j_counter++;

              }

            }

            // Avoiding extra operations inside the id_train loop
            y_mj_t = y_mj.t();
            y_hat_mj_t = y_hat_mj.t();

            // ============================================
            // This step does not iterate over the trees!!!
            // ============================================

            Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);

            for(unsigned int id_train = 0; id_train <data.n; id_train++){
                partial_u.at(id_train) = arma::as_scalar(Sigma_j_mj*(Sigma_mj_mj_inv*(y_mj_t.unsafe_col(id_train)-y_hat_mj_t.unsafe_col(id_train))));
            }

            double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj*(Sigma_mj_mj_inv*Sigma_mj_j));

            data.v_j = v;

            data.sigma_mu_j = data.sigma_mu.at(j);


            // Updating the tree
            for(unsigned int t = 0; t < data.n_tree;t++){


                // Current tree counter
                curr_tree_counter = t + j*data.n_tree;

                // Updating partial residuals
                if(data.n_tree>1){
                    f_sum_excluding_tree_j = f_sum_trees.unsafe_col(j) - tree_fits_store.slice(j).unsafe_col(t);
                    partial_residuals = y_mat.unsafe_col(j) - f_sum_excluding_tree_j;
                } else {
                    partial_residuals = y_mat.unsafe_col(j);
                }


                // At the end of the iteration we will have
                f_sum_trees.unsafe_col(j) = f_sum_excluding_tree_j + tree_fits_store.slice(j).unsafe_col(t); // REMEMBER TO UPDATE TREE_FITS_STORE (specifically the col(t)) BEFORE!


            }
        }
      }

      return;
}
