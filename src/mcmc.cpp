#include <RcppArmadillo.h>
#include "subart_classes.h"
#include "tree.h"


using namespace std;
using namespace arma;

void update_mu_and_predictions(Node* tree,
                               modelParam &data,
                               arma::cube &trees_fit_store,
                               arma::cube &trees_fit_store_test,
                               unsigned int &t,
                               unsigned int &j){

  // Navigate through the whole tree and compute the mu values
  std::vector<Node*> t_nodes(0);
  get_leaves(tree,t_nodes);

  for(auto& leaf:t_nodes){

    leaf->mu = arma::randn(arma::distr_param((leaf->S_j)/(leaf->Gamma_j),sqrt(data.v_j/(leaf->Gamma_j)))) ;

    for(auto& id:leaf->train_index){
      trees_fit_store.at(id,t,j) = leaf->mu;
    }

    // // Here replicate the same for the test
    if(data.fit_test){
      for(auto& id_test:leaf->test_index){
        trees_fit_store_test.at(id_test,t,j) = leaf->mu;
      }
    }
  }



  return;
}

void update_a_j(modelParam &data){

  double shape_j = 0.5*(data.d+data.nu);
  arma::mat Precision = arma::inv(data.Sigma);

  // Calculating shape and scale parameters
  for(unsigned int j = 0; j < data.d; j++){
    double scale_j = 1/(data.A_j_vec.at(j)*data.A_j_vec.at(j))+data.nu*Precision.at(j,j);
    double a_j_vec_double_aux = arma::randg(arma::distr_param(shape_j,1/scale_j));

    data.a_j_vec(j) = 1/a_j_vec_double_aux;
    data.S_0_wish(j,j) = (2*data.nu)/data.a_j_vec.at(j);
  }

  return;
}


void updateSigma(arma::mat &f_sum_trees,
                 modelParam &data){

  arma::mat S(data.d, data.d, arma::fill::zeros);
  for (arma::uword i = 0; i < data.n; ++i) {
    arma::rowvec r = f_sum_trees.row(i) - data.y_mat.row(i);
    S += r.t() * r;
  }

  // arma::mat residuals_mat = f_sum_trees-data.y_mat;
  // S = residuals_mat.t()*residuals_mat;

  // Updating sigma
  data.Sigma = arma::iwishrnd((data.S_0_wish+S),data.nu + data.d - 1 + data.n);

}

// Creating the function for the updating whenn y_mat_missing
void update_y_mat_missing(modelParam & data,
                          arma::mat &y_mat_hat,
                          arma::mat &na_indicators,
                          unsigned int ii){

    if(data.d==1) {

      // For the univariate case.
      // ------------------
      // DO IT LATER
      // -------------------
    } else if(data.d == 2) {

      unsigned int ij;
      // Once there are only two outcomes, if ii=1, the selected column should be the other, such as
      if(ii==0) {
         ij = 1;
      } else if(ii==1) {
         ij = 0;
      }

      double Sigma_j_mj = data.Sigma.at(ii,ij);
      double Sigma_mj_j = data.Sigma.at(ij,ii);
      double Sigma_mj_mj = data.Sigma.at(ij,ij);

      double scale_mean_aux = Sigma_j_mj/Sigma_mj_mj;

      double variance_aux = data.Sigma.at(ii,ii) - scale_mean_aux*Sigma_mj_j;
      double mean_y_ii;

      for(unsigned int na_id = 0; na_id < na_indicators.n_rows;na_id++){
        if(na_indicators(na_id,ii)==1){
          mean_y_ii  = y_mat_hat.at(na_id,ii) + scale_mean_aux*(data.y_mat.at(na_id,ij)-y_mat_hat(na_id,ij));
          data.y_mat(na_id,ii) = arma::randn(arma::distr_param(mean_y_ii,sqrt(variance_aux)));
        }
      }


    } else {


      // Initializing the Sigma auxiliary objects
      arma::rowvec Sigma_j_mj((data.d-1),arma::fill::none);
      arma::colvec Sigma_mj_j((data.d-1),arma::fill::none);
      arma::mat Sigma_mj_mj((data.d-1),(data.d-1),arma::fill::none);
      arma::mat Sigma_mj_mj_inv((data.d-1),(data.d-1),arma::fill::none);

      arma::mat y_mj(data.n,data.d-1,arma::fill::none);
      arma::mat y_hat_mj(data.n,data.d-1,arma::fill::none);

      // Avoiding transposing inside a loop
      arma::mat y_mj_t(data.d-1,data.n,arma::fill::none);
      arma::mat y_hat_mj_t(data.d-1,data.n,arma::fill::none);



      // Declaring Sigmas and auxiliarys int/doubles
      unsigned int aux_j_counter = 0;
      unsigned int extra_aux_j_counter = 0;

      for(unsigned int j = 0; j < data.d;j++){

        // Popoulating Sigma;
        aux_j_counter = 0;

        for(unsigned int id_col = 0; id_col < data.d; id_col++){

          if(id_col!=j){
            Sigma_j_mj[aux_j_counter] = data.Sigma.at(j,id_col);
            Sigma_mj_j[aux_j_counter] = data.Sigma.at(j,id_col);
            y_mj.unsafe_col(aux_j_counter) = data.y_mat.unsafe_col(id_col);
            y_hat_mj.unsafe_col(aux_j_counter) = y_mat_hat.unsafe_col(id_col);
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

      }

      //Rcpp::Rcout << "Value for ii:" << ii << endl;
      Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);
      arma::mat scale_mean_aux = Sigma_j_mj*Sigma_mj_mj_inv;


      double variance_aux = data.Sigma(ii,ii) - arma::as_scalar(scale_mean_aux*Sigma_mj_j);
      double mean_y_ii;

      for(unsigned int na_id = 0; na_id < na_indicators.n_rows;na_id++){
        if(na_indicators(na_id,ii)==1){
          mean_y_ii = y_mat_hat.at(na_id,ii) + arma::as_scalar(scale_mean_aux*(y_mj_t.unsafe_col(na_id)-y_hat_mj_t.unsafe_col(na_id)));
          // Rcpp::Rcout << " Printing the "
          data.y_mat.at(na_id,ii) = arma::randn(arma::distr_param(mean_y_ii,sqrt(variance_aux)));
        }
      }


    }

    return;

}

