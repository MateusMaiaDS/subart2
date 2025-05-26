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
    // for(auto& id_test:leaf->test_index){
    //   tree_fit_store_test.at(id,t,j) = leaf->mu;
    // }
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
  // for (arma::uword i = 0; i < data.n; ++i) {
  //   arma::rowvec r = f_sum_trees.row(i) - data.y_mat.row(i);
  //   S += r.t() * r;
  // }

  arma::mat residuals_mat = f_sum_trees-data.y_mat;
  S = residuals_mat.t()*residuals_mat;

  // Updating sigma
  data.Sigma = arma::iwishrnd((data.S_0_wish+S),data.nu + data.d - 1 + data.n);

}
