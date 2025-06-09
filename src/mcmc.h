#ifndef SUBART_MCMC_H
#define SUBART_MCMC_H

#include "subart_classes.h"

void update_mu_and_predictions(Node* tree,
                              modelParam &data,
                              arma::cube &trees_fit_store,
                              arma::cube &trees_fit_store_test,
                              unsigned int &t,
                              unsigned int &j);

void update_a_j(modelParam &data);
void updateSigma(arma::mat &f_sum_trees,modelParam &data);
void update_y_mat_missing(modelParam & data,arma::mat &y_mat_hat,arma::mat &na_indicators, unsigned int ii);

#endif
