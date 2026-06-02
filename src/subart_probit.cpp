// subart_probit.cpp
// Probit suBART: classification paths using Albert-Chib data augmentation.
//
// cppsubart_univariate_CLASS : univariate binary outcome (d=1)
// cppsubart_CLASS            : multivariate binary outcomes (d>=2)
//
// Methodology: Esser & Maia (2025), AOAS supplement §4.6.
//   For each y_i in {0,1}: draw z_i from truncated normal
//     z_i ~ TN_{[0,inf)}(f(x_i), 1)  if y_i = 1
//     z_i ~ TN_{(-inf,0)}(f(x_i), 1) if y_i = 0
//   Then update trees with z as continuous response, fixing sigma^2 = 1.
//
// Note: Uses Rf_pnorm5 / Rf_qnorm5 from Rmath (available via RcppArmadillo).

#include <RcppArmadillo.h>
#include "subart_classes.h"
#include "tree.h"
#include "tree_univariate.h"
#include "mcmc.h"

// Truncated normal sampler via inverse-CDF (Albert-Chib 1993)
// Sample from TN_{[lower, upper)}(mu, sigma=1)
inline double rtruncnorm(double mu, double lower, double upper) {
  double Fl = Rf_pnorm5(lower - mu, 0.0, 1.0, 1, 0);
  double Fu = Rf_pnorm5(upper - mu, 0.0, 1.0, 1, 0);
  double u = Fl + (Fu - Fl) * R::runif(0.0, 1.0);
  // Clamp to avoid Inf from qnorm at 0 or 1
  if (u < 1e-15) u = 1e-15;
  if (u > 1.0 - 1e-15) u = 1.0 - 1e-15;
  return mu + Rf_qnorm5(u, 0.0, 1.0, 1, 0);
}

// ============================================================
// cppsubart_univariate_CLASS
// Probit BART for a single binary outcome.
// Mirrors subart_univariate.cpp; sigma^2 = 1 is fixed; no sigma update.
// ============================================================

// [[Rcpp::export]]
Rcpp::List cppsubart_univariate_CLASS(arma::mat x_train,
                                      arma::uvec y_bin,
                                      arma::mat x_test,
                                      arma::mat x_cut,
                                      unsigned int n_tree,
                                      unsigned int node_min_size,
                                      unsigned int n_mcmc,
                                      unsigned int n_burn,
                                      double mu_init,
                                      double sigma_mu,
                                      double alpha, double beta,
                                      arma::uvec categorical_indicators,
                                      bool fit_test,
                                      bool varimportance,
                                      bool sv_bool,
                                      arma::umat sv_matrix){

  unsigned int n = x_train.n_rows;

  // Latent z: initialise at probit of class proportion
  double prop1 = (double)arma::sum(y_bin) / (double)n;
  double z_init = Rf_qnorm5(std::max(0.01, std::min(0.99, prop1)), 0.0, 1.0, 1, 0);
  arma::vec z(n, arma::fill::value(z_init));

  // sigma^2 = 1 fixed; pass dummy sigma_init = 1, S_0 = 0, A_j = 0
  double sigma_init = 1.0;
  double S_0 = 0.0;
  double A_j = 0.0;

  modelParam_uni data(x_train,
                      z,
                      x_test,
                      x_cut,
                      n_tree,
                      node_min_size,
                      alpha,
                      beta,
                      /* nu = */ 3.0,
                      sigma_mu,
                      sigma_init,
                      S_0,
                      A_j,
                      n_mcmc,
                      n_burn,
                      categorical_indicators,
                      fit_test);

  if (sv_bool) {
    data.sv_bool   = true;
    data.sv_matrix = sv_matrix;
  }

  unsigned int n_post = n_mcmc - n_burn;
  unsigned int curr = 0;

  // Posterior storage
  arma::mat y_train_hat_post(n, n_post, arma::fill::zeros);
  arma::mat y_test_hat_post(x_test.n_rows, n_post, arma::fill::zeros);
  arma::mat var_imp_post(n_post, data.p, arma::fill::zeros);

  // Tree fit storage (same structure as subart_univariate.cpp)
  arma::mat trees_fit_store(data.n, data.n_tree, arma::fill::zeros);
  arma::mat trees_fit_store_test(data.n_test, data.n_tree, arma::fill::zeros);

  std::vector<Node*> all_trees(data.n_tree);
  for (auto& t : all_trees) t = new Node();
  for (auto nd : all_trees) nd->Stump(data);

  arma::vec f_sum_trees(n, arma::fill::zeros);
  arma::vec f_sum_trees_test(x_test.n_rows, arma::fill::zeros);
  arma::vec f_sum_j_excluding_tree_t(n, arma::fill::zeros);
  arma::vec partial_residuals(n, arma::fill::none);
  arma::vec zeros_x_test(data.n_test, arma::fill::zeros);

  double verb;
  unsigned int printevery = 100;

  Rprintf("\nRunning subart (probit, univariate)\n\n");

  for (unsigned int i = 0; i < n_mcmc; i++) {

    if (i % printevery == 0) Rprintf("done %u (out of %u)\n", i, n_mcmc);

    // Step 1: update latent z (Albert-Chib), sigma^2=1 fixed
    data.sigma_sq = 1.0;
    for (unsigned int ii = 0; ii < n; ii++) {
      if (y_bin[ii] == 1u) {
        z[ii] = rtruncnorm(f_sum_trees[ii], 0.0,  1e10);
      } else {
        z[ii] = rtruncnorm(f_sum_trees[ii], -1e10, 0.0);
      }
    }
    data.y = z;

    // Step 2: update trees (mirrors subart_univariate.cpp)
    if (data.fit_test) {
      f_sum_trees_test = zeros_x_test;
    }

    for (unsigned int t = 0; t < data.n_tree; t++) {
      if (data.n_tree > 1) {
        f_sum_j_excluding_tree_t = f_sum_trees - trees_fit_store.unsafe_col(t);
        partial_residuals = data.y - f_sum_j_excluding_tree_t;
      } else {
        partial_residuals = z;
      }

      verb = arma::randu(arma::distr_param(0.0, 1.0));
      if (all_trees[t]->isLeaf) verb = 0.1;

      if (verb < 0.3) {
        grow_uni(all_trees[t], data, partial_residuals);
      } else if (verb < 0.6) {
        prune_uni(all_trees[t], data, partial_residuals);
      } else {
        change_uni(all_trees[t], data, partial_residuals);
      }

      // Update leaf means and tree predictions (sigma^2=1 is used inside)
      update_mu_and_predictions_uni(all_trees[t], data, trees_fit_store, trees_fit_store_test, t);

      f_sum_trees = f_sum_j_excluding_tree_t + trees_fit_store.unsafe_col(t);

      if (data.fit_test) {
        f_sum_trees_test = f_sum_trees_test + trees_fit_store_test.unsafe_col(t);
      }
    }

    // Step 3: NO sigma update (sigma^2 = 1 fixed for probit)

    if (i >= n_burn) {
      y_train_hat_post.col(curr) = f_sum_trees;
      if (data.fit_test) {
        y_test_hat_post.col(curr) = f_sum_trees_test;
      }

      if (varimportance) {
        arma::uvec vcounts(data.p, arma::fill::zeros);
        for (unsigned int tt = 0; tt < data.n_tree; tt++) {
          get_var_counts(all_trees[tt], vcounts);
        }
        var_imp_post.row(curr) = arma::conv_to<arma::rowvec>::from(vcounts);
      }

      curr++;
    }
  }

  for (auto t : all_trees) delete t;

  Rprintf("\nDONE SUBART CLASS (univariate)\n\n");

  return Rcpp::List::create(y_train_hat_post, //[1]
                            y_test_hat_post,  //[2]
                            var_imp_post);    //[3]
}


// ============================================================
// cppsubart_CLASS
// Probit suBART for d >= 2 binary outcomes.
// Uses Albert-Chib augmentation per outcome (univariate margins).
// For simplicity, updates the correlation matrix R via IW + normalisation.
// ============================================================

// [[Rcpp::export]]
Rcpp::List cppsubart_CLASS(arma::mat x_train,
                           arma::umat y_bin,        // n x d binary matrix
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
                           unsigned int m,           // nu_prop for correlation update
                           arma::uvec categorical_indicators,
                           bool fit_test,
                           bool varimportance,
                           bool sv_bool,
                           arma::umat sv_matrix){

  unsigned int n = x_train.n_rows;
  unsigned int d = y_bin.n_cols;
  unsigned int curr = 0;

  // Latent Z matrix (n x d): initialise at probit of per-outcome class proportion
  arma::mat Z(n, d, arma::fill::zeros);
  for (unsigned int j = 0; j < d; j++) {
    double p1 = (double)arma::sum(y_bin.col(j)) / (double)n;
    double z0 = Rf_qnorm5(std::max(0.01, std::min(0.99, p1)), 0.0, 1.0, 1, 0);
    Z.col(j).fill(z0);
  }

  modelParam data(x_train,
                  Z,
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

  if (sv_bool) {
    data.sv_bool   = true;
    data.sv_matrix = sv_matrix;
  }

  unsigned int n_post = n_mcmc - n_burn;

  arma::cube y_train_hat_post(n, d, n_post, arma::fill::zeros);
  arma::cube y_test_hat_post(x_test.n_rows, d, n_post, arma::fill::zeros);
  arma::cube Sigma_post(d, d, n_post, arma::fill::zeros);
  arma::cube all_Sigma_post(d, d, n_mcmc, arma::fill::zeros);
  arma::cube var_imp_post(d, data.p, n_post, arma::fill::zeros);

  // Tree fit storage (same structure as subart.cpp)
  arma::cube trees_fit_store(n, n_tree, d, arma::fill::zeros);
  arma::cube trees_fit_store_test(x_test.n_rows, n_tree, d, arma::fill::zeros);

  std::vector<Node*> all_trees(n_tree * d);
  for (auto& t : all_trees) t = new Node();
  for (auto nd : all_trees) nd->Stump(data);

  arma::mat f_sum_trees(n, d, arma::fill::zeros);
  arma::mat f_sum_trees_test(x_test.n_rows, d, arma::fill::zeros);
  arma::vec f_sum_j_excluding_tree_t(n, arma::fill::zeros);
  arma::vec partial_residuals(n, arma::fill::none);
  arma::vec partial_u(n, arma::fill::zeros);
  arma::vec zeros_x_test(x_test.n_rows, arma::fill::zeros);

  double verb;
  unsigned int curr_tree_counter;
  unsigned int printevery = 100;

  Rprintf("\nRunning subart (probit, multivariate d=%u)\n\n", d);

  for (unsigned int i = 0; i < n_mcmc; i++) {

    if (i % printevery == 0) Rprintf("done %u (out of %u)\n", i, n_mcmc);

    // Step 1: update latent Z per outcome j (marginal Albert-Chib, diag(Sigma)=1)
    for (unsigned int j = 0; j < d; j++) {
      for (unsigned int ii = 0; ii < n; ii++) {
        double fi = f_sum_trees(ii, j);
        if (y_bin(ii, j) == 1u) {
          Z(ii, j) = rtruncnorm(fi, 0.0,  1e10);
        } else {
          Z(ii, j) = rtruncnorm(fi, -1e10, 0.0);
        }
      }
    }
    data.y_mat = Z;

    // Step 2: update trees per outcome j (mirrors subart.cpp)
    for (unsigned int j = 0; j < d; j++) {

      // Compute conditional mean adjustment partial_u using current Sigma
      arma::uvec other_idx(d - 1);
      {
        unsigned int k = 0;
        for (unsigned int jj = 0; jj < d; jj++) { if (jj != j) other_idx[k++] = jj; }
      }

      arma::rowvec Sigma_j_mj(d - 1);
      for (unsigned int k = 0; k < d - 1; k++) Sigma_j_mj[k] = data.Sigma(j, other_idx[k]); // 1 x (d-1)
      arma::mat Sigma_mj_mj = data.Sigma.submat(other_idx, other_idx);
      double Sigma_j_j = data.Sigma(j, j);
      arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);
      arma::rowvec scale_mean_aux = Sigma_j_mj * Sigma_mj_mj_inv; // 1 x (d-1)
      double v = Sigma_j_j - arma::as_scalar(scale_mean_aux * Sigma_j_mj.t());
      data.v_j = v;

      for (unsigned int ii = 0; ii < n; ii++) {
        arma::vec Z_ii = arma::conv_to<arma::vec>::from(Z.row(ii));
        arma::vec Fii  = arma::conv_to<arma::vec>::from(f_sum_trees.row(ii));
        arma::colvec diff = Z_ii.rows(other_idx) - Fii.rows(other_idx);
        partial_u[ii] = arma::as_scalar(scale_mean_aux * diff);
      }

      if (data.fit_test) {
        f_sum_trees_test.col(j) = zeros_x_test;
      }

      for (unsigned int t = 0; t < n_tree; t++) {
        curr_tree_counter = t + j * n_tree;

        if (n_tree > 1) {
          f_sum_j_excluding_tree_t = f_sum_trees.col(j) - trees_fit_store.slice(j).col(t);
          partial_residuals = data.y_mat.col(j) - f_sum_j_excluding_tree_t;
        } else {
          partial_residuals = data.y_mat.col(j);
        }

        verb = arma::randu(arma::distr_param(0.0, 1.0));
        if (all_trees[curr_tree_counter]->isLeaf) verb = 0.1;

        if (verb < 0.3) {
          grow(all_trees[curr_tree_counter], data, partial_residuals, partial_u, j);
        } else if (verb < 0.6) {
          prune(all_trees[curr_tree_counter], data, partial_residuals, partial_u, j);
        } else {
          change(all_trees[curr_tree_counter], data, partial_residuals, partial_u, j);
        }

        update_mu_and_predictions(all_trees[curr_tree_counter], data,
                                  trees_fit_store, trees_fit_store_test, t, j);

        f_sum_trees.col(j) = f_sum_j_excluding_tree_t + trees_fit_store.slice(j).col(t);

        if (data.fit_test) {
          f_sum_trees_test.col(j) = f_sum_trees_test.col(j) + trees_fit_store_test.slice(j).col(t);
        }
      }
    }

    // Step 3: update correlation matrix R via IW draw + diagonal normalisation
    // Residuals e_i = Z_i - f(x_i)
    arma::mat E = Z - f_sum_trees;
    arma::mat S_post = S_0_wish + E.t() * E;
    arma::mat Sigma_draw = arma::iwishrnd(S_post, nu + static_cast<double>(d) - 1.0 + static_cast<double>(n));

    // Normalise to correlation matrix (fix diag = 1 for probit identifiability)
    arma::vec D = arma::sqrt(arma::diagvec(Sigma_draw));
    arma::mat R_draw = Sigma_draw;
    for (unsigned int jj = 0; jj < d; jj++)
      for (unsigned int kk = 0; kk < d; kk++)
        R_draw(jj, kk) /= (D[jj] * D[kk]);

    data.Sigma = R_draw;
    all_Sigma_post.slice(i) = data.Sigma;

    if (i >= n_burn) {
      y_train_hat_post.slice(curr) = f_sum_trees;
      if (data.fit_test) {
        y_test_hat_post.slice(curr) = f_sum_trees_test;
      }
      Sigma_post.slice(curr) = data.Sigma;

      if (varimportance) {
        for (unsigned int jj = 0; jj < d; jj++) {
          arma::uvec vcounts(data.p, arma::fill::zeros);
          for (unsigned int tt = 0; tt < n_tree; tt++) {
            get_var_counts(all_trees[tt + jj * n_tree], vcounts);
          }
          var_imp_post.slice(curr).row(jj) = arma::conv_to<arma::rowvec>::from(vcounts);
        }
      }

      curr++;
    }
  }

  for (auto t : all_trees) delete t;

  Rprintf("\nDONE SUBART CLASS (multivariate)\n\n");

  return Rcpp::List::create(y_train_hat_post, //[1]
                            y_test_hat_post,  //[2]
                            Sigma_post,       //[3]
                            all_Sigma_post,   //[4]
                            var_imp_post);    //[5]
}
