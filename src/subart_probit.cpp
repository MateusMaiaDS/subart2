// subart_probit.cpp
// Probit suBART: classification via Albert-Chib data augmentation.
//
// cppsubart_univariate_CLASS : univariate binary outcome (d = 1)
// cppsubart_CLASS            : multivariate binary outcomes (d >= 2)
//
// Methodology matches mvnbart6.cpp (MateusMaiaDS/subart repository):
//   - Latent z updated *inside* the j-loop (interleaved), after each j's trees.
//   - z_{ij} | z_{i,-j}, f, R ~ TN(conditional_mean, sqrt(v_j))
//     conditional_mean = f(x_i,j) + Sigma_{j,-j} * Sigma_{-j,-j}^{-1} * (z_{i,-j} - f_{i,-j})
//   - For d >= 2: correlation R updated via MH (IW proposal around current R).
//   - For d = 1: sigma^2 = 1 fixed; marginal z update (no R update needed).
//
// References:
//   Esser & Maia (2025) AOAS supplement §4.6.
//   Original code: github.com/MateusMaiaDS/subart/blob/master/src/mvnbart6.cpp

#include <RcppArmadillo.h>
#include "subart_classes.h"
#include "tree.h"
#include "tree_univariate.h"
#include "mcmc.h"

// ------------------------------------------------------------------
// Truncated normal samplers (inverse-CDF, sigma-aware)
// rtn_pos : TN_{(0, ∞)}(mu, sigma)
// rtn_neg : TN_{(-∞, 0]}(mu, sigma)
// ------------------------------------------------------------------
static inline double rtn_pos(double mu, double sigma) {
  double a  = -mu / sigma;
  double Fa = Rf_pnorm5(a, 0.0, 1.0, 1, 0);
  double u  = Fa + (1.0 - Fa) * R::runif(0.0, 1.0);
  if (u > 1.0 - 1e-15) u = 1.0 - 1e-15;
  return mu + sigma * Rf_qnorm5(u, 0.0, 1.0, 1, 0);
}

static inline double rtn_neg(double mu, double sigma) {
  double a  = -mu / sigma;
  double Fa = Rf_pnorm5(a, 0.0, 1.0, 1, 0);
  double u  = Fa * R::runif(0.0, 1.0);
  if (u < 1e-15) u = 1e-15;
  return mu + sigma * Rf_qnorm5(u, 0.0, 1.0, 1, 0);
}

// ------------------------------------------------------------------
// MH helpers for the correlation-matrix update (mvnbart6 §R-update)
// iwishart_log omits the multivariate-gamma term (cancels in ratio).
// ------------------------------------------------------------------
static double iwishart_log(const arma::mat& X, const arma::mat& Psi, double nu) {
  int p = X.n_rows;
  double logdet_X   = arma::log_det(X).real();
  double logdet_Psi = arma::log_det(Psi).real();
  double tr         = arma::trace(Psi * arma::inv(X));
  return -0.5*(nu + p + 1)*logdet_X - 0.5*tr + 0.5*nu*logdet_Psi
         - 0.5*nu*p*std::log(2.0);
}

static double log_prior_R(const arma::mat& R, const arma::mat& D, double nu) {
  unsigned int d = R.n_cols;
  arma::mat sqD = arma::sqrt(D);
  arma::mat W   = sqD * R * sqD;
  return iwishart_log(W, arma::eye(d, d), nu + d - 1)
       + 0.5*(d - 1) * arma::sum(arma::log(D.diag()));
}

static double log_post_R(const arma::mat& R, const arma::mat& D,
                          double nu, const arma::mat& resid) {
  unsigned int n = resid.n_rows;
  double lp = log_prior_R(R, D, nu) - 0.5*(double)n*arma::log_det(R).real();
  arma::mat Rinv = arma::inv(R);
  for (unsigned int i = 0; i < n; i++)
    lp -= 0.5 * arma::as_scalar(resid.row(i) * Rinv * resid.row(i).t());
  return lp;
}

static double log_prop_R(const arma::mat& R_s, const arma::mat& D_s,
                          const arma::mat& R,   const arma::mat& D, int m) {
  unsigned int d = R.n_cols;
  arma::mat sqDs = arma::sqrt(D_s);
  arma::mat sqD  = arma::sqrt(D);
  arma::mat Ws   = sqDs * R_s * sqDs;
  arma::mat W    = sqD  * R  * sqD;
  return iwishart_log(Ws, (double)m * W, m)
       + 0.5*(d - 1) * arma::sum(arma::log(D_s.diag()));
}

// ------------------------------------------------------------------
// Interleaved conditional z-update for outcome j (mvnbart6: update_z)
// Reuses scale_mean_aux = Sigma_j_mj * Sigma_mj_mj_inv (1 x (d-1))
// ------------------------------------------------------------------
static void update_z_col(arma::mat&       z_mat,
                          const arma::mat& f_mat,
                          const arma::mat& y_orig,     // original 0/1 binary y
                          unsigned int     j,
                          double           v_j,
                          const arma::mat& scale_aux,  // 1 x (d-1)
                          unsigned int     n,
                          unsigned int     d) {
  double sq_v = std::sqrt(v_j);
  for (unsigned int i = 0; i < n; i++) {
    double cond_mean = f_mat(i, j);
    unsigned int idx = 0;
    for (unsigned int k = 0; k < d; k++) {
      if (k == j) continue;
      cond_mean += scale_aux(0, idx) * (z_mat(i, k) - f_mat(i, k));
      idx++;
    }
    if (y_orig(i, j) == 1.0)
      z_mat(i, j) = rtn_pos(cond_mean, sq_v);
    else
      z_mat(i, j) = rtn_neg(cond_mean, sq_v);
  }
}


// ==================================================================
// cppsubart_univariate_CLASS
// d = 1 probit: sigma^2 = 1 fixed, marginal TN update for z.
// ==================================================================
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
                                       arma::umat sv_matrix) {

  unsigned int n = x_train.n_rows;

  double prop1  = (double)arma::sum(y_bin) / (double)n;
  double z_init = Rf_qnorm5(std::max(0.01, std::min(0.99, prop1)), 0.0, 1.0, 1, 0);
  arma::vec z(n, arma::fill::value(z_init));

  modelParam_uni data(x_train, z, x_test, x_cut,
                      n_tree, node_min_size, alpha, beta,
                      /* nu */ 3.0, sigma_mu, /* sigma */ 1.0,
                      /* S_0 */ 0.0, /* A_j */ 0.0,
                      n_mcmc, n_burn, categorical_indicators, fit_test);

  if (sv_bool) { data.sv_bool = true; data.sv_matrix = sv_matrix; }

  unsigned int n_post = n_mcmc - n_burn;
  unsigned int curr   = 0;

  arma::mat y_train_hat_post(n,              n_post, arma::fill::zeros);
  arma::mat y_test_hat_post (x_test.n_rows,  n_post, arma::fill::zeros);
  arma::mat var_imp_post    (n_post, data.p, arma::fill::zeros);

  arma::mat trees_fit_store     (data.n,      data.n_tree, arma::fill::zeros);
  arma::mat trees_fit_store_test(data.n_test, data.n_tree, arma::fill::zeros);

  std::vector<Node*> all_trees(data.n_tree);
  for (auto& t : all_trees) t = new Node();
  for (auto  nd : all_trees) nd->Stump(data);

  arma::vec f_sum_trees     (n,              arma::fill::zeros);
  arma::vec f_sum_trees_test(x_test.n_rows,  arma::fill::zeros);
  arma::vec f_excl          (n,              arma::fill::zeros);
  arma::vec partial_res     (n,              arma::fill::none);
  arma::vec zeros_test      (data.n_test,    arma::fill::zeros);

  double verb;
  unsigned int printevery = 100;
  Rprintf("\nRunning subart (probit, univariate)\n\n");

  for (unsigned int i = 0; i < n_mcmc; i++) {

    if (i % printevery == 0) Rprintf("done %u (out of %u)\n", i, n_mcmc);

    // Update z (marginal TN; sigma^2 = 1 fixed)
    data.sigma_sq = 1.0;
    for (unsigned int ii = 0; ii < n; ii++)
      z[ii] = (y_bin[ii] == 1u) ? rtn_pos(f_sum_trees[ii], 1.0)
                                 : rtn_neg(f_sum_trees[ii], 1.0);
    data.y = z;

    if (data.fit_test) f_sum_trees_test = zeros_test;

    for (unsigned int t = 0; t < data.n_tree; t++) {
      if (data.n_tree > 1) {
        f_excl    = f_sum_trees - trees_fit_store.unsafe_col(t);
        partial_res = data.y - f_excl;
      } else {
        partial_res = z;
      }

      verb = arma::randu(arma::distr_param(0.0, 1.0));
      if (all_trees[t]->isLeaf) verb = 0.1;

      if      (verb < 0.3) grow_uni  (all_trees[t], data, partial_res);
      else if (verb < 0.6) prune_uni (all_trees[t], data, partial_res);
      else                 change_uni(all_trees[t], data, partial_res);

      update_mu_and_predictions_uni(all_trees[t], data,
                                    trees_fit_store, trees_fit_store_test, t);

      f_sum_trees = f_excl + trees_fit_store.unsafe_col(t);
      if (data.fit_test)
        f_sum_trees_test += trees_fit_store_test.unsafe_col(t);
    }
    // No sigma update (sigma^2 = 1 fixed)

    if (i >= n_burn) {
      y_train_hat_post.col(curr) = f_sum_trees;
      if (data.fit_test)
        y_test_hat_post.col(curr) = f_sum_trees_test;

      if (varimportance) {
        arma::uvec vcounts(data.p, arma::fill::zeros);
        for (unsigned int tt = 0; tt < data.n_tree; tt++)
          get_var_counts(all_trees[tt], vcounts);
        var_imp_post.row(curr) = arma::conv_to<arma::rowvec>::from(vcounts);
      }
      curr++;
    }
  }

  for (auto t : all_trees) delete t;
  Rprintf("\nDONE SUBART CLASS (univariate)\n\n");

  return Rcpp::List::create(y_train_hat_post,  //[1]
                             y_test_hat_post,   //[2]
                             var_imp_post);     //[3]
}


// ==================================================================
// cppsubart_CLASS
// d >= 2 probit suBART.
// Tree moves use multivariate leaf posterior (conditional on other z).
// z update is interleaved inside the j-loop (after each j's trees).
// Correlation R updated via Metropolis-Hastings with IW proposal.
// ==================================================================
// [[Rcpp::export]]
Rcpp::List cppsubart_CLASS(arma::mat    x_train,
                            arma::umat   y_bin,
                            arma::mat    x_test,
                            arma::mat    x_cut,
                            unsigned int n_tree,
                            unsigned int node_min_size,
                            unsigned int n_mcmc,
                            unsigned int n_burn,
                            arma::mat    Sigma_init,
                            arma::vec    mu_init,
                            arma::vec    sigma_mu,
                            double alpha, double beta, double nu,
                            arma::mat    S_0_wish,
                            arma::vec    A_j_vec,
                            unsigned int m,
                            arma::uvec   categorical_indicators,
                            bool fit_test,
                            bool varimportance,
                            bool sv_bool,
                            arma::umat   sv_matrix) {

  unsigned int n    = x_train.n_rows;
  unsigned int d    = y_bin.n_cols;
  unsigned int curr = 0;

  // Convert binary y to double matrix kept in data.y_mat (never overwritten)
  arma::mat y_orig = arma::conv_to<arma::mat>::from(y_bin);

  // Latent z: initialise at per-outcome probit(mean)
  arma::mat z_mat(n, d, arma::fill::zeros);
  for (unsigned int j = 0; j < d; j++) {
    double p1 = arma::mean(y_orig.col(j));
    z_mat.col(j).fill(
      Rf_qnorm5(std::max(0.01, std::min(0.99, p1)), 0.0, 1.0, 1, 0));
  }

  // data.y_mat = y_orig (original binary labels); data.R = I, data.D = I
  modelParam data(x_train, y_orig, x_test, x_cut,
                  n_tree, node_min_size, alpha, beta, nu,
                  sigma_mu, Sigma_init, S_0_wish, A_j_vec,
                  n_mcmc, n_burn, categorical_indicators, fit_test);

  if (sv_bool) { data.sv_bool = true; data.sv_matrix = sv_matrix; }

  unsigned int n_post = n_mcmc - n_burn;

  arma::cube y_train_hat_post(n,             d, n_post, arma::fill::zeros);
  arma::cube y_test_hat_post (x_test.n_rows, d, n_post, arma::fill::zeros);
  arma::cube Sigma_post      (d, d, n_post,  arma::fill::zeros);
  arma::cube all_Sigma_post  (d, d, n_mcmc,  arma::fill::zeros);
  arma::cube var_imp_post    (d, data.p, n_post, arma::fill::zeros);

  arma::cube trees_fit_store     (n,             n_tree, d, arma::fill::zeros);
  arma::cube trees_fit_store_test(x_test.n_rows, n_tree, d, arma::fill::zeros);

  std::vector<Node*> all_trees(n_tree * d);
  for (auto& t : all_trees) t = new Node();
  for (auto  nd : all_trees) nd->Stump(data);

  arma::mat f_sum_trees     (n,             d, arma::fill::zeros);
  arma::mat f_sum_trees_test(x_test.n_rows, d, arma::fill::zeros);
  arma::vec f_excl          (n,    arma::fill::zeros);
  arma::vec partial_res     (n,    arma::fill::none);
  arma::vec partial_u       (n,    arma::fill::none);
  arma::vec zeros_test      (x_test.n_rows, arma::fill::zeros);

  // Pre-allocate Sigma partition objects (reused per j)
  arma::rowvec Sigma_j_mj (d - 1, arma::fill::none);
  arma::colvec Sigma_mj_j (d - 1, arma::fill::none);
  arma::mat Sigma_mj_mj   (d - 1, d - 1, arma::fill::none);
  arma::mat Sigma_mj_mj_inv(d - 1, d - 1, arma::fill::none);
  arma::mat scale_mean_aux(1, d - 1, arma::fill::none);

  // Transposed helpers to avoid per-row copies inside id_train loop
  arma::mat z_mj  (n, d - 1, arma::fill::none);
  arma::mat z_mj_t(d - 1, n, arma::fill::none);
  arma::mat f_mj  (n, d - 1, arma::fill::none);
  arma::mat f_mj_t(d - 1, n, arma::fill::none);

  double verb;
  unsigned int curr_tree_counter;
  unsigned int printevery = 100;

  Rprintf("\nRunning subart (probit, multivariate d=%u)\n\n", d);

  for (unsigned int i = 0; i < n_mcmc; i++) {

    if (i % printevery == 0) Rprintf("done %u (out of %u)\n", i, n_mcmc);

    // ---- j-loop: tree updates + interleaved z update ----
    for (unsigned int j = 0; j < d; j++) {

      // Build Sigma partitions using z_mat (latent) for partial_u
      double Sigma_j_j = data.Sigma.at(j, j);
      unsigned int aux = 0;
      for (unsigned int c = 0; c < d; c++) {
        if (c == j) continue;
        Sigma_j_mj[aux] = data.Sigma.at(j, c);
        Sigma_mj_j[aux] = data.Sigma.at(j, c);
        z_mj.unsafe_col(aux) = z_mat.unsafe_col(c);        // latent z, not y_orig
        f_mj.unsafe_col(aux) = f_sum_trees.unsafe_col(c);
        unsigned int aux2 = 0;
        for (unsigned int cc = 0; cc < d; cc++) {
          if (cc == j) continue;
          Sigma_mj_mj.at(aux, aux2) = data.Sigma.at(c, cc);
          aux2++;
        }
        aux++;
      }

      z_mj_t = z_mj.t();
      f_mj_t = f_mj.t();

      Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);
      scale_mean_aux  = Sigma_j_mj * Sigma_mj_mj_inv;  // 1 x (d-1)

      for (unsigned int id_train = 0; id_train < n; id_train++)
        partial_u.at(id_train) = arma::as_scalar(
          scale_mean_aux * (z_mj_t.unsafe_col(id_train) - f_mj_t.unsafe_col(id_train)));

      double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj * (Sigma_mj_mj_inv * Sigma_mj_j));
      data.v_j = v;

      if (data.fit_test) f_sum_trees_test.unsafe_col(j) = zeros_test;

      for (unsigned int t = 0; t < n_tree; t++) {
        curr_tree_counter = t + j * n_tree;

        if (n_tree > 1) {
          f_excl    = f_sum_trees.unsafe_col(j) - trees_fit_store.slice(j).unsafe_col(t);
          partial_res = z_mat.unsafe_col(j) - f_excl;       // residuals from latent z
        } else {
          partial_res = z_mat.unsafe_col(j);
        }

        verb = arma::randu(arma::distr_param(0.0, 1.0));
        if (all_trees[curr_tree_counter]->isLeaf) verb = 0.1;

        if (verb < 0.3) {
          data.move_proposal.at(0)++;
          grow  (all_trees[curr_tree_counter], data, partial_res, partial_u, j);
        } else if (verb < 0.6) {
          data.move_proposal.at(1)++;
          prune (all_trees[curr_tree_counter], data, partial_res, partial_u, j);
        } else {
          data.move_proposal.at(2)++;
          change(all_trees[curr_tree_counter], data, partial_res, partial_u, j);
        }

        update_mu_and_predictions(all_trees[curr_tree_counter], data,
                                  trees_fit_store, trees_fit_store_test, t, j);

        f_sum_trees.unsafe_col(j) = f_excl + trees_fit_store.slice(j).unsafe_col(t);

        if (data.fit_test)
          f_sum_trees_test.unsafe_col(j) += trees_fit_store_test.slice(j).unsafe_col(t);
      }

      // Update z_mat.col(j) via conditional TN (interleaved, inside j-loop)
      update_z_col(z_mat, f_sum_trees, y_orig, j, v, scale_mean_aux, n, d);
    }

    // ---- Metropolis-Hastings update of correlation matrix R ----
    arma::mat sqD   = arma::sqrt(data.D);
    data.W          = sqD * data.R * sqD;
    arma::mat W_prop = arma::iwishrnd((double)m * data.W, m);
    arma::mat D_prop = arma::diagmat(W_prop);
    arma::mat R_prop = arma::inv(arma::sqrt(D_prop)) * W_prop * arma::inv(arma::sqrt(D_prop));

    arma::mat resid = z_mat - f_sum_trees;

    double log_acc = log_post_R(R_prop, D_prop, nu, resid)
                   - log_post_R(data.R, data.D, nu, resid)
                   + log_prop_R(data.R, data.D, R_prop, D_prop, m)
                   - log_prop_R(R_prop, D_prop, data.R, data.D, m);

    if (std::log(R::runif(0.0, 1.0)) < log_acc) {
      data.R = R_prop;
      data.D = D_prop;
    }
    data.Sigma = data.R;   // Sigma is always the correlation matrix (diag = 1)

    all_Sigma_post.slice(i) = data.Sigma;

    if (i >= n_burn) {
      y_train_hat_post.slice(curr) = f_sum_trees;
      if (data.fit_test)
        y_test_hat_post.slice(curr) = f_sum_trees_test;
      Sigma_post.slice(curr) = data.Sigma;

      if (varimportance) {
        for (unsigned int jj = 0; jj < d; jj++) {
          arma::uvec vcounts(data.p, arma::fill::zeros);
          for (unsigned int tt = 0; tt < n_tree; tt++)
            get_var_counts(all_trees[tt + jj * n_tree], vcounts);
          var_imp_post.slice(curr).row(jj) = arma::conv_to<arma::rowvec>::from(vcounts);
        }
      }
      curr++;
    }
  }

  for (auto t : all_trees) delete t;
  Rprintf("\nDONE SUBART CLASS (multivariate)\n\n");

  return Rcpp::List::create(y_train_hat_post,  //[1]
                             y_test_hat_post,   //[2]
                             Sigma_post,        //[3]
                             all_Sigma_post,    //[4]
                             var_imp_post);     //[5]
}
