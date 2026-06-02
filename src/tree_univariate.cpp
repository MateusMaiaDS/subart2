#include <RcppArmadillo.h>
#include "subart_classes.h"
#include "tree.h"
#include "tree_univariate.h"



// Calculating the Loglilelihood of a node
void Node::updateResiduals_uni(modelParam_uni& data,
                           arma::vec &curr_res){

  r_sum = 0.0;

  // Train elements
  for(auto id:train_index){
    r_sum = r_sum + curr_res.at(id);
  }

  Gamma_j  = n_leaf+data.sigma_sq/data.sigma_mu_sq;
  S_j = r_sum;

  return;

}

void Node::nodeLogLike_uni(modelParam_uni& data){

  // Getting the log-likelihood;
  log_likelihood = -0.5*log(2*arma::datum::pi*data.sigma_mu_sq)+0.5*log(data.sigma_sq/Gamma_j) +0.5*(S_j*S_j)/(data.sigma_sq*Gamma_j);
  return;

}


void grow_uni(Node *tree,
          modelParam_uni &data,
          arma::vec &curr_res){


  // Getting the number of terminal nodes
  std::vector<Node*> t_nodes(0);
  std::vector<Node*> nog_nodes(0);

  get_leaves(tree,t_nodes);
  get_nogs(tree,nog_nodes);

  Node* g_node;
  unsigned int number_leaves = t_nodes.size();

  // If the tree os a rooot
  if(tree->isRoot & tree->isLeaf){
    g_node = tree;
  } else {
    g_node = t_nodes[arma::randi(arma::distr_param(0,(number_leaves-1)))];
  }

  for(auto& leaf:t_nodes){
    leaf->updateResiduals_uni(data,curr_res);
  }

  // Residuals already updted, if the selected g_node only have 2/less observations there's no point of growing it
  if(g_node->n_leaf <= 2){
    return;
  }

  // Selecting a splitting variable, respecting specify_variables if active
  unsigned int var_split_candidate;
  if (data.sv_bool) {
    arma::uvec allowed = arma::find(data.sv_matrix.row(0) == 1);
    if (allowed.is_empty()) return;
    var_split_candidate = allowed[arma::randi<arma::uword>(arma::distr_param(0, (int)(allowed.n_elem - 1)))];
  } else {
    var_split_candidate = arma::randi<arma::uword>(arma::distr_param(0, data.p - 1));
  }


  double lower_candidate;
  double upper_candidate;

  // // Obtaining the limits
  g_node->getLimits(var_split_candidate,
                    lower_candidate,
                    upper_candidate);

  // No valid split_var_cutpoint is available
  if(lower_candidate==upper_candidate){
    return;
  }

  arma::vec var_split_rule_numcuts = data.xcut.col(var_split_candidate);

  double var_split_rule_candidate = sample_split_var_rule_from_xcut(var_split_rule_numcuts,
                                                                    lower_candidate,
                                                                    upper_candidate); //

  // If not valid split vars are found
  if(var_split_candidate==-1.0){
    return;
  }

  // Assigned left and right for the current train index
  arma::uvec left_id = g_node->train_index;
  arma::uvec right_id  = g_node->train_index;
  unsigned int left_id_counter = 0;
  unsigned int right_id_counter = 0;


  double r_sum_left = 0.0;

  double r_sum_right = 0.0;

  for(auto& id:g_node->train_index){

    // Here I will update the r_sum and u_sum to avoid to go over through the same iterations when doing left->updateResiduals()
    if(data.x_train.at(id,var_split_candidate) <= var_split_rule_candidate ){

      left_id[left_id_counter] = id;
      r_sum_left = r_sum_left + curr_res[id];

      left_id_counter++;
    } else {

      right_id[right_id_counter] = id;
      r_sum_right = r_sum_right + curr_res[id];
      right_id_counter++;

    }

  }


  if(left_id_counter==0 || right_id_counter==0) {
    return; // Exit the grow move as the new grow is not valid.
  }

  // Calculating sufficient statistics for left and right nodes
  double Gamma_j_left;
  double S_j_left;

  double Gamma_j_right;
  double S_j_right;


  // Updating other sufficientStatistics;
  Gamma_j_left = left_id_counter+data.sigma_sq/data.sigma_mu_sq;
  S_j_left = r_sum_left;

  Gamma_j_right = right_id_counter+data.sigma_sq/data.sigma_mu_sq;
  S_j_right = r_sum_right;


  // Calculate sufficient statistics for the left and right node outside the UpdateResiduals function.

  double new_tree_log_likelihood_ratio = -0.5*log(2*arma::datum::pi*data.sigma_mu_sq)+0.5*log(data.sigma_sq)+ // Remaninig from the operation like_left_node + like_right_node - like_g_node
    (-0.5*log(Gamma_j_left) + 0.5*(S_j_left*S_j_left)/(data.sigma_sq*Gamma_j_left)) + // Core of the likelihood of the left node
    (-0.5*log(Gamma_j_right) + 0.5*(S_j_right*S_j_right)/(data.sigma_sq*Gamma_j_right))- // Core of the likelihood of the left node
    (-0.5*log(g_node->Gamma_j) + 0.5*(g_node->S_j*g_node->S_j)/(data.sigma_sq*g_node->Gamma_j)); // Core of the likelihood for the grown node

  // Reminder the node_log_likelihood is:
  // node->log_likelihood = -0.5*log(2*arma::datum::pi*sigma_mu_j_sq[j])+0.5*log(data.v_j/node->Gamma_j) +0.5*(S_j*S_j)/(data.v_j*node->Gamma_j);


  // Computing the tree prior
  double log_tree_prior_ratio = log(data.alpha*pow(1+g_node->depth,-data.beta)) +
    2*log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) -
    log(1-data.alpha*pow(1+g_node->depth,-data.beta));

  // Getting the transition probability
  double log_transition_prob_ratio = log((0.3)/(nog_nodes.size()+1)) - log(0.3/t_nodes.size()); // 0.3 and 0.3 are the prob of Pru. and Grow, respectively

  // Calculating the acceptance ratio
  double acceptance = exp(new_tree_log_likelihood_ratio + log_tree_prior_ratio + log_transition_prob_ratio);


  if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){

    left_id.resize(left_id_counter); // Maybe in a future think a way of using arma::set_size? which is much faster
    right_id.resize(right_id_counter); // Maybe in the future thinkk in a way of us arma::set_size? which is much faster.

    // Updating the g_node
    g_node->addingLeaves();
    g_node->var_split = var_split_candidate;
    g_node->var_split_rule = var_split_rule_candidate;
    g_node->upper = upper_candidate;
    g_node->lower = lower_candidate;

    // Updating sufficient statistics for the left node
    g_node->left->train_index = left_id;
    g_node->left->n_leaf = left_id_counter;
    // g_node->left->test_index = left_id_test; // TODO: implement the test index here
    g_node->left->S_j = S_j_left;
    g_node->left->Gamma_j = Gamma_j_left;

    // Updating sufficient statistics for the right node
    g_node->right->train_index = right_id;
    g_node->right->n_leaf = right_id_counter;
    // g_node->right->test_index = right_id_test; // TODO: implement the test index here
    g_node->right->S_j = S_j_right;
    g_node->right->Gamma_j = Gamma_j_right;


    if(data.fit_test){

      // Assigned left and right for the current train index
      arma::uvec left_id_test = g_node->test_index;
      arma::uvec right_id_test  = g_node->test_index;
      unsigned int left_id_counter_test = 0;
      unsigned int right_id_counter_test = 0;

      for(auto& id_test:g_node->test_index){

        // Here I will update the r_sum and u_sum to avoid to go over through the same iterations when doing left->updateResiduals()
        if(data.x_test.at(id_test,var_split_candidate) <= var_split_rule_candidate ){

          left_id_test[left_id_counter_test] = id_test;
          left_id_counter_test++;
        } else {

          right_id_test[right_id_counter_test] = id_test;
          right_id_counter_test++;

        }

      }

      left_id_test.resize(left_id_counter_test); // Maybe in a future think a way of using arma::set_size? which is much faster
      right_id_test.resize(right_id_counter_test); // Maybe in the future thinkk in a way of us arma::set_size? which is much faster

      g_node->left->test_index = left_id_test;
      g_node->right->test_index = right_id_test;

    }


  } else {

    // Not need to modify anything all the nodes are already updated

  }


  return;

}

void prune_uni(Node *tree,
           modelParam_uni &data,
           arma::vec &curr_res){


  // Getting the number of terminal nodes
  std::vector<Node*> t_nodes(0);
  std::vector<Node*> nog_nodes(0);

  get_leaves(tree,t_nodes);
  get_nogs(tree,nog_nodes);

  Node* p_node;
  unsigned int number_leaves = t_nodes.size();
  unsigned int number_nogs = nog_nodes.size();

  if(number_nogs == 0){
    return; // Nothing to prune (stump)
  } else if(number_nogs == 1){
    p_node = nog_nodes[0]; // Only one NOG (often the root for a shallow tree)
  } else {
    p_node = nog_nodes[arma::randi(arma::distr_param(0,(int)(number_nogs-1)))];
  }

  for(auto& leaf:t_nodes){
    leaf->updateResiduals_uni(data,curr_res);
  }


  // Calculating the likelhood for the node selected to be grown
  double r_sum = 0.0;

  for(auto& id:p_node->train_index){
    r_sum = r_sum + curr_res[id];
  }


  // Calculating sufficient statistics for left and right nodes
  double p_Gamma_j;
  double p_S_j;

  // Updating other sufficientStatistics;
  p_Gamma_j = p_node->n_leaf+data.sigma_sq/data.sigma_mu_sq;
  p_S_j = r_sum;

  // Calculate sufficient statistics for the left and right node outside the UpdateResiduals function.

  double new_tree_log_likelihood_ratio = 0.5*log(2*arma::datum::pi*data.sigma_mu_sq)-0.5*log(data.sigma_sq)+ // Remaninig from the operation  like_g_node - (like_left_node + like_right_node -)
    (-0.5*log(p_Gamma_j) + 0.5*(p_S_j*p_S_j)/(data.sigma_sq*p_Gamma_j))- // Core of the likelihood for the prune node
    (-0.5*log(p_node->left->Gamma_j) + 0.5*(p_node->left->S_j*p_node->left->S_j)/(data.sigma_sq*p_node->left->Gamma_j)) - // Core of the likelihood of the left node
    (-0.5*log(p_node->right->Gamma_j) + 0.5*(p_node->right->S_j*p_node->right->S_j)/(data.sigma_sq*p_node->right->Gamma_j)); // Core of the likelihood of the left node

  // Reminder the node_log_likelihood is:
  // node->log_likelihood = -0.5*log(2*arma::datum::pi*sigma_mu_j_sq[j])+0.5*log(data.v_j/node->Gamma_j) +0.5*(S_j*S_j)/(data.v_j*node->Gamma_j);


  // Computing the tree prior
  double log_tree_prior_ratio = log(1-data.alpha*pow((1+p_node->depth),-data.beta))-  //Prior of the p_node being terminal
    log(data.alpha*pow((1+p_node->depth),-data.beta)) - // Prior of the p_node being non-terminal
    2*log(1-data.alpha*pow((1+p_node->depth+1),-data.beta)) ;  // Prior of the right & left noide being terminal

  // Calculating the transition loglikelihood
  double log_transition_ratio = log((0.3)/(number_leaves)) - log((0.3)/(number_nogs));

  // Calculating the acceptance ratio
  double acceptance = exp(new_tree_log_likelihood_ratio + log_tree_prior_ratio + log_transition_ratio);


  if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){

    // Updating the g_node
    p_node->S_j = p_S_j;
    p_node->Gamma_j = p_Gamma_j;
    p_node->deletingLeaves();

  } else {

    // Not need to modify anything all the nodes are already updated

  }


  return;

}



void change_uni(Node *tree,
            modelParam_uni &data,
            arma::vec &curr_res){


  // Getting the number of terminal nodes
  std::vector<Node*> t_nodes(0);
  std::vector<Node*> nog_nodes(0);

  get_leaves(tree,t_nodes);
  get_nogs(tree,nog_nodes);

  Node* c_node;
  unsigned int number_leaves = t_nodes.size();

  // If the tree os a root
  if(nog_nodes.size() == 0){
    return; // No NOGs to change (stump)
  } else if(nog_nodes.size()==1){
    c_node = nog_nodes[0];
  } else {
    c_node = nog_nodes[arma::randi(arma::distr_param(0,(int)(nog_nodes.size()-1)))];
  }

  for(auto& leaf:t_nodes){
    leaf->updateResiduals_uni(data,curr_res);
  }


  // Selecting a splitting variable, respecting specify_variables if active
  unsigned int var_split_candidate;
  if (data.sv_bool) {
    arma::uvec allowed = arma::find(data.sv_matrix.row(0) == 1);
    if (allowed.is_empty()) return;
    var_split_candidate = allowed[arma::randi<arma::uword>(arma::distr_param(0, (int)(allowed.n_elem - 1)))];
  } else {
    var_split_candidate = arma::randi<arma::uword>(arma::distr_param(0, data.p - 1));
  }


  double lower_candidate;
  double upper_candidate;

  // // Obtaining the limits
  c_node->getLimits(var_split_candidate,
                    lower_candidate,
                    upper_candidate);

  // No valid split_var_cutpoint is available
  if(lower_candidate==upper_candidate){
    return;
  }

  arma::vec var_split_rule_numcuts = data.xcut.col(var_split_candidate);

  double var_split_rule_candidate = sample_split_var_rule_from_xcut(var_split_rule_numcuts,
                                                                    lower_candidate,
                                                                    upper_candidate); //

  // If not valid split vars are found
  if(var_split_rule_candidate==-1.0){
    return;
  }

  // Assigned left and right for the current train index
  arma::uvec left_id = c_node->train_index;
  arma::uvec right_id  = c_node->train_index;
  unsigned int left_id_counter = 0;
  unsigned int right_id_counter = 0;


  double r_sum_left = 0.0;
  double u_sum_left = 0.0;

  double r_sum_right = 0.0;
  double u_sum_right = 0.0;

  for(auto& id:c_node->train_index){

    // Here I will update the r_sum and u_sum to avoid to go over through the same iterations when doing left->updateResiduals()
    if(data.x_train.at(id,var_split_candidate) <= var_split_rule_candidate ){

      left_id[left_id_counter] = id;
      r_sum_left = r_sum_left + curr_res[id];
      u_sum_left = u_sum_left + curr_res[id];

      left_id_counter++;
    } else {

      right_id[right_id_counter] = id;
      r_sum_right = r_sum_right + curr_res[id];
      u_sum_right = u_sum_right + curr_res[id];
      right_id_counter++;

    }

  }


  if(left_id_counter==0 || right_id_counter==0) {
    return; // Exit the grow move as the new grow is not valid.
  }

  // Calculating sufficient statistics for left and right nodes
  double Gamma_j_left;
  double S_j_left;

  double Gamma_j_right;
  double S_j_right;


  // Updating other sufficientStatistics;
  double sigma_sq_divided_sigma_mu_sq = data.sigma_sq/data.sigma_mu_sq;
  Gamma_j_left = left_id_counter+sigma_sq_divided_sigma_mu_sq;
  S_j_left = r_sum_left;

  Gamma_j_right = right_id_counter+sigma_sq_divided_sigma_mu_sq;
  S_j_right = r_sum_right;


  // Calculate sufficient statistics for the left and right node outside the UpdateResiduals function.

  double new_tree_log_likelihood_ratio = (-0.5*log(Gamma_j_left) + 0.5*(S_j_left*S_j_left)/(data.sigma_sq*Gamma_j_left)) + // Core of the likelihood of the left node
    (-0.5*log(Gamma_j_right) + 0.5*(S_j_right*S_j_right)/(data.sigma_sq*Gamma_j_right))- // Core of the likelihood of the left node
    (-0.5*log(c_node->left->Gamma_j) + 0.5*(c_node->left->S_j*c_node->left->S_j)/(data.sigma_sq*c_node->left->Gamma_j)) -
    (-0.5*log(c_node->right->Gamma_j) + 0.5*(c_node->right->S_j*c_node->right->S_j)/(data.sigma_sq*c_node->right->Gamma_j)) ; // Core of the current left node


  // Reminder the node_log_likelihood is:
  // node->log_likelihood = -0.5*log(2*arma::datum::pi*sigma_mu_j_sq[j])+0.5*log(data.v_j/node->Gamma_j) +0.5*(S_j*S_j)/(data.v_j*node->Gamma_j);

  // Calculating the acceptance ratio
  double acceptance = exp(new_tree_log_likelihood_ratio); // Remember tree prior and log-likelihood just cancel out themselves for the CHANGE;


  if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){

    left_id.resize(left_id_counter); // Maybe in a future think a way of using arma::set_size? which is much faster
    right_id.resize(right_id_counter); // Maybe in the future thinkk in a way of us arma::set_size? which is much faster.

    // Updating the g_node
    c_node->var_split = var_split_candidate;
    c_node->var_split_rule = var_split_rule_candidate;
    c_node->upper = upper_candidate;
    c_node->lower = lower_candidate;

    // Updating sufficient statistics for the left node
    c_node->left->train_index = left_id;
    c_node->left->n_leaf = left_id_counter;
    // g_node->left->test_index = left_id_test; // TODO: implement the test index here
    c_node->left->S_j = S_j_left;
    c_node->left->Gamma_j = Gamma_j_left;

    // Updating sufficient statistics for the right node
    c_node->right->train_index = right_id;
    c_node->right->n_leaf = right_id_counter;
    // g_node->right->test_index = right_id_test; // TODO: implement the test index here
    c_node->right->S_j = S_j_right;
    c_node->right->Gamma_j = Gamma_j_right;


    if(data.fit_test){

      // Assigned left and right for the current train index
      arma::uvec left_id_test = c_node->test_index;
      arma::uvec right_id_test  = c_node->test_index;
      unsigned int left_id_counter_test = 0;
      unsigned int right_id_counter_test = 0;

      for(auto& id_test:c_node->test_index){

        // Here I will update the r_sum and u_sum to avoid to go over through the same iterations when doing left->updateResiduals()
        if(data.x_test.at(id_test,var_split_candidate) <= var_split_rule_candidate ){

          left_id_test[left_id_counter_test] = id_test;
          left_id_counter_test++;
        } else {

          right_id_test[right_id_counter_test] = id_test;
          right_id_counter_test++;

        }

      }

      left_id_test.resize(left_id_counter_test); // Maybe in a future think a way of using arma::set_size? which is much faster
      right_id_test.resize(right_id_counter_test); // Maybe in the future thinkk in a way of us arma::set_size? which is much faster

      c_node->left->test_index = left_id_test;
      c_node->right->test_index = right_id_test;

    }


  } else {

    // Not need to modify anything all the nodes are already updated

  }


  return;

}

