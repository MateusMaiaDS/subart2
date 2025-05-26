#include <RcppArmadillo.h>
#include "subart_classes.h"

void Node::Stump(modelParam& data){

  left = this;
  right = this;
  parent = this;

  n_leaf = data.n;
  n_leaf_test = data.n_test;
  train_index = data.init_train_index;
  test_index = data.init_test_index;

  return;
}

// Getting the leaves (this is the function that gonna do the recursion the
//                      function below is the one that gonna initialise it)
void get_leaves(Node* x,  std::vector<Node*> &leaves_vec) {

  if(x->isLeaf){
    leaves_vec.push_back(x);
  } else {
    get_leaves(x->left, leaves_vec);
    get_leaves(x->right,leaves_vec);
  }

  return;

}

// Sweeping the trees looking for nogs
void get_nogs(Node* node,std::vector<Node*>& nogs){

  if(!node->isLeaf){
    bool bool_left_is_leaf = node->left->isLeaf;
    bool bool_right_is_leaf = node->right->isLeaf;

    // Checking if the current one is a NOGs
    if(bool_left_is_leaf && bool_right_is_leaf){
      nogs.push_back(node);
    } else { // Keep looking for other NOGs
      get_nogs(node->left,nogs);
      get_nogs(node->right,nogs);
    }
  }
  return;
}

// Calculating the Loglilelihood of a node
void Node::updateResiduals(modelParam& data,
                           arma::vec &curr_res,
                           arma::vec &curr_u,
                           unsigned int &j){


  r_sum = 0.0;
  u_sum = 0.0;

  // Train elements
  for(auto id:train_index){
    r_sum = r_sum + curr_res.at(id);
    u_sum = u_sum + curr_u.at(id);
  }

  Gamma_j  = n_leaf+data.v_j/data.sigma_mu_j_sq.at(j);
  S_j = r_sum-u_sum;

  return;

}

void Node::nodeLogLike(modelParam& data, unsigned int& j){

  // Getting the log-likelihood;
  log_likelihood = -0.5*log(2*arma::datum::pi*data.sigma_mu_j_sq.at(j))+0.5*log(data.v_j/Gamma_j) +0.5*(S_j*S_j)/(data.v_j*Gamma_j);
  return;

}

void Node::addingLeaves(){

  // Create the two new nodes
  left = new Node(); // Creating a new vector object to the
  right = new Node();
  isLeaf = false;

  // Modifying the left node
  left -> isRoot = false;
  left -> isLeaf = true;
  left -> left = left;
  left -> right = left;
  left -> parent = this;
  left -> depth = depth+1;

  right -> isRoot = false;
  right -> isLeaf = true;
  right -> left = right; // Recall that you are saving the address of the right node.
  right -> right = right;
  right -> parent = this;
  right -> depth = depth+1;


  return;
}


void Node::deletingLeaves(){

  // Deleting
  delete left; // This release the memory from the left point
  delete right; // This release the memory from the right point
  left = this;  // The new pointer for the left become the node itself
  right = this; // The new pointer for the right become the node itself
  isLeaf = true;

  return;

}

void Node::getLimits(unsigned int split_var_candidate,
                      double &lower_candidate,
                      double &upper_candidate){

  Node* dummy_node = this;
  lower_candidate = 0.0;
  upper_candidate = 1.0;

  bool node_bool = dummy_node->isRoot ? false : true;
  while(node_bool) {

    bool is_left = dummy_node->isLeft();
    dummy_node = dummy_node->parent;
    node_bool = dummy_node->isRoot ? false : true;

    if(dummy_node->var_split == split_var_candidate){
      node_bool = false; // This is false because all the other parents from it will already carry the information from lower and upper from previous nodes

      if(is_left){
        upper = dummy_node->var_split_rule; //This is simple, think about a simple tree wiht two nodes, if the rule from the parent is x_{1}<5, and we are trying to grow its children, if the left node all values should be already below to 5
        lower = dummy_node->lower;
      }  else {
        upper = dummy_node->upper;
        lower = dummy_node->var_split_rule;// Same logic as before, as all values are above 5 the lower limit become its value
      }
    }
  }

}

double sample_split_var_rule_from_xcut(arma::vec& xcut_col, double lower_candidate, double upper_candidate) {
  const double* start = std::lower_bound(xcut_col.begin(), xcut_col.end(), lower_candidate + std::numeric_limits<double>::epsilon());
  const double* end = std::lower_bound(xcut_col.begin(), xcut_col.end(), upper_candidate);

  arma::uword length = end - start;

  if (length == 0) {
    return -1.0; // No value in range
  }

  // Uniform discrete sampling: generate an index in [0, length-1]
  arma::uword random_index = arma::randi<arma::uword>(arma::distr_param(0, length - 1));
  return *(start + random_index);
}

void grow(Node *tree,
          modelParam &data,
          arma::vec &curr_res,
          arma::vec &curr_u,
          unsigned int &j){


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
    leaf->updateResiduals(data,curr_res,curr_u,j);
  }

  // Residuals already updted, if the selected g_node only have 2/less observations there's no point of growing it
  if(g_node->n_leaf <= 2){
    return;
  }

  // Selecting a splitting variable and a split rule
  //(explore the logic of selecting a good candidate for the split rule)
  unsigned int var_split_candidate = arma::randi<arma::uword>(arma::distr_param(0, data.d - 1));


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
  double u_sum_left = 0.0;

  double r_sum_right = 0.0;
  double u_sum_right = 0.0;

  for(auto& id:g_node->train_index){

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
  Gamma_j_left = left_id_counter+data.v_j/data.sigma_mu_j_sq[j];
  S_j_left = r_sum_left-u_sum_left;

  Gamma_j_right = right_id_counter+data.v_j/data.sigma_mu_j_sq[j];
  S_j_right = r_sum_right-u_sum_right;


  // Calculate sufficient statistics for the left and right node outside the UpdateResiduals function.

  double new_tree_log_likelihood_ratio = -0.5*log(2*arma::datum::pi*data.sigma_mu_j_sq[j])+0.5*log(data.v_j)+ // Remaninig from the operation like_left_node + like_right_node - like_g_node
                                       (-0.5*log(Gamma_j_left) + 0.5*(S_j_left*S_j_left)/(data.v_j*Gamma_j_left)) + // Core of the likelihood of the left node
                                       (-0.5*log(Gamma_j_right) + 0.5*(S_j_right*S_j_right)/(data.v_j*Gamma_j_right))- // Core of the likelihood of the left node
                                       (-0.5*log(g_node->Gamma_j) + 0.5*(g_node->S_j*g_node->S_j)/(data.v_j*g_node->Gamma_j)); // Core of the likelihood for the grown node

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

  } else {

    // Not need to modify anything all the nodes are already updated

  }


  return;

}

void prune(Node *tree,
          modelParam &data,
          arma::vec &curr_res,
          arma::vec &curr_u,
          unsigned int &j){


  // Getting the number of terminal nodes
  std::vector<Node*> t_nodes(0);
  std::vector<Node*> nog_nodes(0);

  get_leaves(tree,t_nodes);
  get_nogs(tree,nog_nodes);

  Node* p_node;
  unsigned int number_leaves = t_nodes.size();
  unsigned int number_nogs = nog_nodes.size();

  // If the tree os a rooot
  if(tree->isRoot){
    p_node = tree;
  } else {
    p_node = t_nodes[arma::randi(arma::distr_param(0,(number_nogs-1)))];
  }

  for(auto& leaf:t_nodes){
    leaf->updateResiduals(data,curr_res,curr_u,j);
  }


  // Calculating the likelhood for the node selected to be grown
  double r_sum = 0.0;
  double u_sum = 0.0;

  for(auto& id:p_node->train_index){
    r_sum = r_sum + curr_res[id];
    u_sum = u_sum + curr_res[id];
  }


  // Calculating sufficient statistics for left and right nodes
  double p_Gamma_j;
  double p_S_j;

  // Updating other sufficientStatistics;
  p_Gamma_j = p_node->n_leaf+data.v_j/data.sigma_mu_j_sq[j];
  p_S_j = r_sum-u_sum;

  // Calculate sufficient statistics for the left and right node outside the UpdateResiduals function.

  double new_tree_log_likelihood_ratio = 0.5*log(2*arma::datum::pi*data.sigma_mu_j_sq[j])-0.5*log(data.v_j)+ // Remaninig from the operation  like_g_node - (like_left_node + like_right_node -)
    (-0.5*log(p_Gamma_j) + 0.5*(p_S_j*p_S_j)/(data.v_j*p_Gamma_j))- // Core of the likelihood for the prune node
    (-0.5*log(p_node->left->Gamma_j) + 0.5*(p_node->left->S_j*p_node->left->S_j)/(data.v_j*p_node->left->Gamma_j)) - // Core of the likelihood of the left node
    (-0.5*log(p_node->right->Gamma_j) + 0.5*(p_node->right->S_j*p_node->right->S_j)/(data.v_j*p_node->right->Gamma_j)); // Core of the likelihood of the left node

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
