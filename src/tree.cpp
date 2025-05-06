#include "subart_classes.h"
#include <RcppArmadillo.h>

void Node::Stump(modelParam& data){

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

  Gamma_j  = n_leaf+data.v_j/data.sigma_mu_j_sq;
  S_j = r_sum-u_sum;

  return;

}

void Node::nodeLogLike(modelParam& data, unsigned int& j){

  // Getting the log-likelihood;
  log_likelihood = -0.5*log(2*arma::datum::pi*data.sigma_mu_j_sq.at(j))+0.5*log(data.v_j/Gamma_j) +0.5*(S_j*S_j)/(data.v_j*Gamma_j);
  return;

}

void Node::addingLeaves(modelParam &data){

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

void Node::grow(Node *tree,
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
  if(tree->isRoot){
    g_node = &tree[0];
  } else {
    g_node = t_nodes[arma::randi(arma::distr_param(0,(number_leaves-1)))];
  }

  for(auto& leaf:t_nodes){
    leaf->updateResiduals(data,curr_res,curr_u,j);
  }

  // Update the residuals and return the same tree
  if(g_node->n_leaf < 2){
    return;
  }

  // Selecting a splitting variable and a split rule
  //(explore the logic of selecting a good candidate for the split rule)
  g_node->var_split_rule = 1.0;


  // Assigned left and right for the current train index
  arma::vec left_id = train_index;
  arma::vec right_id  = train_index;
  int left_id_counter = 0;
  unsigned int right_id_counter = 0;


  double r_sum_left = 0.0;
  double u_sum_left = 0.0;

  double r_sum_right = 0.0;
  double u_sum_right = 0.0;

  for(auto& id:train_index){

    // Here I will update the r_sum and u_sum to avoid to go over through the same iterations when doing left->updateResiduals()
    if(data.x_train.at(id,g_node->var_split) <= g_node->var_split_rule ){

      left_id[left_id_counter] = id;
      r_sum_left = r_sum_left + curr_res[id];
      u_sum_left = u_sum_left + curr_res[id];

      left_id_counter++;
    } else {

      right[right_id_counter] = id;
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

  g_node->addingLeaves(data)

  // Calculate sufficient statistics for the left and right node outside the UpdateResiduals function.

  // ,,,,,,,,,,,,,,,,,,,,,,,,,,,
  // ,,,, continue from here ,,,
  // ,,,,,,,,,,,,,,,,,,,,,,,,,,,
  left_id.resize(left_id_counter); // Maybe in a future think a way of using arma::set_size? which is much faster
  right_id.resize(right_id_counter); // Maybe in the future thinkk in a way of us arma::set_size? which is much faster.


}
