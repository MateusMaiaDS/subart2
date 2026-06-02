#ifndef TREE_H
#define TREE_H

#include "subart_classes.h"

void get_leaves(Node* x,  std::vector<Node*> &leaves_vec);
void get_nogs(Node* node,std::vector<Node*>& nogs);
double sample_split_var_rule_from_xcut(arma::vec& xcut_col, double lower_candidate, double upper_candidate);

// Count variable usage across all internal nodes of a tree
void get_var_counts(Node* tree, arma::uvec& counts);

void grow(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec &curr_u,unsigned int &j);
void prune(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u,unsigned int &j);
void change(Node* tree, modelParam &data, arma::vec&curr_res, arma::vec &curr_u,unsigned int &j);


#endif
