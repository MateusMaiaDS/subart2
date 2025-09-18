#ifndef TREE_UNIVARIATE_H
#define TREE_UNIVARIATE_H

#include "subart_classes.h"


void grow_uni(Node* tree, modelParam_uni &data, arma::vec &curr_res);
void prune_uni(Node* tree, modelParam_uni &data, arma::vec &curr_res);
void change_uni(Node* tree, modelParam_uni &data, arma::vec &curr_res);


#endif
