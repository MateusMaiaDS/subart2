#include "subart_classes.h"
#include <RcppArmadillo.h>

void Node::Stump(modelParam& data){

  n_leaf = data.n;
  n_leaf_test = data.n_test;
  train_index = data.init_train_index;
  test_index = data.init_test_index;

  return;
}
