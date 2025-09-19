#include <Rcpp.h>
using namespace Rcpp;


double unnormalize_bart( double z, double a, double b) {
  return((b - a) * (z + 0.5) + a);
}


// [[Rcpp::export]]
NumericMatrix unnormalize_bart_matrix(const NumericMatrix& Z, double a, double b) {
  int n = Z.size();  // total elements
  NumericMatrix out(Z.nrow(), Z.ncol());

  // direct pointers to memory
  const double* zptr = Z.begin();
  double* outptr = out.begin();

  for (int i = 0; i < n; i++) {
    outptr[i] = unnormalize_bart(zptr[i], a, b);
  }

  return out;
}
