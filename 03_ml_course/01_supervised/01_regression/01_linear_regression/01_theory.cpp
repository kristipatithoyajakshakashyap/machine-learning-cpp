// 01_theory.cpp
// Lesson: everything a linear model is in one page.
// Theory summary:
//   - A LINEAR MODEL predicts a weighted sum: yhat = w0 + w1 x1 + ... + wp xp.
//     "Linear" means linear in the WEIGHTS, not in the features -- polynomial
//     features still fit a linear model.
//   - OLS finds the w that minimises   J(w) = (1/2n) sum_i (yhat_i - y_i)^2 .
//     Setting the gradient to zero gives the NORMAL EQUATIONS
//       (M'M) w = M'y    where M = [1, X] is the augmented design,
//     solved via partial-pivot Gauss elimination -> w = (M'M)^{-1} M'y.
//   - RIDGE adds lambda * ||w[1:]||^2 (the intercept is never penalised), so
//     (M'M + lambda*I_slopes) w = M'y.  Bigger lambda shrinks slopes toward 0.
//   - LASSO adds lambda * ||w[1:]||_1 instead; the L1 ball makes exact zeros,
//     so lasso SELECTS features while ridge only shrinks them.
// The tiny example below reproduces the normal equations by hand so you can
// verify every number before trusting a real dataset.
//
// Data: 3 toy points, x = 0,1,2  y = 1,3,5.  sklearn equivalents:
// LinearRegression, Ridge(alpha), Lasso(alpha).
//
// EXPECTED OUTPUT:
//   toy y = 1 3 5 
//   M'M      3.0000000 3.0000000      M'y  9.0000000 13.0000000
//            3.0000000 5.0000000
//   ols      w0 1  w1 2
//   ridge(1)   w0 1.666667  w1 1.333333
//   ridge(10)   w0 2.666667  w1 0.3333333
//   lasso(1)   w0 2.5  w1 0.5

#include <iomanip>
#include <iostream>
#include <string>

#include "helper/math/matrix.hpp"
#include "LinearRegression.hpp"

int main() {
  std::cout << std::setprecision(7);

  ml::Mat F = {{0.0}, {1.0}, {2.0}};  // the one feature: x
  ml::Vec y = {1.0, 3.0, 5.0};        // the targets

  // Augmented design matrix M = [1 | x], exactly what fit_ols builds.
  ml::Mat M = ml::make_mat(F.size(), 2);
  for (size_t i = 0; i < F.size(); ++i) {
    M[i][0] = 1.0;
    M[i][1] = F[i][0];
  }
  std::cout << "toy y = ";
  for (double v : y) std::cout << v << " ";
  std::cout << "\n";

  ml::Mat XtX = ml::matmul(ml::transpose(M), M);
  ml::Vec Xty = ml::matvec(ml::transpose(M), y);
  std::cout << "M'M      " << ml::print_vec(XtX[0], 2) << "      M'y  "
            << ml::print_vec(Xty, 2) << "\n";
  std::cout << "         " << ml::print_vec(XtX[1], 2) << "\n";

  ml::LinearRegression lr;
  lr.fit_ols(F, y);
  std::cout << "ols      w0 " << lr.intercept() << "  w1 " << lr.coef()[1]
            << "\n";

  for (double lam : {1.0, 10.0}) {
    ml::LinearRegression r;
    r.fit_ridge(F, y, lam);
    std::cout << "ridge(" << lam << ")   w0 " << r.intercept() << "  w1 "
              << r.coef()[1] << "\n";
  }
  {
    ml::LinearRegression l;
    l.fit_lasso(F, y, 1.0);
    std::cout << "lasso(1)   w0 " << l.intercept() << "  w1 " << l.coef()[1]
              << "\n";
  }
  return 0;
}
