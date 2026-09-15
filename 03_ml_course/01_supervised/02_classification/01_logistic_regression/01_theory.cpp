// 01_theory.cpp
// Lesson: logistic regression in one page.
// Theory summary:
//   - Linear regression predicts on (-inf, +inf); classification needs a
//     PROBABILITY in [0,1].  The logistic (sigmoid) link does that:
//       p(class=1 | x) = 1 / (1 + exp(-(w.x + b)))
//     A linear score w.x + b is therefore a LOG-ODDS: log(p/(1-p)).
//   - The training loss is cross-entropy (the negative log-likelihood of the
//     labels under the sigmoid).  Its gradient is beautiful:
//       dL/dw = (1/n) sum_i (p_i - y_i) x_i
//     i.e. we slide w uphill on the misclassification residuals, exactly like
//     linear regression slides on raw residuals.
//   - Here we fit a 1-D toy: features increase smoothly (0,0.5,1,1.5,2) with
//     labels (0,0,1,1,1); a glance at the sigmoid curve shows the probability
//     crossing 0.5 between x=1 and x=1.5.
// sklearn equivalent: sklearn.linear_model.LogisticRegression.
//
// EXPECTED OUTPUT:
//   toy: x 0 0.5 1 1.5 2   y 0 0 1 1 1
//   intercept -11.93368  slope 15.99875
//   p(class=1): 6.56548e-06 0.01918406 0.9831277 0.9999942 1

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "LogisticRegression.hpp"

int main() {
  std::cout << std::setprecision(7);
  std::cout << "toy: x ";
  ml::Mat X;
  ml::Vec y = {0, 0, 1, 1, 1};
  for (double v : {0.0, 0.5, 1.0, 1.5, 2.0}) {
    std::cout << v << " ";
    X.push_back({v});
  }
  std::cout << "  y 0 0 1 1 1\n";

  ml::LogisticRegression lr(2, 1.0);   // binary, C = 1
  lr.fit(X, y);

  std::cout << "intercept " << lr.intercept()[0] << "  slope "
            << lr.feature_weights(0)[0] << "\n";

  std::cout << "p(class=1):";
  auto P = lr.predict_proba(X);
  for (const auto& row : P) std::cout << " " << row[1];
  std::cout << "\n";
  return 0;
}
