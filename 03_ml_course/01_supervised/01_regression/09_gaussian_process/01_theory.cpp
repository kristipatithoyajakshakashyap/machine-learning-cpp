#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(A Gaussian process is a prior over functions: any finite set of function values is jointly Gaussian with covariance given by a kernel. The RBF kernel says nearby inputs have similar outputs, with "nearby" measured in units of the length scale. Conditioning the prior on noisy observations gives a posterior whose mean interpolates the data and whose variance shrinks near observed points and returns to the prior far from them. The marginal likelihood scores kernel hyper-parameters without a validation set.)LESSON"
      << "\n";
  course::GaussianProcess gp(1.0, 1.0, 1e-4);
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0.5, -0.5, -0.5, 0.5};
  gp.fit(X, y);
  std::cout << "posterior mean at training points " << ml::print_vec(gp.predict(X))
            << "\nposterior std at 0 and at 10: "
            << ml::print_vec(gp.predict_std({{0}, {10}})) << "\n";
}
