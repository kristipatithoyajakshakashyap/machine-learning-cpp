// 03_ml_course/01_supervised/03_model_diagnostics/03_hyperparameter_search/02_math_intuition.cpp
//
// Purpose : Lesson 2 - the two numbers behind hyper-parameter search: how
//           many random draws are needed to hit a top-q region of the grid,
//           and how optimistic "best of m noisy CV scores" is. Ends by
//           decoding one index of the module's 3 x 3 ParamSpace.
// Inputs  : none (closed-form arithmetic only); no defines used.
// Outputs : prints only.
// Run     : target hps_math_intuition (no arguments).
#include "Model.hpp"
#include "helper/eval/search.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
// Print-only: how many draws random search needs to land in the top q
// fraction of a grid, and the optimistic bias of "best of m noisy scores".
int main() {
  std::cout
      << R"LESSON(Random search: if a fraction q of the space is "good enough", the chance that n independent draws all miss it is (1-q)^n. Sixty draws hit a top-5% region with probability 1-0.95^60 ~ 0.95, regardless of how many parameters there are. Optimistic bias: the maximum of m unbiased noisy scores is biased upward; with m candidates of equal true quality and fold noise sigma, E[max] grows roughly like sigma*sqrt(2 ln m). Nested CV removes that bias by scoring the selected configuration on rows the selection never saw.)LESSON"
      << "\n";
  std::cout << std::fixed << std::setprecision(3);
  // P(at least one of n draws lands in the top q fraction) = 1 - (1 - q)^n.
  for (double q : {0.05, 0.10, 0.20})
    for (size_t n : {10, 30, 60})
      std::cout << "P(hit top " << q * 100 << "% in " << n
                << " draws) = " << 1.0 - std::pow(1.0 - q, double(n)) << "\n";
  // Upper bound on E[max of m standard normals]: sqrt(2 ln m) (0 for m = 1).
  for (size_t m : {1, 4, 9, 36})
    std::cout << "expected max of " << m << " N(0,1) scores ~ "
              << (m == 1 ? 0.0 : std::sqrt(2.0 * std::log(double(m)))) << " sigma (upper bound)\n";
  // ParamSpace enumerates the cartesian product; at(5) shows the decoding.
  ml::ParamSpace space{{{"max_depth", {4, 8, 16}}, {"min_leaf", {1, 2, 5}}}};
  std::cout << "space size " << space.size() << "; index 5 decodes to "
            << ml::params_to_string(space.at(5)) << "\n";
}
