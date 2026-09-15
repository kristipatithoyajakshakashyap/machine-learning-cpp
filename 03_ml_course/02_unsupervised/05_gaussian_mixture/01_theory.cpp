#include <iostream>
int main() {
  std::cout
      << R"LESSON(A Gaussian mixture models a density as a weighted sum of Gaussian components. Responsibilities are soft memberships; component labels are not necessarily semantic classes. EM alternates conditional responsibilities and weighted parameter updates. This implementation uses diagonal covariance, which assumes within-component conditional independence along the supplied axes. Initialization matters and likelihood can increase by collapsing a component without regularization.)LESSON"
      << '\n';
}
