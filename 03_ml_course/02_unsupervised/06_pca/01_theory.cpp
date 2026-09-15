#include <iostream>
int main() {
  std::cout
      << R"LESSON(PCA finds orthogonal directions of maximal centered variance. Retaining leading components is also the optimal rank-k linear reconstruction in squared error. Feature scaling changes the covariance geometry. PCA is unsupervised: a low-variance feature may still carry predictive information, and high explained variance does not imply high predictive accuracy.)LESSON"
      << '\n';
}
