#include <iostream>
int main() {
  std::cout
      << R"LESSON(C=X_centered^T X_centered/(n-1). C v_j=lambda_j v_j, with unit orthogonal v_j. Projection z=(x-mu)V_k and reconstruction x_hat=mu+z V_k^T. Explained ratio=lambda_j/sum lambda. For (-1,-1),(0,0),(1,1), covariance [[1,1],[1,1]] has eigenvalues 2 and 0; one component along (1,1)/sqrt(2) reconstructs exactly.)LESSON"
      << '\n';
}
