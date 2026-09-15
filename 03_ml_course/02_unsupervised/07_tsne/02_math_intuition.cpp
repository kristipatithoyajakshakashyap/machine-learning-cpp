// 03_ml_course/02_unsupervised/07_tsne/02_math_intuition.cpp
// Purpose: print the four t-SNE formulas (high-dim affinities P, low-dim
//          Student-t affinities Q, the KL objective and its gradient) and
//          check the worked perplexity example numerically.
// Inputs:  none (constants in code).
// Outputs: prints only.
// Run target: utsne_math_intuition.
#include <cmath>
#include <iostream>
int main() {
  // The lesson text: formulas plus a three-point hand example on a line.
  std::cout
      << R"LESSON(High-dimensional affinities: p_{j|i} = exp(-||x_i-x_j||^2 beta_i) / sum_{k != i} exp(-||x_i-x_k||^2 beta_i), with beta_i chosen so that the entropy H(P_i) = log(perplexity). Symmetrise p_ij = (p_{j|i} + p_{i|j}) / (2n).
Low-dimensional affinities: q_ij = (1 + ||y_i-y_j||^2)^-1 / sum_{k != l} (1 + ||y_k-y_l||^2)^-1.
Objective: KL(P||Q) = sum_{i != j} p_ij log(p_ij / q_ij).
Gradient: dC/dy_i = 4 sum_j (p_ij - q_ij)(1 + ||y_i-y_j||^2)^-1 (y_i - y_j).
Hand example: points 0, 1, 3 on a line, beta = 1 for point 0: unnormalised weights exp(-1) = 0.3679 and exp(-9) = 0.000123, so p_{1|0} = 0.99967 and p_{3|0} = 0.00033. Entropy = 0.0027 nats, perplexity = exp(0.0027) = 1.0027: with beta = 1 point 0 effectively sees a single neighbour. Shrinking beta towards 0 makes both weights equal, entropy log 2, perplexity 2. The binary search over beta lands between those extremes at the requested perplexity.)LESSON"
      << '\n';
  // Numeric check of the hand example: squared distances from point 0 are
  // 1 and 9, beta = 1, so the weights are exp(-1) and exp(-9).
  const double w1 = std::exp(-1.0), w3 = std::exp(-9.0), s = w1 + w3;
  const double p1 = w1 / s, p3 = w3 / s;
  // Shannon entropy in nats; perplexity = exp(entropy) = "effective
  // neighbour count" of point 0 under this bandwidth.
  const double h = -(p1 * std::log(p1) + p3 * std::log(p3));
  std::cout << "check: p_{1|0} = " << p1 << ", p_{3|0} = " << p3
            << ", entropy = " << h << ", perplexity = " << std::exp(h)
            << '\n';
}
