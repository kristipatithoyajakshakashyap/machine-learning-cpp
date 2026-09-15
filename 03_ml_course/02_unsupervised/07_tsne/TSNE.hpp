#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// t-distributed Stochastic Neighbour Embedding (van der Maaten & Hinton
// 2008).  High-dimensional affinities P are Gaussian with a per-point
// bandwidth found by binary search so every row has the requested perplexity;
// low-dimensional affinities Q use a Student-t kernel with one degree of
// freedom; gradient descent with momentum minimises KL(P || Q).
//
// t-SNE has no out-of-sample map: fit() embeds exactly the rows it is given.
// The saved state stores the parameters, the seed and the embedding, so a
// reloaded model can either return the stored coordinates or refit on the
// same rows and reproduce them bit for bit.
class TSNE {
  static constexpr double kEpsilon = 1e-12;
  static constexpr size_t kExaggerationIterations = 100;
  static constexpr double kEarlyExaggeration = 12;
  static constexpr size_t kMomentumSwitch = 250;
  static constexpr double kInitialMomentum = .5, kFinalMomentum = .8;
  static constexpr size_t kBinarySearchSteps = 50;
  static constexpr double kEntropyTolerance = 1e-5;
  size_t dims_, iterations_;
  double perplexity_, learning_rate_;
  unsigned seed_;
  Mat Y_;
  double kl_ = 0;
  Vec kl_history_;

  // Conditional affinities p_{j|i} with bandwidth beta_i chosen by binary
  // search so that the Shannon entropy of row i equals log(perplexity).
  static Vec conditional_row(const Vec& d2, size_t i, double target_entropy) {
    const size_t n = d2.size();
    Vec row(n, 0);
    double beta = 1, lo = 0, hi = std::numeric_limits<double>::infinity();
    for (size_t step = 0; step < kBinarySearchSteps; ++step) {
      double sum = 0;
      for (size_t j = 0; j < n; ++j) {
        row[j] = j == i ? 0 : std::exp(-beta * d2[j]);
        sum += row[j];
      }
      sum = std::max(sum, kEpsilon);
      double entropy = 0;
      for (size_t j = 0; j < n; ++j) {
        row[j] /= sum;
        if (row[j] > 0) entropy -= row[j] * std::log(row[j]);
      }
      const double diff = entropy - target_entropy;
      if (std::abs(diff) < kEntropyTolerance) break;
      if (diff > 0) {  // distribution too flat: sharpen the Gaussian
        lo = beta;
        beta = std::isinf(hi) ? beta * 2 : (beta + hi) / 2;
      } else {
        hi = beta;
        beta = (beta + lo) / 2;
      }
    }
    return row;
  }

 public:
  explicit TSNE(size_t dims = 2, double perplexity = 30,
                size_t iterations = 500, double learning_rate = 200,
                unsigned seed = 42)
      : dims_(dims),
        iterations_(iterations),
        perplexity_(perplexity),
        learning_rate_(learning_rate),
        seed_(seed) {}
  // Symmetric joint affinities P (n x n) for preprocessed rows X.
  Mat joint_affinities(const Mat& X) const {
    validate_dense(X);
    const size_t n = X.size();
    if (!(perplexity_ > 0) || perplexity_ >= double(n))
      throw std::invalid_argument("perplexity must lie in (0, n)");
    Mat P(n, Vec(n, 0));
    const double target = std::log(perplexity_);
    for (size_t i = 0; i < n; ++i) {
      Vec d2(n, 0);
      for (size_t j = 0; j < n; ++j) d2[j] = distance2(X[i], X[j]);
      P[i] = conditional_row(d2, i, target);
    }
    for (size_t i = 0; i < n; ++i)
      for (size_t j = i + 1; j < n; ++j) {
        const double v = std::max((P[i][j] + P[j][i]) / (2.0 * n), kEpsilon);
        P[i][j] = P[j][i] = v;
      }
    return P;
  }
  // X must already be preprocessed (imputed, scaled) by the caller.
  void fit(const Mat& X) {
    if (!dims_ || !iterations_ || !(learning_rate_ > 0))
      throw std::invalid_argument("invalid t-SNE hyper-parameters");
    const Mat P = joint_affinities(X);
    const size_t n = X.size();
    std::mt19937 gen(seed_);
    std::normal_distribution<double> init(0, 1e-4);
    Y_.assign(n, Vec(dims_, 0));
    for (auto& row : Y_)
      for (double& v : row) v = init(gen);
    Mat velocity(n, Vec(dims_, 0)), gains(n, Vec(dims_, 1)),
        grad(n, Vec(dims_, 0));
    Mat W(n, Vec(n, 0));  // Student-t numerators (1 + ||y_i - y_j||^2)^-1
    kl_history_.assign(iterations_, 0);
    for (size_t t = 0; t < iterations_; ++t) {
      const double exaggeration =
          t < kExaggerationIterations ? kEarlyExaggeration : 1;
      const double momentum =
          t < kMomentumSwitch ? kInitialMomentum : kFinalMomentum;
      double sumW = 0;
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
          W[i][j] = W[j][i] = 1.0 / (1.0 + distance2(Y_[i], Y_[j]));
          sumW += 2 * W[i][j];
        }
      sumW = std::max(sumW, kEpsilon);
      double kl = 0;
      for (size_t i = 0; i < n; ++i) {
        std::fill(grad[i].begin(), grad[i].end(), 0);
        for (size_t j = 0; j < n; ++j) {
          if (j == i) continue;
          const double q = std::max(W[i][j] / sumW, kEpsilon);
          kl += P[i][j] * std::log(P[i][j] / q);
          const double mult = 4 * (exaggeration * P[i][j] - q) * W[i][j];
          for (size_t d = 0; d < dims_; ++d)
            grad[i][d] += mult * (Y_[i][d] - Y_[j][d]);
        }
      }
      kl_history_[t] = kl;
      Vec centre(dims_, 0);
      for (size_t i = 0; i < n; ++i)
        for (size_t d = 0; d < dims_; ++d) {
          const bool same_sign = (grad[i][d] > 0) == (velocity[i][d] > 0);
          gains[i][d] =
              std::max(same_sign ? gains[i][d] * .8 : gains[i][d] + .2, .01);
          velocity[i][d] = momentum * velocity[i][d] -
                           learning_rate_ * gains[i][d] * grad[i][d];
          Y_[i][d] += velocity[i][d];
          centre[d] += Y_[i][d] / double(n);
        }
      for (auto& row : Y_)
        for (size_t d = 0; d < dims_; ++d) row[d] -= centre[d];
    }
    kl_ = kl_history_.back();
  }
  size_t dims() const { return dims_; }
  double perplexity() const { return perplexity_; }
  size_t iterations() const { return iterations_; }
  double learning_rate() const { return learning_rate_; }
  unsigned seed() const { return seed_; }
  // Coordinates of the fitted rows, in the order they were given.
  Mat embedding() const {
    if (Y_.empty()) throw std::logic_error("fit first");
    return Y_;
  }
  // KL(P || Q) after the last iteration, computed without exaggeration.
  double kl_divergence() const {
    if (Y_.empty()) throw std::logic_error("fit first");
    return kl_;
  }
  const Vec& kl_history() const { return kl_history_; }
  void save(std::ostream& o) const {
    archive::write(o, std::string("TSNE_V1"), dims_, perplexity_, iterations_,
                   learning_rate_, seed_, Y_, kl_);
  }
  void load(std::istream& i) {
    std::string tag;
    archive::read(i, tag, dims_, perplexity_, iterations_, learning_rate_,
                  seed_, Y_, kl_);
    if (tag != "TSNE_V1" || !dims_ || !iterations_ || !(perplexity_ > 0) ||
        !(learning_rate_ > 0) || Y_.empty())
      throw std::runtime_error("invalid t-SNE snapshot");
    for (const auto& row : Y_)
      if (row.size() != dims_)
        throw std::runtime_error("invalid t-SNE snapshot");
    kl_history_.clear();
  }
};

// Trustworthiness (Venna & Kaski 2001): 1 minus a penalty for points that are
// among the k nearest neighbours in the embedding but not in the original
// space, weighted by how far down the original ranking they sit.  1 means
// perfect neighbourhood preservation.  Unlike the KL objective it is
// comparable across perplexities, so the project selects with it.
inline double trustworthiness(const Mat& X, const Mat& Y, size_t k = 10) {
  validate_dense(X);
  validate_dense(Y);
  const size_t n = X.size();
  if (Y.size() != n) throw std::invalid_argument("row count mismatch");
  if (!k || 2 * n <= 3 * k + 1)
    throw std::invalid_argument("k too large for trustworthiness");
  auto ranking = [&](const Mat& M, size_t i) {
    std::vector<size_t> order;
    for (size_t j = 0; j < n; ++j)
      if (j != i) order.push_back(j);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return distance2(M[i], M[a]) < distance2(M[i], M[b]);
    });
    return order;
  };
  double penalty = 0;
  for (size_t i = 0; i < n; ++i) {
    const auto original = ranking(X, i), embedded = ranking(Y, i);
    std::vector<size_t> rank(n, 0);
    for (size_t r = 0; r < original.size(); ++r) rank[original[r]] = r + 1;
    for (size_t r = 0; r < k; ++r)
      if (rank[embedded[r]] > k) penalty += double(rank[embedded[r]] - k);
  }
  return 1 - 2.0 / (double(n) * double(k) * (2.0 * n - 3.0 * k - 1)) * penalty;
}
}  // namespace ml
