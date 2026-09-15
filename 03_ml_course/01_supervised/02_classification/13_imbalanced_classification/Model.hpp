#pragma once
// 03_ml_course/01_supervised/02_classification/13_imbalanced_classification/Model.hpp
//
// Purpose : cost-sensitive binary logistic regression. Every positive row (y == 1)
//           carries weight `positive_weight` in the log-loss, every negative row
//           weight 1, so a missed positive costs more than a false alarm.
// Inputs  : standardised feature matrix X and 0/1 labels y (the pipeline
//           standardises; this class does not).
// Outputs : none written here; persisted through save()/load() with the tag
//           "WeightedLogistic_V1".
// Run     : compiled into ml_cimb (Model.cpp); used by all cimb_* targets and by
//           tests/model_test.cpp.
//
// Cost-sensitive logistic regression: the positive class carries weight w in
// the log-loss, the negative class weight 1.  Header-only (namespace course).
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace course {
// Weighted logistic regression trained by full-batch gradient descent.
// Hyper-parameters: positive_weight (> 0), learning rate lr (> 0), epochs (> 0)
// and an L2 penalty l2 (>= 0) applied to the weights but not the bias.
// Invariant: after fit(), weights_.size() == number of feature columns.
class WeightedLogistic {
public:
  explicit WeightedLogistic(double positive_weight = 1.0, double lr = 0.1,
                            int epochs = 2000, double l2 = 1e-3)
      : positive_weight_(positive_weight), lr_(lr), epochs_(epochs), l2_(l2) {}

  // Full-batch gradient descent on the weighted binary cross-entropy.
  // Loss: -(1/W) sum_i w_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + (l2/2)|theta|^2
  // with W = sum_i w_i; the gradient is (1/W) sum_i w_i (p_i - y_i) x_i.
  // Throws std::invalid_argument on bad data or hyper-parameters (see validate).
  // Complexity: O(epochs * n * p).
  void fit(const ml::Mat &X, const ml::Vec &y) {
    validate(X, y);
    const size_t n = X.size(), p = X[0].size();
    weights_.assign(p, 0.0); // start at the origin: p = 0.5 for every row
    bias_ = 0.0;
    ml::Vec grad(p);
    for (int epoch = 0; epoch < epochs_; ++epoch) {
      std::fill(grad.begin(), grad.end(), 0.0);
      double grad_b = 0.0, total_weight = 0.0;
      for (size_t i = 0; i < n; ++i) {
        // err = w_i (p_i - y_i): positives pull `positive_weight` times harder.
        const double w = y[i] == 1.0 ? positive_weight_ : 1.0;
        const double err = w * (sigmoid(logit(X[i])) - y[i]);
        for (size_t j = 0; j < p; ++j)
          grad[j] += err * X[i][j];
        grad_b += err;
        total_weight += w;
      }
      // Normalise by the total weight (not n) so lr means the same for every w;
      // the L2 term shrinks weights only, never the bias.
      for (size_t j = 0; j < p; ++j)
        weights_[j] -= lr_ * (grad[j] / total_weight + l2_ * weights_[j]);
      bias_ -= lr_ * grad_b / total_weight;
    }
  }

  // Rows of {P(y = 0), P(y = 1)}. Throws std::runtime_error if unfitted and
  // std::invalid_argument if a row has the wrong width.
  ml::Mat predict_proba(const ml::Mat &X) const {
    if (weights_.empty())
      throw std::runtime_error("WeightedLogistic is not fitted");
    ml::Mat out;
    out.reserve(X.size());
    for (const auto &row : X) {
      if (row.size() != weights_.size())
        throw std::invalid_argument("Feature count mismatch");
      const double p1 = sigmoid(logit(row));
      out.push_back({1.0 - p1, p1});
    }
    return out;
  }

  // Hard labels at the fixed 0.5 threshold; the weight moves the boundary
  // during training, threshold moving is left to the lessons/exercises.
  ml::Vec predict(const ml::Mat &X) const {
    ml::Vec out;
    for (const auto &row : predict_proba(X))
      out.push_back(row[1] >= 0.5 ? 1.0 : 0.0);
    return out;
  }

  // Serialise tag, hyper-parameters and fitted parameters in that order.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("WeightedLogistic_V1"),
                       positive_weight_, lr_, epochs_, l2_, bias_, weights_);
  }
  // Inverse of save(); rejects a foreign tag or a non-finite bias.
  void load(std::istream &in) {
    std::string tag;
    ml::archive::read(in, tag);
    if (tag != "WeightedLogistic_V1")
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, positive_weight_, lr_, epochs_, l2_, bias_, weights_);
    if (!std::isfinite(bias_))
      throw std::runtime_error("Corrupt WeightedLogistic archive");
  }

  // Read-only accessors used by the lessons, the tests and the reports.
  double positive_weight() const { return positive_weight_; }
  double learning_rate() const { return lr_; }
  int epochs() const { return epochs_; }
  double l2() const { return l2_; }
  double bias() const { return bias_; }
  const ml::Vec &weights() const { return weights_; }

private:
  // Logits are clipped to +-30 before exp(): exp(30) is ~1e13, far from
  // overflow, while sigmoid(+-30) is already 1 or 0 to double precision.
  static constexpr double kLogitClip = 30.0;
  static double sigmoid(double z) {
    z = std::clamp(z, -kLogitClip, kLogitClip);
    return 1.0 / (1.0 + std::exp(-z));
  }
  // Linear score z = bias + weights . x for one row.
  double logit(const ml::Vec &x) const {
    double z = bias_;
    for (size_t j = 0; j < x.size(); ++j)
      z += weights_[j] * x[j];
    return z;
  }
  // Guards for fit(): rectangular finite X, labels exactly 0 or 1, and
  // positive weight/learning rate, positive epochs, non-negative L2.
  void validate(const ml::Mat &X, const ml::Vec &y) const {
    if (X.empty() || X[0].empty() || X.size() != y.size())
      throw std::invalid_argument("WeightedLogistic needs nonempty X and y");
    if (!(positive_weight_ > 0) || !(lr_ > 0) || epochs_ <= 0 || l2_ < 0)
      throw std::invalid_argument("Invalid WeightedLogistic hyperparameters");
    for (size_t i = 0; i < X.size(); ++i) {
      if (X[i].size() != X[0].size() || (y[i] != 0.0 && y[i] != 1.0))
        throw std::invalid_argument("Rows must be rectangular with 0/1 labels");
      for (double v : X[i])
        if (!std::isfinite(v))
          throw std::invalid_argument("Nonfinite feature");
    }
  }
  double positive_weight_, lr_;
  int epochs_;
  double l2_, bias_ = 0.0;
  ml::Vec weights_;
};
} // namespace course
