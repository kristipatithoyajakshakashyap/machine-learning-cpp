// 03_ml_course/05_text_classification/04_linear_text_classifier/Model.hpp
//
// Purpose : L2-regularised binary logistic regression on sparse text
//           features, trained by deterministic full-batch gradient descent
//           with a step size derived from a Lipschitz bound (no tuning, no
//           randomness).
// Inputs  : Sparse rows from Vectorizer::transform (TF-IDF recommended).
// Outputs : none directly; Workflow.hpp archives the model as
//           <run>/model/model.txt via save()/load().
// Used by : text_linear_end_to_end / text_linear_predict through
//           Workflow.hpp, and every text module test (header-only).
#pragma once
#include "05_text_classification/02_count_and_tfidf/Model.hpp"
namespace text_course {
// Logistic regression for labels {0 = ham, 1 = spam}.
// Model contract used by Workflow.hpp: Model(parameter), fit(X, y, p),
// probabilities(X) -> p(spam) per row, save/load.
// Invariant after fit(): weights.size() == p; bias is unregularised.
class LinearTextClassifier {
public:
  double regularization, bias = 0; // L2 strength lambda (>= 0) and intercept
  size_t epochs;                   // number of full-batch gradient steps
  std::vector<double> weights;     // one weight per vocabulary feature
  // lambda     : L2 penalty (the workflow's tuned parameter).
  // iterations : gradient-descent epochs (each visits every row once).
  explicit LinearTextClassifier(double lambda = 0.001, size_t iterations = 500)
      : regularization(lambda), epochs(iterations) {}
  // Minimise mean log-loss + lambda/2 * ||w||^2 by gradient descent.
  // Parameters: X - sparse rows; y - labels 0/1; p - feature count.
  // The fixed step 1 / (0.25 * (1 + max ||x||^2) + lambda) is the inverse of
  // an upper bound on the loss curvature (sigmoid' <= 1/4, row norm plus
  // the bias column), which guarantees monotone convergence without a
  // learning-rate search. Throws std::invalid_argument on empty data, size
  // mismatch, p == 0, negative lambda, zero epochs or non-binary labels.
  // Complexity: O(epochs * (non-zeros + p)).
  void fit(const Sparse &X, const std::vector<double> &y, size_t p) {
    if (X.empty() || X.size() != y.size() || !p || regularization < 0 ||
        !epochs)
      throw std::invalid_argument("Invalid linear text data");
    weights.assign(p, 0);
    bias = 0;
    for (double label : y)
      if (label != 0 && label != 1)
        throw std::invalid_argument("Binary labels required");
    // Largest squared row norm bounds the Hessian of the data term.
    double max_norm = 0;
    for (const auto &row : X) {
      double norm = 0;
      for (const auto &term : row)
        norm += term.second * term.second;
      max_norm = std::max(max_norm, norm);
    }
    const double step = 1 / (0.25 * (1 + max_norm) + regularization);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      // Accumulate the mean gradient over all rows, then take one step.
      std::vector<double> gradient(p);
      double gb = 0;
      for (size_t i = 0; i < X.size(); ++i) {
        double score = bias;
        for (const auto &v : X[i])
          score += weights.at(v.first) * v.second;
        // err = sigmoid(score) - y is the derivative of log-loss w.r.t. score.
        double err =
            1 / (1 + std::exp(-std::clamp(score, -700.0, 700.0))) - y[i];
        gb += err / X.size();
        for (const auto &v : X[i])
          gradient[v.first] += err * v.second / X.size();
      }
      // Weight decay applies to weights only; the bias is not penalised.
      for (size_t j = 0; j < p; ++j)
        weights[j] -= step * (gradient[j] + regularization * weights[j]);
      bias -= step * gb;
    }
  }
  // P(spam | doc) = sigmoid(bias + w . x) per row; the logit is clamped to
  // [-700, 700] so exp() cannot overflow. Empty rows return sigmoid(bias).
  // Throws std::runtime_error if not fitted. Complexity: O(non-zeros).
  std::vector<double> probabilities(const Sparse &X) const {
    if (weights.empty())
      throw std::runtime_error("Linear text model not fitted");
    std::vector<double> out;
    for (const auto &row : X) {
      double v = bias;
      for (const auto &term : row)
        v += weights.at(term.first) * term.second;
      out.push_back(1 / (1 + std::exp(-std::clamp(v, -700.0, 700.0))));
    }
    return out;
  }
  // Versioned text archive: type tag, lambda, bias, epochs, weights.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("LinearTextClassifier_V1"));
    ml::archive::write(out, regularization, bias, epochs, weights);
  }
  // Restore from save(); rejects any other model type tag.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("LinearTextClassifier_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, regularization, bias, epochs, weights);
  }
};
} // namespace text_course
