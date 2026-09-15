// 03_ml_course/05_text_classification/03_multinomial_naive_bayes/Model.hpp
//
// Purpose : binary Multinomial Naive Bayes over sparse term counts.
//           P(class | doc) is proportional to P(class) * prod_w P(w|class)^n_w
//           with Laplace (additive) smoothing alpha on the word counts.
// Inputs  : Sparse rows from Vectorizer::transform (raw counts are the
//           textbook input; TF-IDF weights also work as fractional counts).
// Outputs : none directly; Workflow.hpp archives the model as
//           <run>/model/model.txt via save()/load().
// Used by : text_nb_end_to_end / text_nb_predict through Workflow.hpp, and
//           the other text modules' tests (header-only, no own target).
#pragma once
#include "05_text_classification/02_count_and_tfidf/Model.hpp"
namespace text_course {
// Multinomial NB for labels {0 = ham, 1 = spam}.
// Model contract used by Workflow.hpp: Model(parameter), fit(X, y, p),
// probabilities(X) -> p(spam) per row, save/load.
// Invariants after fit(): logprior.size() == 2, logprob.size() == 2,
// logprob[c].size() == p, and exp(logprob[c]) sums to 1 for each class.
class MultinomialNB {
public:
  double alpha;                                // smoothing pseudo-count > 0
  std::vector<double> logprior;                // log P(class), 2 entries
  std::vector<std::vector<double>> logprob;    // log P(word | class), 2 x p
  // a : additive smoothing alpha; the workflow's tuned parameter.
  explicit MultinomialNB(double a = 1) : alpha(a) {}
  // Estimate class priors and smoothed word likelihoods.
  // Parameters: X - sparse rows; y - labels 0/1 (same length as X);
  //             p - feature count (vocabulary size), bounds every index.
  // logprob[c][w] = log((alpha + count of w in class c) / (alpha*p + total_c)).
  // Throws std::invalid_argument on empty data, size mismatch, alpha <= 0,
  // non-binary labels, out-of-range indices, negative counts or a missing
  // class. Complexity: O(non-zeros + 2p).
  void fit(const Sparse &X, const std::vector<double> &y, size_t p) {
    if (X.empty() || X.size() != y.size() || !p || alpha <= 0)
      throw std::invalid_argument("Invalid NB data");
    // Start every count at alpha; logprior temporarily holds class counts.
    logprob.assign(2, std::vector<double>(p, alpha));
    logprior.assign(2, 0);
    for (size_t i = 0; i < X.size(); ++i) {
      if (y[i] != 0 && y[i] != 1)
        throw std::invalid_argument("Binary labels required");
      size_t c = static_cast<size_t>(y[i]);
      ++logprior[c];
      for (const auto &v : X[i]) {
        if (v.first >= p || v.second < 0)
          throw std::invalid_argument("Invalid count");
        logprob[c][v.first] += v.second;
      }
    }
    // Normalise counts into log-probabilities per class.
    for (size_t c = 0; c < 2; ++c) {
      if (logprior[c] == 0)
        throw std::invalid_argument("Both classes required");
      logprior[c] = std::log(logprior[c] / X.size());
      double total = 0;
      for (double v : logprob[c])
        total += v;
      for (double &v : logprob[c])
        v = std::log(v / total);
    }
  }
  // Posterior P(spam | doc) for each row.
  // Log-scores a (ham) and b (spam) are compared through the sigmoid of
  // their difference: P(spam) = 1 / (1 + exp(a - b)). The difference is
  // clamped to [-700, 700] so exp() never overflows to inf.
  // Features unknown at fit time never appear in X (Vectorizer drops them);
  // an empty row therefore returns the prior ratio.
  // Throws std::runtime_error if not fitted. Complexity: O(non-zeros).
  std::vector<double> probabilities(const Sparse &X) const {
    if (logprob.empty())
      throw std::runtime_error("NB not fitted");
    std::vector<double> out;
    for (const auto &row : X) {
      double a = logprior[0], b = logprior[1];
      for (const auto &v : row) {
        a += v.second * logprob[0].at(v.first);
        b += v.second * logprob[1].at(v.first);
      }
      double d = std::clamp(a - b, -700.0, 700.0);
      out.push_back(1 / (1 + std::exp(d)));
    }
    return out;
  }
  // Versioned text archive: type tag, alpha, logprior, logprob.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("MultinomialNB_V1"));
    ml::archive::write(out, alpha, logprior, logprob);
  }
  // Restore from save(); rejects any other model type tag.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("MultinomialNB_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, alpha, logprior, logprob);
  }
};
} // namespace text_course
