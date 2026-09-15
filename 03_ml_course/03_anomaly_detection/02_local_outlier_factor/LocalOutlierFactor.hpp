#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "02_unsupervised/validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// Local Outlier Factor (Breunig et al. 2000) in novelty mode: the training
// matrix is kept, and any query row is scored against its k nearest training
// rows.  Scores near 1 mean "as dense as the neighbourhood"; larger values
// mean the row sits in a sparser region than its neighbours.
class LocalOutlierFactor {
  struct Neighbour {
    double distance = 0;
    size_t id = 0;
  };
  static constexpr double kEpsilon = 1e-12;
  static constexpr size_t kNone = size_t(-1);
  size_t k_requested_, k_ = 0;
  double threshold_ = 1.5;
  Mat train_;
  Vec kdist_, lrd_, lof_;
  static bool closer(const Neighbour& a, const Neighbour& b) {
    return a.distance < b.distance ||
           (a.distance == b.distance && a.id < b.id);
  }
  // The k nearest training rows to q.  `self` is skipped so training rows do
  // not count themselves during fit; kNone keeps every row.
  std::vector<Neighbour> neighbours(const Vec& q, size_t self) const {
    std::vector<Neighbour> all;
    all.reserve(train_.size());
    for (size_t i = 0; i < train_.size(); ++i)
      if (i != self) all.push_back({std::sqrt(distance2(q, train_[i])), i});
    std::partial_sort(all.begin(), all.begin() + k_, all.end(), closer);
    all.resize(k_);
    return all;
  }
  // lrd(q) = k / sum_o reach_k(q, o), reach_k(q, o) = max(kdist(o), d(q, o)).
  double density(const std::vector<Neighbour>& nb) const {
    double reach = 0;
    for (const auto& n : nb) reach += std::max(kdist_[n.id], n.distance);
    return double(k_) / std::max(reach, kEpsilon);
  }
  double factor(const std::vector<Neighbour>& nb, double own) const {
    double sum = 0;
    for (const auto& n : nb) sum += lrd_[n.id];
    return sum / k_ / own;
  }

 public:
  explicit LocalOutlierFactor(size_t k = 20) : k_requested_(k) {}
  // X must already be preprocessed (imputed, scaled) by the caller.
  void fit(const Mat& X) {
    validate_dense(X);
    if (X.size() < 2)
      throw std::invalid_argument("at least two training rows required");
    if (!k_requested_) throw std::invalid_argument("k must be positive");
    train_ = X;
    k_ = std::min(k_requested_, X.size() - 1);
    std::vector<std::vector<Neighbour>> nb;
    kdist_.assign(X.size(), 0);
    for (size_t i = 0; i < X.size(); ++i) {
      nb.push_back(neighbours(X[i], i));
      kdist_[i] = nb.back().back().distance;
    }
    lrd_.assign(X.size(), 0);
    for (size_t i = 0; i < X.size(); ++i) lrd_[i] = density(nb[i]);
    lof_.assign(X.size(), 0);
    for (size_t i = 0; i < X.size(); ++i) lof_[i] = factor(nb[i], lrd_[i]);
  }
  size_t k() const { return k_; }
  // Outlier-mode scores of the training rows themselves (self excluded).
  const Vec& fitted_scores() const { return lof_; }
  // Novelty-mode scores: every query is compared with the stored training
  // rows only.  Higher means more anomalous.
  Vec score_samples(const Mat& Q) const {
    if (train_.empty()) throw std::logic_error("fit first");
    validate_dense(Q, train_[0].size());
    Vec scores;
    scores.reserve(Q.size());
    for (const auto& q : Q) {
      auto nb = neighbours(q, kNone);
      scores.push_back(factor(nb, density(nb)));
    }
    return scores;
  }
  void set_threshold(double value) {
    if (!(value > 0) || !std::isfinite(value))
      throw std::invalid_argument("threshold must be positive and finite");
    threshold_ = value;
  }
  double threshold() const { return threshold_; }
  // 1 = anomaly (score above threshold), 0 = ordinary.
  Vec predict(const Mat& Q, double threshold = -1) const {
    if (threshold == -1) threshold = threshold_;
    if (!(threshold > 0) || !std::isfinite(threshold))
      throw std::invalid_argument("threshold must be positive and finite");
    auto s = score_samples(Q);
    for (double& v : s) v = v > threshold ? 1 : 0;
    return s;
  }
  void save(std::ostream& o) const {
    archive::write(o, std::string("LOF_V1"), k_, threshold_, train_, kdist_,
                   lrd_);
  }
  void load(std::istream& i) {
    std::string tag;
    archive::read(i, tag, k_, threshold_, train_, kdist_, lrd_);
    if (tag != "LOF_V1" || train_.size() < 2 || !k_ || k_ >= train_.size() ||
        kdist_.size() != train_.size() || lrd_.size() != train_.size())
      throw std::runtime_error("invalid LOF snapshot");
    k_requested_ = k_;
    lof_.clear();
  }
};
}  // namespace ml
