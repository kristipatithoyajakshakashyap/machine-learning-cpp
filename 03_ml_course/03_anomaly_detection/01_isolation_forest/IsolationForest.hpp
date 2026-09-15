// 03_ml_course/03_anomaly_detection/01_isolation_forest/IsolationForest.hpp
// Purpose: header-only Isolation Forest (Liu, Ting & Zhou 2008). Random trees
//          split on a random feature at a random cut; anomalies are isolated in
//          few splits, so a short average path length means a high score.
// Inputs:  none (library header; included by uif_* lessons, uif_predict,
//          unsupervised_tests and unsupervised_reload).
// Outputs: none (writes nothing; save()/load() stream the fitted forest so the
//          end_to_end run can persist it under model/model_state.txt).
// Run target: not an executable; compiled into the ml_isolation_forest
//             INTERFACE library.
// sklearn equivalent: sklearn.ensemble.IsolationForest.
#pragma once
#include <algorithm>
#include <numeric>
#include <random>

#include "02_unsupervised/validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// One forest of `trees_` random isolation trees fitted on `sample_size_` rows.
// Invariants after fit(): forest_.size() == trees_, every tree has node 0 as its
// root, features_ > 0 and fitted_sample_ >= 2 (needed by the path normaliser).
class IsolationForest {
  // One tree node stored in a flat vector. left/right are indices into the same
  // vector; -1 marks a leaf. `size` is the number of training rows that reached
  // the node, which the leaf correction term c(size) needs at scoring time.
  struct Node {
    size_t feature = 0, size = 0;
    double split = 0;
    int left = -1, right = -1;
    // Archive helpers: fields in fixed order so save() and load() mirror.
    void save(std::ostream& o) const {
      archive::write(o, feature, size, split, left, right);
    }
    void load(std::istream& i) {
      archive::read(i, feature, size, split, left, right);
    }
  };
  size_t trees_, sample_size_, features_ = 0, fitted_sample_ = 0;
  unsigned seed_;
  double threshold_ = .6;  // score >= threshold_ -> flagged as anomaly
  std::vector<std::vector<Node>> forest_;
  // c(n): expected path length of an unsuccessful BST search over n points,
  // 2*H(n-1) - 2(n-1)/n with H the harmonic number. Used both to extend a
  // leaf's depth by its remaining size and to normalise the average path.
  // Returns 0 for n < 2 and 1 for n == 2. Complexity O(n).
  static double correction(size_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    double h = 0;
    for (size_t i = 1; i < n; ++i) h += 1.0 / i;
    return 2 * h - 2.0 * (n - 1) / n;
  }
  // Recursively grow one isolation tree over the rows in `ids`.
  // Parameters: X - full matrix; ids - rows in this node; depth - current depth;
  //             limit - max depth (ceil(log2(sample))); rng - shared generator;
  //             tree - flat node store appended to.
  // Returns the index of the node created for `ids`. A node becomes a leaf when
  // it holds < 2 rows, hits the depth limit, or no feature has any spread.
  // Complexity O(|ids| * p) per node.
  int grow(const Mat& X, const std::vector<size_t>& ids, size_t depth,
           size_t limit, std::mt19937& rng, std::vector<Node>& tree) {
    int index = int(tree.size());
    tree.push_back(Node{});
    tree[index].size = ids.size();
    if (ids.size() < 2 || depth >= limit) return index;
    // Only features with lo < hi can be split; find the range of each.
    std::vector<size_t> choices;
    Vec lo(features_, std::numeric_limits<double>::infinity()),
        hi(features_, -std::numeric_limits<double>::infinity());
    for (size_t i : ids)
      for (size_t j = 0; j < features_; ++j) {
        lo[j] = std::min(lo[j], X[i][j]);
        hi[j] = std::max(hi[j], X[i][j]);
      }
    for (size_t j = 0; j < features_; ++j)
      if (hi[j] > lo[j]) choices.push_back(j);
    if (choices.empty()) return index;
    // Random feature, random cut strictly inside (lo, hi] so both sides can be
    // non-empty; nextafter guards a cut that lands exactly on lo.
    size_t f = choices[rng() % choices.size()];
    double cut = std::uniform_real_distribution<double>(lo[f], hi[f])(rng);
    if (cut <= lo[f]) cut = std::nextafter(lo[f], hi[f]);
    std::vector<size_t> left, right;
    for (size_t i : ids) (X[i][f] < cut ? left : right).push_back(i);
    if (left.empty() || right.empty()) return index;
    // Children are grown first; tree may reallocate, so index via tree[index]
    // only after the recursive calls return.
    int l = grow(X, left, depth + 1, limit, rng, tree),
        r = grow(X, right, depth + 1, limit, rng, tree);
    tree[index].feature = f;
    tree[index].split = cut;
    tree[index].left = l;
    tree[index].right = r;
    return index;
  }

 public:
  // Parameters: trees - number of isolation trees; sample_size - rows drawn per
  //             tree (capped at n at fit time); seed - RNG seed for determinism.
  IsolationForest(size_t trees = 100, size_t sample_size = 256,
                  unsigned seed = 42)
      : trees_(trees), sample_size_(sample_size), seed_(seed) {}
  // Fit the forest on a dense finite matrix X (n x p).
  // Each tree sees a fresh random subsample of min(sample_size, n) rows and is
  // grown to depth ceil(log2(sample)), the height at which average points are
  // expected to isolate. Throws on empty/ragged/non-finite X or bad parameters.
  // Complexity O(trees * sample * log(sample) * p).
  void fit(const Mat& X) {
    validate_dense(X);
    if (!trees_ || sample_size_ < 2 || X.size() < 2)
      throw std::invalid_argument("invalid forest parameters");
    features_ = X[0].size();
    fitted_sample_ = std::min(sample_size_, X.size());
    std::mt19937 rng(seed_);
    std::vector<size_t> ids(X.size());
    std::iota(ids.begin(), ids.end(), 0);
    forest_.clear();
    for (size_t t = 0; t < trees_; ++t) {
      std::shuffle(ids.begin(), ids.end(), rng);
      std::vector<size_t> sub(ids.begin(), ids.begin() + fitted_sample_);
      forest_.emplace_back();
      grow(X, sub, 0, size_t(std::ceil(std::log2(double(fitted_sample_)))), rng,
           forest_.back());
    }
  }
  // Anomaly score per row: s = 2^(-E[h(x)] / c(sample)), in (0, 1).
  // E[h(x)] is the mean path length over all trees, where a leaf reached at
  // depth d with `size` rows contributes d + c(size). Scores near 1 mean the
  // row isolates quickly (anomalous); near 0.5 or below means ordinary.
  // Throws logic_error before fit() and invalid_argument on a width mismatch.
  // Complexity O(n * trees * log(sample)).
  Vec score_samples(const Mat& X) const {
    if (forest_.empty()) throw std::logic_error("fit first");
    validate_dense(X, features_);
    Vec scores;
    for (const auto& x : X) {
      double path = 0;
      for (const auto& tree : forest_) {
        int u = 0;
        size_t depth = 0;
        while (tree[u].left >= 0) {  // descend until a leaf
          const auto& n = tree[u];
          u = x[n.feature] < n.split ? n.left : n.right;
          ++depth;
        }
        path += depth + correction(tree[u].size);
      }
      scores.push_back(
          std::pow(2, -path / forest_.size() / correction(fitted_sample_)));
    }
    return scores;
  }
  // Store the calibrated decision threshold (open interval (0, 1)); the
  // end_to_end run sets it to a validation-score quantile.
  void set_threshold(double value) {
    if (!(value > 0 && value < 1))
      throw std::invalid_argument("threshold must lie in (0,1)");
    threshold_ = value;
  }
  double threshold() const { return threshold_; }
  // Binary flags: 1 when score >= threshold, else 0. The sentinel -1 means
  // "use the stored threshold". Same cost as score_samples().
  Vec predict(const Mat& X, double threshold = -1) const {
    if (threshold == -1) threshold = threshold_;
    if (!(threshold > 0 && threshold < 1))
      throw std::invalid_argument("threshold must lie in (0,1)");
    auto s = score_samples(X);
    for (double& v : s) v = v >= threshold ? 1 : 0;
    return s;
  }
  // Persist everything needed to reproduce score_samples() exactly, including
  // the whole forest, so a fresh process needs no refit.
  void save(std::ostream& o) const {
    archive::write(o, trees_, sample_size_, features_, fitted_sample_, seed_,
                   threshold_, forest_);
  }
  // Inverse of save(); rejects snapshots whose shape breaks the invariants.
  void load(std::istream& i) {
    archive::read(i, trees_, sample_size_, features_, fitted_sample_, seed_,
                  threshold_, forest_);
    if (forest_.size() != trees_ || !features_ || fitted_sample_ < 2)
      throw std::runtime_error("invalid forest snapshot");
  }
};
}  // namespace ml
