// 03_ml_course/02_unsupervised/02_agglomerative/Agglomerative.cpp
// Purpose: implementation of ml::AgglomerativeClustering (Agglomerative.hpp):
//          builds the full merge tree with single/complete/average/Ward
//          linkage (Lance-Williams updates), cuts it at k clusters and saves
//          or reloads the hierarchy as text.
// Inputs:  a dense finite matrix passed to fit() by the lessons.
// Outputs: none of its own (prints nothing, writes nothing).
// Run target: compiled into the static library ml_aggl, linked by uag_*,
//             uce_*, unsupervised_tests and unsupervised_reload.
#include "Agglomerative.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "../validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// Store hyper-parameters only; the tree is built in fit().
AgglomerativeClustering::AgglomerativeClustering(size_t k, Linkage l)
    : n_clusters_(k), linkage_(l) {}
// Build the complete hierarchy (n-1 merges) and set labels_ = cut(n_clusters_).
// Parameters: X [n][p], finite. Throws on k outside [1, n] or a bad linkage.
// Invariants: node ids 0..n-1 are leaves, n+t is the cluster created by merge t;
//             linkage_matrix_[t] = {child_a, child_b, height, size}, like
//             scipy's linkage output. Ties are broken by lowest active index.
// Complexity O(n^3) time and O(n^2) memory (naive pairwise scan per merge),
// fine for the 344-row penguin data.
void AgglomerativeClustering::fit(const Mat& X) {
  validate_dense(X);
  const size_t n = X.size();
  n_samples_ = n;
  if (!n_clusters_ || n_clusters_ > n)
    throw std::invalid_argument("invalid cluster count");
  if (static_cast<int>(linkage_) < 0 || static_cast<int>(linkage_) > 3)
    throw std::invalid_argument("invalid linkage");
  history_.clear();
  linkage_matrix_.clear();
  // D is indexed by node id (leaves and merged nodes), hence 2n-1 wide.
  Mat D(2 * n - 1, Vec(2 * n - 1, 0));
  std::vector<size_t> count(2 * n - 1, 1), active(n), representative(2 * n - 1);
  std::iota(active.begin(), active.end(), 0);
  std::iota(representative.begin(), representative.end(), 0);
  // Leaf-to-leaf Euclidean distances (sqrt so heights are in feature units).
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
      D[i][j] = D[j][i] = std::sqrt(distance2(X[i], X[j]));
  while (active.size() > 1) {
    // Find the closest pair of active clusters (strict < keeps the first, i.e.
    // lowest-index, pair on ties for determinism).
    double best = std::numeric_limits<double>::infinity();
    size_t ai = 0, bi = 1;
    for (size_t i = 0; i < active.size(); ++i)
      for (size_t j = i + 1; j < active.size(); ++j)
        if (D[active[i]][active[j]] < best) {
          best = D[active[i]][active[j]];
          ai = i;
          bi = j;
        }
    size_t a = active[ai], b = active[bi], u = n + linkage_matrix_.size();
    count[u] = count[a] + count[b];
    linkage_matrix_.push_back({double(a), double(b), best, double(count[u])});
    history_.push_back({{representative[a], representative[b]}, best});
    representative[u] = std::min(representative[a], representative[b]);
    // Lance-Williams update: distance from the new cluster u to every other
    // active cluster v, derived from D[a][v], D[b][v] and the merge height.
    for (size_t v : active)
      if (v != a && v != b) {
        double da = D[a][v], db = D[b][v], d = 0;
        if (linkage_ == Linkage::Single)
          d = std::min(da, db);  // nearest members
        else if (linkage_ == Linkage::Complete)
          d = std::max(da, db);  // farthest members
        else if (linkage_ == Linkage::Average)
          d = (count[a] * da + count[b] * db) / double(count[u]);  // size-weighted
        else {
          // Ward: increase in within-cluster variance if u and v were merged;
          // max(0, .) guards tiny negative values from rounding.
          double total = double(count[u] + count[v]);
          d = std::sqrt(std::max(
              0.0, ((count[v] + count[a]) * da * da +
                    (count[v] + count[b]) * db * db - count[v] * best * best) /
                       total));
        }
        D[u][v] = D[v][u] = d;
      }
    // Retire a and b (erase the larger index first so ai stays valid).
    active.erase(active.begin() + bi);
    active.erase(active.begin() + ai);
    active.push_back(u);
  }
  labels_ = cut(n_clusters_);
}
// Cut the stored tree into k flat clusters by replaying the first n-k merges.
// Returns labels 0..k-1 assigned in order of each cluster's smallest row index,
// so label ids are stable and independent of merge order.
// Throws before fit() or when k is outside [1, n]. Complexity O(n^2) worst case.
Vec AgglomerativeClustering::cut(size_t k) const {
  if (!n_samples_ || !k || k > n_samples_)
    throw std::invalid_argument("invalid cut or unfitted hierarchy");
  std::vector<std::vector<size_t>> members(2 * n_samples_ - 1);
  std::vector<bool> active(2 * n_samples_ - 1, false);
  for (size_t i = 0; i < n_samples_; ++i) {
    members[i] = {i};
    active[i] = true;
  }
  // Replaying n-k merges leaves exactly k active nodes.
  for (size_t t = 0; t < n_samples_ - k; ++t) {
    size_t a = size_t(linkage_matrix_[t][0]), b = size_t(linkage_matrix_[t][1]),
           u = n_samples_ + t;
    members[u] = members[a];
    members[u].insert(members[u].end(), members[b].begin(), members[b].end());
    active[a] = active[b] = false;
    active[u] = true;
  }
  // Sort groups lexicographically (i.e. by smallest member) to number them.
  std::vector<std::vector<size_t>> groups;
  for (size_t i = 0; i < active.size(); ++i)
    if (active[i]) {
      std::sort(members[i].begin(), members[i].end());
      groups.push_back(members[i]);
    }
  std::sort(groups.begin(), groups.end());
  Vec y(n_samples_);
  for (size_t c = 0; c < groups.size(); ++c)
    for (size_t i : groups[c]) y[i] = double(c);
  return y;
}
// Persist k, linkage code, n, labels and the linkage matrix. history_ is not
// stored; load() rebuilds it from the linkage matrix.
void AgglomerativeClustering::save(std::ostream& o) const {
  archive::write(o, n_clusters_, static_cast<int>(linkage_), n_samples_,
                 labels_, linkage_matrix_);
}
// Reload and validate (a hierarchy over n rows must have exactly n-1 merges),
// then replay the merges to recover the representative-based merge history.
void AgglomerativeClustering::load(std::istream& i) {
  int l;
  archive::read(i, n_clusters_, l, n_samples_, labels_, linkage_matrix_);
  linkage_ = static_cast<Linkage>(l);
  if (!n_samples_ || linkage_matrix_.size() != n_samples_ - 1)
    throw std::runtime_error("invalid hierarchy snapshot");
  history_.clear();
  std::vector<size_t> reps(2 * n_samples_ - 1);
  std::iota(reps.begin(), reps.end(), 0);
  for (size_t t = 0; t < linkage_matrix_.size(); ++t) {
    auto& r = linkage_matrix_[t];
    size_t a = size_t(r[0]), b = size_t(r[1]);
    history_.push_back({{reps[a], reps[b]}, r[2]});
    reps[n_samples_ + t] = std::min(reps[a], reps[b]);
  }
}
}  // namespace ml
