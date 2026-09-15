#ifndef MLC_MODULE_AGGLOMERATIVE_HPP
#define MLC_MODULE_AGGLOMERATIVE_HPP

// ===========================================================================
// 02_unsupervised/02_agglomerative/Agglomerative.hpp (+ Agglomerative.cpp)
// ---------------------------------------------------------------------------
// SELF-CONTAINED module implementation.
//
// Agglomerative (bottom-up hierarchical) clustering:
//   1. Start with every row as its own cluster.
//   2. At each step merge (agglomerate) the two clusters whose DISTANCE is
//      smallest, where cluster distance depends on the linkage:
//        SINGLE   : min distance between a member of A and a member of B;
//        COMPLETE : max distance (farthest pair);
//        AVERAGE  : mean of all pairwise A-B distances.
//   3. Build the full tree, then cut at n_clusters.
// The merge history (which pair merged, at what distance) is kept, which is
// exactly the data a dendrogram is drawn from.
//
// Deterministic: ties are broken by the lowest cluster ids, so results are
// reproducible; other libraries may resolve equal-distance ties differently.
//
// sklearn equivalent: sklearn.cluster.AgglomerativeClustering(
//   n_clusters, linkage='single'|'complete'|'average'|'ward').
// ===========================================================================

#include <cstddef>
#include <iosfwd>
#include <utility>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {

enum class Linkage { Single, Complete, Average, Ward };

class AgglomerativeClustering {
 public:
  // n_clusters: number of clusters to stop at.
  // linkage:    how to measure the distance between two clusters.
  AgglomerativeClustering(size_t n_clusters = 2,
                          Linkage linkage = Linkage::Average);

  void fit(const Mat& X);
  Vec cut(size_t n_clusters) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  const Mat& linkage_matrix() const { return linkage_matrix_; }

  // Cluster id (0 .. n_clusters-1, ordered by cluster's smallest row index)
  // for every training row (sklearn's labels_).
  const Vec& labels() const { return labels_; }

  // (RepA, RepB) merged at each step, with the merge distance; RepA/B are
  // the smallest row index of each merged cluster (legacy lesson interface).
  // linkage_matrix() exposes actual child-node IDs for the complete tree.
  const std::vector<std::pair<std::pair<size_t, size_t>, double>>&
  merge_history() const {
    return history_;
  }

 private:
  Mat linkage_matrix_;  // child A, child B, height, member count; leaves 0..n-1
  size_t n_samples_ = 0;
  size_t n_clusters_;
  Linkage linkage_;
  Vec labels_;
  std::vector<std::pair<std::pair<size_t, size_t>, double>> history_;
};

}  // namespace ml

#endif  // MLC_MODULE_AGGLOMERATIVE_HPP