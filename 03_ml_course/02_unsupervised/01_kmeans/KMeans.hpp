#ifndef MLC_MODULE_KMEANS_HPP
#define MLC_MODULE_KMEANS_HPP

// ===========================================================================
// 02_unsupervised/01_kmeans/KMeans.hpp  (+ the matching KMeans.cpp)
// ---------------------------------------------------------------------------
// SELF-CONTAINED module implementation: everything this module's lessons need
// lives in this one file pair, so the module folder stands alone.
//
// k-means clustering:
//   1. INIT: pick k centroids (deterministic k-means++ seeding, like
//      sklearn's random_state + init='k-means++').
//   2. ASSIGN: put every row into the cluster of its nearest centroid.
//   3. UPDATE: move each centroid to the arithmetic mean of its rows
//      (this always lowers the objective, the sum of squared distances).
//   4. REPEAT 2-3 until no label changes or max_iter is reached.
// The objective  J = sum over rows of ||x - c_label(x)||^2  (inertia) is
// non-convex, so we restart n_init times with different seeds and keep the
// lowest-inertia run, using the same objective as conventional implementations.
//
// sklearn equivalent: sklearn.cluster.KMeans(init='k-means++',
//   n_init=..., random_state=<seed>, n_clusters=k).
// ===========================================================================

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {

class KMeans {
 public:
  // n_clusters: number of clusters k.
  // n_init:     restarts; the lowest-inertia run is kept.
  //             We default to 10; results can vary across implementations.
  // max_iter:   Lloyd-iteration budget per run.
  // seed:       rng seed; k-means++ seeding is deterministic per seed.
  KMeans(size_t n_clusters = 8, size_t n_init = 10, size_t max_iter = 300,
         std::uint32_t seed = 42);

  void fit(const Mat& X);
  void save(std::ostream&) const;
  void load(std::istream&);
  // Nearest-centroid mapping of new rows (0..n_clusters-1).
  Vec predict(const Mat& X) const;

  // The learned assignments for the training rows (sklearn's labels_).
  const Vec& labels() const { return labels_; }
  // Final centroids, [k][p] (sklearn's cluster_centers_).
  const Mat& centers() const { return centers_; }
  // Lowest inertia seen over the n_init runs (sklearn's inertia_).
  double inertia() const { return inertia_; }
  // Iterations used by the winning run (sklearn's n_iter_).
  size_t n_iter() const { return n_iter_; }

 private:
  // One k-means++ seeded run on X; fills centers_/labels_, returns inertia.
  double run(const Mat& X, std::uint32_t seed);
  void assign(const Mat& X, const Mat& centers, Vec& labels) const;

  size_t n_clusters_;
  size_t n_init_;
  size_t max_iter_;
  std::uint32_t seed_;
  Vec labels_;
  Mat centers_;
  double inertia_ = 0.0;
  size_t n_iter_ = 0;
};

}  // namespace ml

#endif  // MLC_MODULE_KMEANS_HPP