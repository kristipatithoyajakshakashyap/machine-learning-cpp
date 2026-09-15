// 03_ml_course/02_unsupervised/04_dbscan/DBSCAN.hpp
// Purpose: self-contained DBSCAN (density-based clustering) used by the udb_*
//          lessons and by tests/verify_reload.cpp. Header-only; no .cpp pair.
// Inputs:  a dense finite feature matrix passed to fit(); hyper-parameters
//          eps (neighbourhood radius) and min_samples (density threshold).
// Outputs: none directly (no printing, no files); callers read labels() and
//          core_samples() and persist the object with save()/load().
// Run target: not an executable; linked via the ml_dbscan INTERFACE library.
//
// Algorithm (Ester et al. 1996):
//   1. For every row find its eps-neighbourhood (all rows within distance eps,
//      the row itself included, as in sklearn).
//   2. A row is a CORE point if that neighbourhood holds >= min_samples rows.
//   3. Starting from each unvisited core point, flood-fill through core
//      neighbours; every row reached joins the current cluster.
//   4. Rows never reached from a core point are NOISE (label -1).
// sklearn equivalent: sklearn.cluster.DBSCAN(eps, min_samples, metric='euclidean').
#pragma once
#include <deque>

#include "../validation.hpp"
#include "helper/persistence/archive.hpp"
namespace ml {
// Brute-force Euclidean DBSCAN. Neighbourhoods are found by an all-pairs scan,
// so fit() costs O(n^2 p) time and O(n * avg neighbours) memory; fine for the
// few hundred rows used in this course, not for large n (use a KD-tree then).
// Labels: 0,1,2,... for clusters in discovery order, -1 for noise.
class DBSCAN {
  double eps_;            // neighbourhood radius (compared as squared distance)
  size_t minimum_;        // min_samples, counting the point itself
  Vec labels_;            // one label per fitted row; -1 = noise
  Mat fitted_;            // copy of the training rows (kept for the snapshot)
  std::vector<int> core_; // 1 if row i is a core point, else 0

 public:
  // eps: radius of the neighbourhood ball; min_samples: rows (self included)
  // needed inside that ball for a point to be core. Validated in fit().
  DBSCAN(double eps = .5, size_t min_samples = 5)
      : eps_(eps), minimum_(min_samples) {}
  // Cluster X in place: fills labels_ and core_. Deterministic (no RNG); the
  // cluster ids depend only on row order. Throws on invalid input/parameters.
  void fit(const Mat& X) {
    validate_dense(X);
    // eps must be a positive finite number and min_samples at least 1.
    if (!(eps_ > 0) || !std::isfinite(eps_) || !minimum_)
      throw std::invalid_argument("invalid DBSCAN parameters");
    fitted_ = X;
    // Label sentinel -2 = "not visited yet"; -1 = noise; >= 0 = cluster id.
    labels_.assign(X.size(), -2);
    core_.assign(X.size(), 0);
    // Step 1-2: eps-neighbourhoods and core flags. Comparing squared distances
    // against eps^2 avoids a sqrt per pair without changing the result.
    std::vector<std::vector<size_t>> neighbors(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
      for (size_t j = 0; j < X.size(); ++j)
        if (distance2(X[i], X[j]) <= eps_ * eps_) neighbors[i].push_back(j);
      core_[i] = neighbors[i].size() >= minimum_;
    }
    // Step 3-4: breadth-first expansion from each unvisited core point.
    int cluster = 0;
    for (size_t i = 0; i < X.size(); ++i) {
      if (labels_[i] != -2) continue;  // already assigned or marked noise
      if (!core_[i]) {
        labels_[i] = -1;  // provisional noise; may be claimed later as a border
        continue;
      }
      std::deque<size_t> q{i};
      labels_[i] = cluster;
      while (!q.empty()) {
        size_t u = q.front();
        q.pop_front();
        for (size_t v : neighbors[u]) {
          // A point marked noise earlier that is within eps of a core point
          // becomes a border point of this cluster (but is not expanded).
          if (labels_[v] == -1) labels_[v] = cluster;
          if (labels_[v] != -2) continue;  // visited: skip
          labels_[v] = cluster;
          if (core_[v]) q.push_back(v);  // only core points spread the cluster
        }
      }
      ++cluster;
    }
  }
  // Cluster id per fitted row (-1 = noise). Valid only after fit()/load().
  const Vec& labels() const { return labels_; }
  // Core-point indicator per fitted row (1 = core, 0 = border or noise).
  const std::vector<int>& core_samples() const { return core_; }
  // Persist parameters, labels, the fitted rows and the core flags. DBSCAN has
  // no predict() for new rows, so the snapshot exists to reproduce labels().
  void save(std::ostream& o) const {
    archive::write(o, eps_, minimum_, labels_, fitted_, core_);
  }
  // Inverse of save(). Re-validates the matrix and checks the three per-row
  // vectors have matching length so a truncated file cannot load silently.
  void load(std::istream& i) {
    archive::read(i, eps_, minimum_, labels_, fitted_, core_);
    validate_dense(fitted_);
    if (labels_.size() != fitted_.size() || core_.size() != fitted_.size())
      throw std::runtime_error("invalid DBSCAN snapshot");
  }
};
}  // namespace ml
