// 03_ml_course/02_unsupervised/01_kmeans/KMeans.cpp
// Purpose: implementation of ml::KMeans (declared in KMeans.hpp): k-means++
//          seeding, Lloyd iterations, multi-restart selection, nearest-centroid
//          prediction and text save/load.
// Inputs:  a dense finite matrix passed to fit()/predict() by the lessons.
// Outputs: none of its own (prints nothing, writes nothing); the end_to_end
//          lesson persists the state via save().
// Run target: compiled into the static library ml_kmeans, linked by ukm_*,
//             uce_*, unsupervised_tests and unsupervised_reload.
#include "KMeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "../validation.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

namespace {
// Deterministic 32-bit RNG.  std::mt19937 is used here purely so results are
// reproducible across runs (same seed -> same centroids, like sklearn with
// random_state).  The distribution used by k-means++ picks candidate
// centroids proportionally to their distance to the current set.
// Squared Euclidean distance; no sqrt because only comparisons are needed.
// Complexity O(p).
double squared_distance(const Vec& a, const Vec& b) {
  double s = 0.0;
  for (size_t c = 0; c < a.size(); ++c) {
    double d = a[c] - b[c];
    s += d * d;
  }
  return s;
}
}  // namespace

// Store hyper-parameters only; nothing is computed until fit().
KMeans::KMeans(size_t n_clusters, size_t n_init, size_t max_iter,
               std::uint32_t seed)
    : n_clusters_(n_clusters),
      n_init_(n_init),
      max_iter_(max_iter),
      seed_(seed) {}

// ASSIGN step: label every row with the index of its nearest centroid.
// Parameters: X [n][p], centers [k][p]; labels is resized to n and overwritten.
// Ties go to the lowest centroid index. Complexity O(n*k*p).
void KMeans::assign(const Mat& X, const Mat& centers, Vec& labels) const {
  labels.resize(X.size());
  for (size_t i = 0; i < X.size(); ++i) {
    double best = std::numeric_limits<double>::infinity();
    size_t best_k = 0;
    for (size_t k = 0; k < centers.size(); ++k) {
      double d = squared_distance(X[i], centers[k]);
      if (d < best) {
        best = d;
        best_k = k;
      }
    }
    labels[i] = static_cast<double>(best_k);
  }
}

// One complete k-means run from a given seed.
// Parameters: X - standardized rows; seed - drives both k-means++ and nothing
//             else (Lloyd's step is deterministic given the seeds).
// Returns the run's inertia; side effects: centers_, labels_, n_iter_ hold this
// run's result (fit() decides whether to keep them).
// Complexity O(n*k*p) per seeding step and per Lloyd iteration.
double KMeans::run(const Mat& X, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  const size_t n = X.size();
  const size_t p = X[0].size();

  // --- k-means++ seeding ------------------------------------------------
  // D[i] tracks the squared distance from row i to its nearest chosen centroid;
  // the next centroid is drawn with probability proportional to D[i], which
  // spreads the seeds out and makes Lloyd's step converge to better optima.
  Mat c;
  c.reserve(n_clusters_);
  Vec D(n, std::numeric_limits<double>::infinity());
  // First centroid: a uniform random row.
  c.push_back(X[static_cast<size_t>(gen() % n)]);
  for (size_t k = 1; k < n_clusters_; ++k) {
    double dsum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double d = squared_distance(X[i], c.back());
      D[i] = std::min(D[i], d);
      dsum += D[i];
    }
    // Inverse-CDF sampling over the cumulative D[] weights.
    double target = unit(gen) * dsum;
    double acc = 0.0;
    size_t pick = n - 1;  // fallback guards rounding at the very end
    for (size_t i = 0; i < n; ++i) {
      acc += D[i];
      if (acc >= target) {
        pick = i;
        break;
      }
    }
    c.push_back(X[pick]);
  }

  // --- Lloyd's iteration ------------------------------------------------
  // Alternate ASSIGN (nearest centroid) and UPDATE (centroid = mean of its
  // rows) until no centroid moves or the max_iter budget is used up.
  Vec labels(n);
  size_t it = 0;
  while (it < max_iter_) {
    ++it;
    assign(X, c, labels);
    Mat new_c(n_clusters_, Vec(p, 0.0));
    std::vector<size_t> cnt(n_clusters_, 0);
    for (size_t i = 0; i < n; ++i) {
      size_t k = static_cast<size_t>(labels[i]);
      for (size_t j = 0; j < p; ++j) new_c[k][j] += X[i][j];
      ++cnt[k];
    }
    bool converged = true;
    for (size_t k = 0; k < n_clusters_; ++k) {
      if (cnt[k] > 0)
        for (size_t j = 0; j < p; ++j)
          new_c[k][j] /= static_cast<double>(cnt[k]);
      else
        new_c[k] = c[k];  // Empty clusters keep their previous center.
      if (new_c[k] != c[k]) converged = false;  // exact compare: fixed point
      c[k] = new_c[k];
    }
    if (converged) break;
  }

  // Inertia of this run: sum of squared distances to the assigned centroid.
  // Re-assign once more so labels match the final centroids exactly.
  assign(X, c, labels);
  double inertia = 0.0;
  for (size_t i = 0; i < n; ++i)
    inertia += squared_distance(X[i], c[static_cast<size_t>(labels[i])]);
  centers_ = c;
  labels_ = labels;
  n_iter_ = it;
  return inertia;
}

// Fit with n_init restarts (seeds seed_, seed_+1, ...) and keep the run with the
// lowest inertia. Guards: finite dense X, positive budgets, 1 <= k <= n.
// Complexity O(n_init * max_iter * n * k * p) worst case.
void KMeans::fit(const Mat& X) {
  validate_dense(X);
  if (!n_init_ || !max_iter_)
    throw std::invalid_argument(
        "KMeans: positive restart and iteration budgets required");
  if (n_clusters_ == 0 || n_clusters_ > X.size())
    throw std::runtime_error("KMeans: invalid n_clusters");

  double best_inertia = std::numeric_limits<double>::infinity();
  Mat best_centers;
  Vec best_labels;
  size_t best_iter = 0;
  for (size_t r = 0; r < n_init_; ++r) {
    double in = run(X, static_cast<std::uint32_t>(seed_ + r));
    if (in < best_inertia) {
      best_inertia = in;
      best_centers = centers_;
      best_labels = labels_;
      best_iter = n_iter_;
    }
  }
  centers_ = best_centers;
  labels_ = best_labels;
  inertia_ = best_inertia;
  n_iter_ = best_iter;
}

// Nearest-centroid labels for new rows (must have the training width).
// Returns Vec of cluster ids in [0, k). Throws logic_error before fit().
Vec KMeans::predict(const Mat& X) const {
  if (centers_.empty()) throw std::logic_error("KMeans: fit first");
  if (!X.empty()) validate_dense(X, centers_[0].size());
  Vec out;
  assign(X, centers_, out);
  return out;
}

// Text persistence via helper/persistence/archive.hpp. Field order must match
// between save and load; load re-validates the centroids so a corrupt file
// fails loudly instead of producing garbage predictions.
void KMeans::save(std::ostream& o) const {
  archive::write(o, n_clusters_, n_init_, max_iter_, seed_, labels_, centers_,
                 inertia_, n_iter_);
}
void KMeans::load(std::istream& i) {
  archive::read(i, n_clusters_, n_init_, max_iter_, seed_, labels_, centers_,
                inertia_, n_iter_);
  validate_dense(centers_);
}

}  // namespace ml
