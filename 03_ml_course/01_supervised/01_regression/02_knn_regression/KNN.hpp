#ifndef MLC_MODULE_KNN_HPP
#define MLC_MODULE_KNN_HPP

// k-Nearest Neighbours: brute-force distance lookup.  The sklearn
// equivalents are sklearn.neighbors.KNeighborsRegressor and
// KNeighborsClassifier with algorithm='brute' and metric='minkowski' p=2
// (Euclidean).
//
// fit() stores the training set; predict() computes distances from every
// test row to every training row, picks the k closest, and either averages
// the targets (regression) or takes a majority vote (classification).
// Deterministic: ties in classification are broken by class order (smallest
// class index wins), matching sklearn's np.argmax(np.bincount).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"

namespace ml {

// ---------------------------------------------------------------------------
// KNNRegressor: yhat = (1/k) * sum of y over k nearest neighbours.
// ---------------------------------------------------------------------------
class KNNRegressor {
 public:
  explicit KNNRegressor(size_t k = 5) : k_(k) {}
  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  size_t k() const { return k_; }

  void save(std::ostream& out) const { archive::write(out,std::string("KNNRegressor"),k_,X_train_,y_train_); }
  void load(std::istream& in);
 private:
  size_t k_;
  Mat X_train_;
  Vec y_train_;
};

// ---------------------------------------------------------------------------
// KNNClassifier: majority vote among k nearest neighbours.
// predict_proba returns class fractions (proportion of each class in the
// neighbourhood).
// Training labels must be contiguous integer codes 0, ..., C-1. Features
// and targets must be finite; impute missing features before fitting.
// ---------------------------------------------------------------------------
class KNNClassifier {
 public:
  KNNClassifier(size_t k = 5) : k_(k) {}
  void fit(const Mat& X, const Vec& y);
  Vec predict(const Mat& X) const;
  Mat predict_proba(const Mat& X) const;
  size_t k() const { return k_; }
  size_t n_classes() const { return n_classes_; }

  void save(std::ostream& out) const { archive::write(out,std::string("KNNClassifier"),k_,n_classes_,X_train_,y_train_); }
  void load(std::istream& in);
 private:
  size_t k_;
  size_t n_classes_ = 0;
  Mat X_train_;
  Vec y_train_;
};

}  // namespace ml

#endif  // ML_KNN_HPP
