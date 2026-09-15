#include "KNN.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ml {

namespace {
void validate_features(const Mat& X,size_t columns) {
  for(const auto& row:X){
    if(row.size()!=columns)throw std::invalid_argument("knn: feature count mismatch");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("knn: features must be finite");
  }
}
void validate_training(const Mat& X,const Vec& y,size_t k) {
  if(X.empty()||X[0].empty()||X.size()!=y.size())throw std::invalid_argument("knn: nonempty matching training data required");
  if(k==0||k>X.size())throw std::invalid_argument("k must be between 1 and training rows");
  validate_features(X,X[0].size());
  for(double value:y)if(!std::isfinite(value))throw std::invalid_argument("knn: targets must be finite");
}
size_t validated_class_count(const Vec& y) {
  // A contiguous encoding has at most n classes, so check that bound before
  // converting to an unsigned index or allocating a class-count vector.
  size_t classes=0;
  std::vector<bool> present(y.size(),false);
  for(double label:y){
    if(label<0||label!=std::floor(label)||label>=static_cast<double>(y.size()))
      throw std::invalid_argument("knn: class codes must be contiguous integers from zero");
    const size_t code=static_cast<size_t>(label);present[code]=true;classes=std::max(classes,code+1);
  }
  for(size_t code=0;code<classes;++code)if(!present[code])throw std::invalid_argument("knn: missing class code in encoding");
  return classes;
}
// Squared Euclidean distance (p=2 Minkowski, which is sklearn's default
// metric).  Distances are compared on their squares to avoid sqrt().
double dist2(const Vec& a, const Vec& b) {
  if (a.size() != b.size())
    throw std::runtime_error("knn: feature count mismatch");
  double d = 0.0;
  for (size_t j = 0; j < a.size(); ++j) d += (a[j] - b[j]) * (a[j] - b[j]);
  return d;
}
}  // namespace

// ---- KNNRegressor --------------------------------------------------------
void KNNRegressor::fit(const Mat& X, const Vec& y) {
  validate_training(X,y,k_);
  X_train_ = X;
  y_train_ = y;
}

// For each test row: find the k nearest training rows, average their targets.
Vec KNNRegressor::predict(const Mat& X) const {
  if (X_train_.empty())
    throw std::runtime_error("knn: model not fitted");
  validate_features(X,X_train_[0].size());
  Vec out(X.size(), 0.0);
  for (size_t i = 0; i < X.size(); ++i) {
    // Gather (distance^2, training index) for every training row.
    std::vector<std::pair<double, size_t>> ds;
    ds.reserve(X_train_.size());
    for (size_t t = 0; t < X_train_.size(); ++t)
      ds.emplace_back(dist2(X[i], X_train_[t]), t);
    std::partial_sort(
        ds.begin(), ds.begin() + static_cast<ptrdiff_t>(std::min(k_, ds.size())),
        ds.end());
    double sum = 0.0;
    for (size_t m = 0; m < k_ && m < ds.size(); ++m)
      sum += y_train_[ds[m].second];
    out[i] = sum / static_cast<double>(std::min(k_, ds.size()));
  }
  return out;
}

// ---- KNNClassifier -------------------------------------------------------
void KNNClassifier::fit(const Mat& X, const Vec& y) {
  validate_training(X,y,k_);
  const size_t classes=validated_class_count(y);
  X_train_ = X;
  y_train_ = y;
  n_classes_ = classes;
}

// Majority vote over the k nearest neighbours.  Ties are broken by the
// small-voted-class index (matches sklearn's np.argmax(np.bincount)).
Vec KNNClassifier::predict(const Mat& X) const {
  if (X_train_.empty()) throw std::runtime_error("knn: model not fitted");
  validate_features(X,X_train_[0].size());
  Vec out(X.size(), 0.0);
  for (size_t i = 0; i < X.size(); ++i) {
    std::vector<std::pair<double, size_t>> ds;
    ds.reserve(X_train_.size());
    for (size_t t = 0; t < X_train_.size(); ++t)
      ds.emplace_back(dist2(X[i], X_train_[t]), t);
    std::partial_sort(
        ds.begin(), ds.begin() + static_cast<ptrdiff_t>(std::min(k_, ds.size())),
        ds.end());
    // Count class votes.  Use a vector sized n_classes_; track the leader.
    Vec votes(n_classes_, 0.0);
    size_t best = 0;
    for (size_t m = 0; m < k_ && m < ds.size(); ++m) {
      size_t c = static_cast<size_t>(y_train_[ds[m].second]);
      ++votes[c];

    }
    for(size_t c=0;c<votes.size();++c) if(votes[c]>votes[best]) best=c;
    out[i] = static_cast<double>(best);
  }
  return out;
}

// Class fractions in the k-neighbourhood (sklearn predict_proba).
Mat KNNClassifier::predict_proba(const Mat& X) const {
  if (X_train_.empty()) throw std::runtime_error("knn: model not fitted");
  validate_features(X,X_train_[0].size());
  Mat proba(X.size(), Vec(n_classes_, 0.0));
  for (size_t i = 0; i < X.size(); ++i) {
    std::vector<std::pair<double, size_t>> ds;
    ds.reserve(X_train_.size());
    for (size_t t = 0; t < X_train_.size(); ++t)
      ds.emplace_back(dist2(X[i], X_train_[t]), t);
    std::partial_sort(
        ds.begin(), ds.begin() + static_cast<ptrdiff_t>(std::min(k_, ds.size())),
        ds.end());
    size_t km = std::min(k_, ds.size());
    for (size_t m = 0; m < km; ++m)
      ++proba[i][static_cast<size_t>(y_train_[ds[m].second])];
    for (double& v : proba[i]) v /= static_cast<double>(km);
  }
  return proba;
}

void KNNRegressor::load(std::istream& in) {
  std::string tag;size_t k=0;Mat X;Vec y;
  archive::read(in,tag);
  if(tag!="KNNRegressor")throw std::runtime_error("Wrong model type");
  archive::read(in,k,X,y);
  KNNRegressor candidate(k);candidate.fit(X,y);
  *this=std::move(candidate);
}

void KNNClassifier::load(std::istream& in) {
  std::string tag;size_t k=0,saved_classes=0;Mat X;Vec y;
  archive::read(in,tag);
  if(tag!="KNNClassifier")throw std::runtime_error("Wrong model type");
  archive::read(in,k,saved_classes,X,y);
  KNNClassifier candidate(k);candidate.fit(X,y);
  if(candidate.n_classes_!=saved_classes)throw std::runtime_error("Invalid archived class count");
  *this=std::move(candidate);
}
}  // namespace ml
