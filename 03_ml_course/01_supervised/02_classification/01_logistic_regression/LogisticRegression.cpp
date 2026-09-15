// 03_ml_course/01_supervised/02_classification/01_logistic_regression/LogisticRegression.cpp
// ml::LogisticRegression implementation: binary (sigmoid, BCE loss) and
// multinomial (softmax, cross-entropy loss) logistic regression with an L2
// penalty of strength 1/C on the slopes, trained by Adam on a flat parameter
// vector [w..., b]. Labels are integers 0..n_classes-1 stored as doubles.
//
// Reads:   nothing on disk (data arrives as ml::Mat / ml::Vec).
// Writes:  nothing; save()/load() in LogisticRegression.hpp serialise the model.
// Build:   compiled into the static library ml_clog (with LinearRegression.cpp)
//          and linked by clog_theory, clog_math_intuition, clog_implementation,
//          clog_end_to_end and clog_predict. Tests: ctest -R clog.
// Layout:  weights_[k][j] is the slope of feature j for class k (one row for the
//          binary case); intercept_[k] is the bias of class k. After each fit a
//          finite-difference gradient_check() runs at the solution; its result is
//          intentionally ignored (it exists for debugging and the lessons).
#include "LogisticRegression.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "helper/math/optim.hpp"

namespace ml {

namespace {
// Sigmoid: maps a real-valued score into (0, 1).
// Two branches so exp() is always called with a non-positive argument (no overflow).
double sigmoid(double z) {
  if (z >= 0) {
    double e = std::exp(-z);
    return 1.0 / (1.0 + e);
  }
  double e = std::exp(z);
  return e / (1.0 + e);
}

// Stable softmax: subtract the row max to keep exp() finite.
void softmax_inplace(Vec& v) {
  double mx = *std::max_element(v.begin(), v.end());
  double s = 0.0;
  for (double& x : v) {
    x = std::exp(x - mx);
    s += x;
  }
  for (double& x : v) x /= s;
}
}  // namespace

// Construct an untrained model. n_classes >= 2; C > 0 is the inverse L2 strength;
// lr and epochs configure Adam. Throws std::invalid_argument otherwise.
LogisticRegression::LogisticRegression(size_t n_classes, double C, double lr,
                                       size_t epochs)
    : n_classes_(n_classes), C_(C), lr_(lr), epochs_(epochs) {
  if (n_classes_ < 2 || !std::isfinite(C_) || !(C_ > 0) || !std::isfinite(lr_) || !(lr_ > 0) || epochs_ == 0) throw std::invalid_argument("Invalid logistic parameters");
}

// Fit delegates to the binary or multinomial path — the only difference is
// the activation (sigmoid vs softmax) and the loss (BCE vs cross-entropy).
// X: n x p finite features (rows equal length); y: n labels in [0, n_classes).
// Throws std::runtime_error on a shape mismatch and std::invalid_argument on
// ragged rows, non-finite values or out-of-range labels.
void LogisticRegression::fit(const Mat& X, const Vec& y) {
  if (X.empty() || X[0].empty() || X.size() != y.size())
    throw std::runtime_error("logistic: dimension mismatch");
  for(const auto& row:X){
    if(row.size()!=X[0].size())throw std::invalid_argument("Ragged logistic feature matrix");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("Logistic features must be finite");
  }
  for(double label:y)if(!std::isfinite(label)||label<0||label!=std::floor(label)||label>=static_cast<double>(n_classes_))
    throw std::invalid_argument("Logistic label outside configured class range");
  if (n_classes_ == 2) fit_binary(X, y);
  else fit_multiclass(X, y);
}

// Binary logistic regression:
//   p(w,b) = sigmoid(X w + b)
//   L      = -(1/n) sum [ y log p + (1-y) log(1-p) ] + (1/2C) ||w||^2
// Minimise over the combined parameter vector [w_0..w_{p-1}, b] with Adam.
void LogisticRegression::fit_binary(const Mat& X, const Vec& y) {
  size_t n = X.size();
  size_t p = X[0].size();
  // Flat parameter layout: w[0..p-1] slopes, w[p] intercept. Starting at zero gives
  // p = 0.5 everywhere, a symmetric starting point.
  size_t dim = p + 1;  // p slopes + 1 intercept
  Vec w0(dim, 0.0);

  auto loss = [&](const Vec& w) -> double {
    double l = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double z = w[p];
      for (size_t j = 0; j < p; ++j) z += w[j] * X[i][j];
      double pr = sigmoid(z);
      // Clamp probabilities at 1e-15 so log() stays finite for confident mistakes.
      l -= y[i] * std::log(std::max(pr, 1e-15)) +
           (1.0 - y[i]) * std::log(std::max(1.0 - pr, 1e-15));
    }
    l /= static_cast<double>(n);
    // L2 penalty on the slopes only (like sklearn's C).
    for (size_t j = 0; j < p; ++j) l += 0.5 / C_ * w[j] * w[j];
    return l;
  };
  auto grad = [&](const Vec& w, Vec& g) -> void {
    g.assign(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
      double z = w[p];
      for (size_t j = 0; j < p; ++j) z += w[j] * X[i][j];
      double pr = sigmoid(z);
      double err = pr - y[i];  // dL/dz = (p - y)
      g[p] += err;
      for (size_t j = 0; j < p; ++j) g[j] += err * X[i][j];
    }
    for (size_t d = 0; d < dim; ++d) g[d] /= static_cast<double>(n);
    for (size_t j = 0; j < p; ++j) g[j] += w[j] / C_;
  };

  Vec w = adam(w0, epochs_, lr_, loss, grad).w;
  weights_.assign(1, Vec(p));
  for (size_t j = 0; j < p; ++j) weights_[0][j] = w[j];
  intercept_.assign(1, w[p]);
  (void)gradient_check(loss, grad, w, 1e-4);
}

// Multinomial (softmax) logistic regression: one weight vector per class.
//   p_k(x) = softmax( W_k x + b_k )
//   L      = -(1/n) sum_i log p_{y_i}(x_i) + (1/2C) sum_k ||W_k||^2
void LogisticRegression::fit_multiclass(const Mat& X, const Vec& y) {
  size_t n = X.size();
  size_t p = X[0].size();
  size_t dim = n_classes_ * (p + 1);
  Vec w0(dim, 0.0);

  // Map (class k, param type) <-> flat index for compact gradient math.
  // Class k occupies a contiguous block of p + 1 entries; jj == p is its intercept.
  auto idx = [&](size_t k, size_t jj) { return k * (p + 1) + jj; };

  auto loss = [&](const Vec& w) -> double {
    double l = 0.0;
    for (size_t i = 0; i < n; ++i) {
      Vec scores(n_classes_);
      for (size_t k = 0; k < n_classes_; ++k) {
        scores[k] = w[idx(k, p)];
        for (size_t j = 0; j < p; ++j) scores[k] += w[idx(k, j)] * X[i][j];
      }
      softmax_inplace(scores);
      l -= std::log(std::max(scores[static_cast<size_t>(y[i])], 1e-15));
    }
    l /= static_cast<double>(n);
    for (size_t d = 0; d < dim; ++d) {
      size_t jj = d % (p + 1);
      if (jj < p) l += 0.5 / C_ * w[d] * w[d];
    }
    return l;
  };
  auto grad = [&](const Vec& w, Vec& g) -> void {
    g.assign(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
      Vec scores(n_classes_);
      for (size_t k = 0; k < n_classes_; ++k) {
        scores[k] = w[idx(k, p)];
        for (size_t j = 0; j < p; ++j) scores[k] += w[idx(k, j)] * X[i][j];
      }
      softmax_inplace(scores);
      size_t yi = static_cast<size_t>(y[i]);
      for (size_t k = 0; k < n_classes_; ++k) {
        double err = scores[k] - (k == yi ? 1.0 : 0.0);  // dL/dz_k
        g[idx(k, p)] += err;
        for (size_t j = 0; j < p; ++j) g[idx(k, j)] += err * X[i][j];
      }
    }
    for (size_t d = 0; d < dim; ++d) g[d] /= static_cast<double>(n);
    for (size_t d = 0; d < dim; ++d) {
      size_t jj = d % (p + 1);
      if (jj < p) g[d] += w[d] / C_;
    }
  };

  Vec w = adam(w0, epochs_, lr_, loss, grad).w;
  weights_.assign(n_classes_, Vec(p));
  intercept_.assign(n_classes_, 0.0);
  for (size_t k = 0; k < n_classes_; ++k)
    for (size_t j = 0; j < p; ++j) weights_[k][j] = w[idx(k, j)];
  for (size_t k = 0; k < n_classes_; ++k) intercept_[k] = w[idx(k, p)];
  (void)gradient_check(loss, grad, w, 1e-4);
}

// Hard prediction: argmax probability.
Vec LogisticRegression::predict(const Mat& X) const {
  if (weights_.empty()) throw std::runtime_error("logistic: model not fitted");
  Mat pr = predict_proba(X);
  Vec out(X.size(), 0.0);
  for (size_t i = 0; i < X.size(); ++i) {
    double best = pr[i][0];
    size_t best_k = 0;
    for (size_t k = 1; k < n_classes_; ++k)
      if (pr[i][k] > best) {
        best = pr[i][k];
        best_k = k;
      }
    out[i] = static_cast<double>(best_k);
  }
  return out;
}

// Class probabilities (n x n_classes). Binary models fill column 1 with sigmoid(z)
// and column 0 with its complement; multiclass models use the softmax scores.
// An empty X returns an empty matrix. Throws before fit(), on a feature-count
// mismatch, or on non-finite features.
Mat LogisticRegression::predict_proba(const Mat& X) const {
  if (weights_.empty()) throw std::runtime_error("logistic: model not fitted");
  if(X.empty())return {};
  for(const auto& row:X){
    if(row.size()!=weights_[0].size())throw std::invalid_argument("Logistic prediction feature count mismatch");
    for(double value:row)if(!std::isfinite(value))throw std::invalid_argument("Logistic features must be finite");
  }
  size_t n = X.size(), p = X[0].size();
  Mat proba(n, Vec(n_classes_, 0.0));
  for (size_t i = 0; i < n; ++i) {
    if (n_classes_ == 2) {
      double z = intercept_[0];
      for (size_t j = 0; j < p; ++j) z += weights_[0][j] * X[i][j];
      proba[i][0] = 1.0 - sigmoid(z);
      proba[i][1] = sigmoid(z);
    } else {
      Vec scores(n_classes_);
      for (size_t k = 0; k < n_classes_; ++k) {
        scores[k] = intercept_[k];
        for (size_t j = 0; j < p; ++j) scores[k] += weights_[k][j] * X[i][j];
      }
      softmax_inplace(scores);
      proba[i] = scores;
    }
  }
  return proba;
}

}  // namespace ml
