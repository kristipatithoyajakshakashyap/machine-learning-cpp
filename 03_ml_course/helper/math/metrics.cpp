#include "helper/math/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace ml {
namespace {
void matched(const Vec &y, const Vec &p) {
  if (y.empty() || y.size() != p.size())
    throw std::invalid_argument("Metrics require equal nonempty vectors");
  for (size_t i = 0; i < y.size(); ++i)
    if (!std::isfinite(y[i]) || !std::isfinite(p[i]))
      throw std::invalid_argument("Nonfinite metric input");
}
} // namespace

// ---------------------------------------------------------------------------
// Metrics
//
// Every scorer takes observed values first, predicted values second, and
// returns a double so lesson output is easy to diff against sklearn.
// ---------------------------------------------------------------------------

// Arithmetic mean of a vector (empty -> 0).
double mean(const Vec &v) {
  if (v.empty())
    return 0.0;
  double s = 0.0;
  for (double x : v)
    s += x;
  return s / static_cast<double>(v.size());
}

// Variance: population uses /N, sample uses /(N-1) (sklearn default).
double variance(const Vec &v, bool population) {
  if (v.size() < 2)
    return 0.0;
  double m = mean(v);
  double s = 0.0;
  for (double x : v)
    s += (x - m) * (x - m);
  double denom = static_cast<double>(v.size()) - (population ? 0.0 : 1.0);
  if (denom <= 0.0)
    denom = 1.0;
  return s / denom;
}

// Full report for a binary task.  y_true and y_pred hold 0/1 labels; the
// optional `scores` vector lets the caller also get the AUC (per-class raw
// scores), matching sklearn's classification_report + roc_auc_score.
BinaryScores binary_scores(const Vec &y_true, const Vec &y_pred,
                           const Vec *scores) {
  if (y_true.size() != y_pred.size())
    throw std::runtime_error("binary_scores: length mismatch");
  matched(y_true, y_pred);
  BinaryScores s;
  size_t n = y_true.size();
  // Count the four confusion-matrix cells for a 0.5 decision threshold.
  for (size_t i = 0; i < n; ++i) {
    int t = y_true[i] > 0.5 ? 1 : 0;
    int p = y_pred[i] > 0.5 ? 1 : 0;
    if (t == 1 && p == 1)
      ++s.tp;
    if (t == 0 && p == 1)
      ++s.fp;
    if (t == 0 && p == 0)
      ++s.tn;
    if (t == 1 && p == 0)
      ++s.fn;
  }
  // Derived rates; guards against divide-by-zero on degenerate data.
  s.accuracy = static_cast<double>(s.tp + s.tn) / static_cast<double>(n);
  s.precision = (s.tp + s.fp) ? static_cast<double>(s.tp) /
                                    static_cast<double>(s.tp + s.fp)
                              : 0.0;
  s.recall = (s.tp + s.fn)
                 ? static_cast<double>(s.tp) / static_cast<double>(s.tp + s.fn)
                 : 0.0;
  s.f1 = (s.precision + s.recall) > 0.0
             ? 2.0 * s.precision * s.recall / (s.precision + s.recall)
             : 0.0;
  if (scores)
    s.auc = auc(y_true, *scores);
  return s;
}

// Exact-match accuracy for integer (or other) labels: fraction of rows whose
// predicted label equals the true label.
double accuracy(const Vec &y_true, const Vec &y_pred) {
  matched(y_true, y_pred);
  size_t hit = 0;
  for (size_t i = 0; i < y_true.size(); ++i)
    if (y_true[i] == y_pred[i])
      ++hit;
  return static_cast<double>(hit) / static_cast<double>(y_true.size());
}

// C x C confusion matrix: cell [true][prediction] counts rows of that pair.
// Labels outside [0, C) are silently skipped.
Mat confusion(const Vec &y_true, const Vec &y_pred, size_t n_classes) {
  matched(y_true, y_pred);
  if (n_classes == 0)
    throw std::invalid_argument("Zero classes");
  Mat C(n_classes, Vec(n_classes, 0.0));
  for (size_t i = 0; i < y_true.size(); ++i) {
    size_t t = static_cast<size_t>(y_true[i]);
    size_t p = static_cast<size_t>(y_pred[i]);
    if (y_true[i] < 0 || y_pred[i] < 0 || y_true[i] != std::floor(y_true[i]) ||
        y_pred[i] != std::floor(y_pred[i]) || t >= n_classes || p >= n_classes)
      throw std::invalid_argument("Invalid confusion label");
    C[t][p] += 1.0;
  }
  return C;
}

// Binary cross-entropy (log loss).  Probabilities are clamped to [eps, 1-eps]
// so log(0) never fires; matches sklearn log_loss on identical probabilities.
double binary_logloss(const Vec &y, const Vec &p) {
  matched(y, p);
  const double eps = 1e-15;
  double s = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    double pi = std::min(std::max(p[i], eps), 1.0 - eps);
    s += y[i] * std::log(pi) + (1.0 - y[i]) * std::log(1.0 - pi);
  }
  return -s / static_cast<double>(y.size());
}

// Multiclass cross-entropy: P[i][c] holds P(class c | row i); average over
// rows of -log P(true class).  Probability clamp as above.
double multiclass_logloss(const Vec &y, const Mat &P) {
  const double eps = 1e-15;
  double s = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    size_t c = static_cast<size_t>(y[i]);
    double pi = std::min(std::max(P[i][c], eps), 1.0 - eps);
    s += std::log(pi);
  }
  return -s / static_cast<double>(y.size());
}

// ROC points: for every distinct score used as a threshold (plus +/-inf
// sentinels so the curve reaches (0,0) and (1,1)), compute FPR (x) and TPR
// (y).  Thresholds are walked in DECREASING order so the returned pairs run
// left-to-right from (0,0) to (1,1).  Requires both classes present.
std::vector<std::pair<double, double>> roc_curve(const Vec &y,
                                                 const Vec &scores) {
  matched(y, scores);
  std::vector<double> thr;
  for (double s : scores)
    thr.push_back(s);
  thr.push_back(-std::numeric_limits<double>::infinity());
  thr.push_back(std::numeric_limits<double>::infinity());
  // Decreasing threshold order so the returned points run (0,0) -> (1,1):
  // at t=+inf nothing is flagged (0,0), and as t drops more samples are
  // flagged until everything is flagged at t=-inf (1,1).
  std::sort(thr.rbegin(), thr.rend());
  thr.erase(std::unique(thr.begin(), thr.end()), thr.end());

  size_t n = y.size();
  size_t npos = 0;
  for (double v : y)
    if (v > 0.5)
      ++npos;
  size_t nneg = n - npos;
  if (npos == 0 || nneg == 0)
    throw std::runtime_error("roc_curve: need both classes");

  std::vector<std::pair<double, double>> pts; // (fpr, tpr)
  for (double t : thr) {
    double tp = 0, fp = 0;
    for (size_t i = 0; i < n; ++i) {
      bool pos = scores[i] >= t; // sample flagged positive at threshold t
      if (pos && y[i] > 0.5)
        tp += 1.0;
      if (pos && y[i] <= 0.5)
        fp += 1.0;
    }
    pts.emplace_back(fp / static_cast<double>(nneg),
                     tp / static_cast<double>(npos));
  }
  // Ensure monotone (sort ascending by fpr keeps order since thr ascending).
  return pts;
}

// Area under the ROC curve via the trapezoid rule on the curve points.
double auc(const Vec &y, const Vec &scores) {
  auto pts = roc_curve(y, scores);
  double area = 0.0;
  for (size_t i = 1; i < pts.size(); ++i) {
    double x0 = pts[i - 1].first, y0 = pts[i - 1].second;
    double x1 = pts[i].first, y1 = pts[i].second;
    area += (x1 - x0) * (y0 + y1) / 2.0; // trapezoid height x average width
  }
  return area;
}

// Mean squared error.
double mse(const Vec &y, const Vec &yhat) {
  matched(y, yhat);
  double s = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    double d = y[i] - yhat[i];
    s += d * d;
  }
  return s / static_cast<double>(y.size());
}

// Root mean squared error (same units as y).
double rmse(const Vec &y, const Vec &yhat) { return std::sqrt(mse(y, yhat)); }

// Mean absolute error (robust to outliers).
double mae(const Vec &y, const Vec &yhat) {
  matched(y, yhat);
  double s = 0.0;
  for (size_t i = 0; i < y.size(); ++i)
    s += std::fabs(y[i] - yhat[i]);
  return s / static_cast<double>(y.size());
}

// Coefficient of determination: 1 - SS_res / SS_tot.  1.0 = perfect fit,
// 0 = predicting the mean, negative = worse than predicting the mean.
double r2(const Vec &y, const Vec &yhat) {
  matched(y, yhat);
  double ss_res = 0.0, ss_tot = 0.0;
  double m = mean(y);
  for (size_t i = 0; i < y.size(); ++i) {
    ss_res += (y[i] - yhat[i]) * (y[i] - yhat[i]);
    ss_tot += (y[i] - m) * (y[i] - m);
  }
  if (ss_tot <= 0.0)
    return ss_res == 0.0 ? 1.0 : 0.0;
  return 1.0 - ss_res / ss_tot;
}

// R2 adjusted for the number of features (penalises adding useless columns).
double r2_adjusted(const Vec &y, const Vec &yhat, size_t n_features) {
  double r = r2(y, yhat);
  size_t n = y.size();
  if (n < 2 || n_features >= n - 1)
    throw std::invalid_argument("Adjusted R2 needs n > p+1");
  return 1.0 - (1.0 - r) * static_cast<double>(n - 1) /
                   static_cast<double>(n - n_features - 1);
}

Vec silhouette_samples(const Mat &X, const Vec &labels) {
  const size_t n = X.size();
  if (n < 3 || labels.size() != n || X[0].empty())
    throw std::invalid_argument("Silhouette needs at least 3 rows and labels");
  std::map<double, size_t> cluster;
  for (double label : labels) {
    if (!std::isfinite(label))
      throw std::invalid_argument("Nonfinite cluster label");
    if (!cluster.count(label))
      cluster[label] = cluster.size();
  }
  const size_t k = cluster.size();
  if (k < 2 || k >= n)
    throw std::invalid_argument("Silhouette requires 2 <= clusters < samples");
  std::vector<size_t> count(k), encoded(n);
  for (size_t i = 0; i < n; ++i) {
    if (X[i].size() != X[0].size())
      throw std::invalid_argument("Ragged silhouette input");
    for (double v : X[i])
      if (!std::isfinite(v))
        throw std::invalid_argument("Nonfinite silhouette feature");
    encoded[i] = cluster.at(labels[i]);
    ++count[encoded[i]];
  }
  Mat sums(n, Vec(k, 0));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j) {
      double ds = 0;
      for (size_t f = 0; f < X[0].size(); ++f) {
        double v = X[i][f] - X[j][f];
        ds += v * v;
      }
      double d = std::sqrt(ds);
      sums[i][encoded[j]] += d;
      sums[j][encoded[i]] += d;
    }
  Vec result(n, 0);
  for (size_t i = 0; i < n; ++i) {
    size_t c = encoded[i];
    if (count[c] == 1)
      continue;
    double a = sums[i][c] / static_cast<double>(count[c] - 1),
           b = std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < k; ++j)
      if (j != c)
        b = std::min(b, sums[i][j] / static_cast<double>(count[j]));
    double denom = std::max(a, b);
    result[i] = denom > 0 ? (b - a) / denom : 0;
  }
  return result;
}
double silhouette(const Mat &X, const Vec &labels) {
  return mean(silhouette_samples(X, labels));
}
} // namespace ml
