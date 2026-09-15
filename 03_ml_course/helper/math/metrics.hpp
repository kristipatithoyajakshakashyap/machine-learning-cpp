#ifndef HELPER_MATH_METRICS_HPP
#define HELPER_MATH_METRICS_HPP

// ===========================================================================
// helper/math/metrics.hpp  (+ the matching helper/math/metrics.cpp)
// ---------------------------------------------------------------------------
// Scoring functions for classifiers and regressors, used by EVERY module's
// "03_dataset_implementation" and end-to-end lessons.  The sklearn equivalents
// live in sklearn.metrics (accuracy_score, precision_score, roc_auc_score,
// mean_squared_error, r2_score, ...).  All values use plain double arithmetic
// so the numbers are bit-identical run to run and can be diffed against
// sklearn's output digit-for-digit.
// ===========================================================================

#include <cstddef>
#include <utility>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {
// Per-observation silhouette. Singleton clusters contribute zero.
Vec silhouette_samples(const Mat &X, const Vec &labels);

// -------- classification -------------------------------------------

// Bundle of the usual binary-classification numbers, computed at the
// default 0.5 decision threshold.  tp/fp/tn/fn are the confusion cells.
struct BinaryScores {
  double accuracy = 0.0, precision = 0.0, recall = 0.0, f1 = 0.0, auc = 0.0;
  size_t tp = 0, fp = 0, tn = 0, fn = 0;
  size_t true_positive_ct() const { return tp; }
};

// y_true/y_pred hold 0/1 labels.  Pass `scores` (raw predicted probabilities
// or decision values) to also fill in s.auc.
BinaryScores binary_scores(const Vec &y_true, const Vec &y_pred,
                           const Vec *scores = nullptr);

// Multiclass accuracy for integer-coded labels (exact match fraction).
double accuracy(const Vec &y_true, const Vec &y_pred);

// C x C confusion matrix for integer-coded labels in [0, C).
// Cell [true][prediction] counts matching rows.
Mat confusion(const Vec &y_true, const Vec &y_pred, size_t n_classes);

// Binary log-loss given target labels (0/1) and predicted probabilities
// (clamped to [eps, 1-eps] to avoid log(0)).
double binary_logloss(const Vec &y, const Vec &p);

// Multiclass log-loss: y holds class indices, P[i][j] = P(class j | row i).
double multiclass_logloss(const Vec &y, const Mat &P);

// ROC curve: (fpr, tpr) points for every distinct score threshold (plus the
// +/-inf sentinels so the curve starts at (0,0) and ends at (1,1)), sorted
// left-to-right.  Requires both classes present.
std::vector<std::pair<double, double>> roc_curve(const Vec &y,
                                                 const Vec &scores);
// Area under the ROC curve via the trapezoid rule (equivalent to sklearn's
// roc_auc_score when scores include both classes).
double auc(const Vec &y, const Vec &scores);

// -------- regression ------------------------------------------------

// Mean squared error.
double mse(const Vec &y, const Vec &yhat);
// Root mean squared error (same units as y).
double rmse(const Vec &y, const Vec &yhat);
// Mean absolute error (robust to outliers).
double mae(const Vec &y, const Vec &yhat);
// Coefficient of determination: 1 - SS_res/SS_tot.
double r2(const Vec &y, const Vec &yhat);
// R2 penalised for the number of model features.
double r2_adjusted(const Vec &y, const Vec &yhat, size_t n_features);
// Simple statistics used by many lessons.
double mean(const Vec &v);
double variance(const Vec &v, bool population = true);

// -------- clustering ------------------------------------------------

// Average silhouette score for a hard clustering (sklearn.metrics.
// silhouette_score).  label[i] = cluster id of row i; a(i) = mean distance
// to co-members of its own cluster, b(i) = smallest mean distance to any
// other cluster, s(i) = (b-a)/max(a,b) clipped to 0 when the point is alone
// in its cluster.  Returns the mean of s(i).  Higher (closer to +1) means
// compact, well-separated clusters.
double silhouette(const Mat &X, const Vec &labels);

} // namespace ml

#endif // ML_METRICS_HPP
