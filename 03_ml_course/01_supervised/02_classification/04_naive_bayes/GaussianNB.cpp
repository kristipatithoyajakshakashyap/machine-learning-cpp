// 03_ml_course/01_supervised/02_classification/04_naive_bayes/GaussianNB.cpp
// ml::GaussianNB implementation: per-class feature means/variances plus class
// priors, combined through the naive (conditional independence) assumption into
// a log-posterior per class. Labels must be contiguous integers 0..K-1 and every
// class must appear in the training set.
//
// Reads:   nothing on disk (data arrives as ml::Mat / ml::Vec).
// Writes:  nothing; save()/load() serialise the fitted statistics to a stream.
// Build:   static library ml_cnb, linked by cnb_theory, cnb_math_intuition,
//          cnb_implementation, cnb_end_to_end and cnb_predict. Tests: ctest -R cnb.
// Numerics: moments are accumulated in long double; variances get an additive
//          epsilon = var_smoothing * max feature variance (sklearn convention).
#include "GaussianNB.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ml {
namespace {
// var_smoothing must be finite and >= 0; shared by the constructor, fit and load.
void check_smoothing(double value) {
  if (!std::isfinite(value) || value < 0)
    throw std::invalid_argument("nbayes: smoothing must be finite and nonnegative");
}
}

// Construct an unfitted model; only the smoothing hyper-parameter is validated.
GaussianNB::GaussianNB(double var_smoothing) : var_smoothing_(var_smoothing) {
  check_smoothing(var_smoothing_);
}

// Estimate class priors, per-class means and per-class variances from X (n x p,
// finite, equal-length rows) and labels y. Throws std::invalid_argument on shape
// errors, non-finite values, non-integer or negative labels, or a class index
// that never occurs. The fit is built in a temporary and swapped in at the end,
// so a throw leaves any previous fit untouched.
void GaussianNB::fit(const Mat& X, const Vec& y) {
  check_smoothing(var_smoothing_);
  if (X.empty() || X.front().empty() || X.size() != y.size())
    throw std::invalid_argument("nbayes: training needs matching nonempty rows and labels");
  const size_t n = X.size(), p = X.front().size();
  size_t classes = 0;
  for (size_t i = 0; i < n; ++i) {
    if (X[i].size() != p) throw std::invalid_argument("nbayes: ragged training rows");
    for (double value : X[i])
      if (!std::isfinite(value)) throw std::invalid_argument("nbayes: nonfinite training feature");
    // A contiguous encoding on n rows cannot contain label >= n. Validate
    // before converting double to size_t or allocating class arrays.
    if (!std::isfinite(y[i]) || y[i] < 0 || std::floor(y[i]) != y[i] ||
        y[i] >= static_cast<double>(n))
      throw std::invalid_argument("nbayes: labels must be contiguous integers starting at zero");
    classes = std::max(classes, static_cast<size_t>(y[i]) + 1);
  }
  std::vector<size_t> count(classes, 0);
  for (double label : y) ++count[static_cast<size_t>(label)];
  for (size_t frequency : count)
    if (frequency == 0) throw std::invalid_argument("nbayes: missing class in contiguous encoding");

  // Compute a candidate so rejected input never damages an existing fit.
  // Extended precision prevents avoidable intermediate overflow in moments.
  GaussianNB candidate(var_smoothing_);
  candidate.n_classes_ = classes;
  candidate.class_prior_.resize(classes);
  candidate.classes_.resize(classes);
  candidate.class_mean_.assign(classes, Vec(p));
  candidate.class_var_.assign(classes, Vec(p));
  std::vector<std::vector<long double>> means(classes, std::vector<long double>(p));
  std::vector<std::vector<long double>> variances(classes, std::vector<long double>(p));
  std::vector<long double> global_mean(p, 0), global_var(p, 0);
  for (size_t i = 0; i < n; ++i) {
    const size_t k = static_cast<size_t>(y[i]);
    for (size_t j = 0; j < p; ++j) {
      means[k][j] += static_cast<long double>(X[i][j]) / count[k];
      global_mean[j] += static_cast<long double>(X[i][j]) / n;
    }
  }
  for (size_t i = 0; i < n; ++i) {
    const size_t k = static_cast<size_t>(y[i]);
    for (size_t j = 0; j < p; ++j) {
      const long double delta = static_cast<long double>(X[i][j]) - means[k][j];
      const long double global_delta = static_cast<long double>(X[i][j]) - global_mean[j];
      variances[k][j] += delta * delta / count[k];
      global_var[j] += global_delta * global_delta / n;
    }
  }
  // Variance floor: the larger of machine epsilon and var_smoothing * largest
  // global feature variance, added to every class variance so log(var) and the
  // division in predict_proba stay finite for constant features.
  const long double maximum_variance = *std::max_element(global_var.begin(), global_var.end());
  const long double epsilon = std::max(
      static_cast<long double>(std::numeric_limits<double>::epsilon()),
      static_cast<long double>(var_smoothing_) * maximum_variance);
  for (size_t k = 0; k < classes; ++k) {
    candidate.class_prior_[k] = static_cast<double>(count[k]) / n;
    candidate.classes_[k] = static_cast<double>(k);
    for (size_t j = 0; j < p; ++j) {
      candidate.class_mean_[k][j] = static_cast<double>(means[k][j]);
      candidate.class_var_[k][j] = static_cast<double>(variances[k][j] + epsilon);
    }
  }
  candidate.validate_state();
  *this = std::move(candidate);
}

// Consistency checks on the fitted (or loaded) state: array sizes agree, priors
// are in (0, 1] and sum to 1, means finite, variances finite and positive.
// Throws std::runtime_error so a corrupt archive cannot be used for prediction.
void GaussianNB::validate_state() const {
  check_smoothing(var_smoothing_);
  if (!n_classes_ || class_prior_.size() != n_classes_ || class_mean_.size() != n_classes_ ||
      class_var_.size() != n_classes_ || classes_.size() != n_classes_ || class_mean_.front().empty())
    throw std::runtime_error("nbayes: unfitted or invalid archive dimensions");
  const size_t p = class_mean_.front().size();
  double prior_sum = 0;
  for (size_t k = 0; k < n_classes_; ++k) {
    if (classes_[k] != static_cast<double>(k) || !std::isfinite(class_prior_[k]) ||
        class_prior_[k] <= 0 || class_prior_[k] > 1 ||
        class_mean_[k].size() != p || class_var_[k].size() != p)
      throw std::runtime_error("nbayes: invalid class state");
    prior_sum += class_prior_[k];
    for (size_t j = 0; j < p; ++j)
      if (!std::isfinite(class_mean_[k][j]) || !std::isfinite(class_var_[k][j]) || class_var_[k][j] <= 0)
        throw std::runtime_error("nbayes: invalid or overflowing Gaussian statistics");
  }
  if (std::abs(prior_sum - 1) > 1e-8) throw std::runtime_error("nbayes: priors do not sum to one");
}

// Guard for prediction: model fitted, every row has p finite features.
void GaussianNB::validate_rows(const Mat& X) const {
  validate_state();
  const size_t p = class_mean_.front().size();
  for (const auto& row : X) {
    if (row.size() != p) throw std::invalid_argument("nbayes: prediction feature count mismatch");
    for (double value : row)
      if (!std::isfinite(value)) throw std::invalid_argument("nbayes: nonfinite prediction feature");
  }
}

// Posterior probabilities (n x K). For each class the log score is
//   log pi_k - 0.5 * sum_j [ log(2 pi) + log var_kj + (x_j - mu_kj)^2 / var_kj ]
// then a max-shifted exp normalises across classes (log-sum-exp trick).
// Throws std::overflow_error if every score is non-finite.
Mat GaussianNB::predict_proba(const Mat& X) const {
  validate_rows(X);
  Mat result;
  result.reserve(X.size());
  constexpr long double log_two_pi = 1.837877066409345483560659472811L;
  for (const auto& row : X) {
    std::vector<long double> scores(n_classes_);
    for (size_t k = 0; k < n_classes_; ++k) {
      long double score = std::log(static_cast<long double>(class_prior_[k]));
      for (size_t j = 0; j < row.size(); ++j) {
        const long double variance = class_var_[k][j];
        const long double delta = static_cast<long double>(row[j]) - class_mean_[k][j];
        score -= 0.5L * (log_two_pi + std::log(variance) + delta * delta / variance);
      }
      scores[k] = score;
    }
    const long double maximum = *std::max_element(scores.begin(), scores.end());
    if (!std::isfinite(maximum)) throw std::overflow_error("nbayes: likelihood exceeds numeric range");
    long double sum = 0;
    for (long double& score : scores) { score = std::exp(score - maximum); sum += score; }
    Vec probabilities;
    for (long double score : scores) probabilities.push_back(static_cast<double>(score / sum));
    result.push_back(std::move(probabilities));
  }
  return result;
}

// Argmax of predict_proba per row; ties resolve to the lowest class index.
Vec GaussianNB::predict(const Mat& X) const {
  Vec predictions;
  for (const auto& row : predict_proba(X))
    predictions.push_back(static_cast<double>(std::max_element(row.begin(), row.end()) - row.begin()));
  return predictions;
}

// Serialise tag, smoothing and the fitted statistics; refuses an unfitted model.
void GaussianNB::save(std::ostream& out) const {
  validate_state();
  archive::write(out, std::string("GaussianNB"), var_smoothing_, n_classes_,
                 class_prior_, class_mean_, class_var_, classes_);
}

// Read into a temporary, validate, then swap in so a bad archive cannot leave
// the object half-updated.
void GaussianNB::load(std::istream& in) {
  std::string tag;
  archive::read(in, tag);
  if (tag != "GaussianNB") throw std::runtime_error("Wrong model type");
  GaussianNB candidate;
  archive::read(in, candidate.var_smoothing_, candidate.n_classes_, candidate.class_prior_,
                candidate.class_mean_, candidate.class_var_, candidate.classes_);
  candidate.validate_state();
  *this = std::move(candidate);
}

}  // namespace ml
