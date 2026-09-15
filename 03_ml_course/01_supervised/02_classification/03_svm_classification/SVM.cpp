#include "SVM.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "helper/math/optim.hpp"

namespace ml {

// ---------------------------------------------------------------------------
// LinearSVM: hinge-loss L2-SVM trained with Adam.
//   L(w, b) = (1/n) sum_i max(0, 1 - y_i (w . x_i + b)) + (1/2C) ||w||^2
// y_i in {+1, -1} (converted internally).  Decision = sign(w . x + b).
// This matches sklearn SGDClassifier(loss='hinge', penalty='l2') closely.
// ---------------------------------------------------------------------------

namespace {
void positive_finite(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) throw std::invalid_argument(name);
}

void validate_linear_parameters(double C, double lr, size_t epochs) {
  positive_finite(C, "LinearSVM: C must be positive and finite");
  positive_finite(lr, "LinearSVM: learning rate must be positive and finite");
  if (!epochs)
    throw std::invalid_argument("LinearSVM: epochs must be positive");
}

void validate_kernel_parameters(double C, double gamma, double tol,
                                size_t max_passes, bool allow_auto) {
  positive_finite(C, "KernelSVM: C must be positive and finite");
  if (!(allow_auto && gamma == -1.0))
    positive_finite(
        gamma, "KernelSVM: gamma must be positive or the -1 auto sentinel");
  positive_finite(tol, "KernelSVM: tolerance must be positive and finite");
  if (!max_passes)
    throw std::invalid_argument("KernelSVM: max_passes must be positive");
}

// Empty queries are useful in pipelines; training always requires
// rows/features.
void validate_features(const Mat& X, size_t width, bool training) {
  if (training && (X.empty() || X.front().empty()))
    throw std::invalid_argument("SVM: training requires rows and features");
  for (const auto& row : X) {
    if (row.size() != width)
      throw std::invalid_argument("SVM: feature dimension mismatch");
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("SVM: features must be finite");
  }
}

void validate_training(const Mat& X, const Vec& y) {
  validate_features(X, X.empty() ? 0 : X.front().size(), true);
  if (X.size() != y.size())
    throw std::invalid_argument("SVM: feature and label row counts differ");
  bool zero = false, one = false;
  for (double label : y) {
    if (label == 0.0)
      zero = true;
    else if (label == 1.0)
      one = true;
    else
      throw std::invalid_argument("SVM: supported labels are exactly 0 and 1");
  }
  if (!zero || !one)
    throw std::invalid_argument("SVM: training requires both binary classes");
}

// Standard hinge subgradient: d[max(0,1-z)]/dz = -1 if z < 1 else 0.
double hinge_subgrad(double z) { return z < 1.0 ? 1.0 : 0.0; }
}  // namespace

LinearSVM::LinearSVM(double C, double lr, size_t epochs)
    : C_(C), lr_(lr), epochs_(epochs) {
  validate_linear_parameters(C_, lr_, epochs_);
}

void LinearSVM::fit(const Mat& X, const Vec& y) {
  validate_training(X, y);
  size_t n = X.size();
  size_t p = X[0].size();
  size_t dim = p + 1;
  Vec w0(dim, 0.0);

  // Convert labels to {+1, -1} (the hinge form), internal to this fit.
  Vec ybin(n);
  for (size_t i = 0; i < n; ++i) ybin[i] = y[i] == 1.0 ? 1.0 : -1.0;

  auto loss = [&](const Vec& w) -> double {
    double l = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double z = w[p];
      for (size_t j = 0; j < p; ++j) z += w[j] * X[i][j];
      l += std::max(0.0, 1.0 - ybin[i] * z);
    }
    l /= static_cast<double>(n);
    for (size_t j = 0; j < p; ++j) l += 0.5 / C_ * w[j] * w[j];
    return l;
  };
  auto grad = [&](const Vec& w, Vec& g) -> void {
    g.assign(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
      double z = w[p];
      for (size_t j = 0; j < p; ++j) z += w[j] * X[i][j];
      double h = hinge_subgrad(ybin[i] * z);
      g[p] += -(ybin[i] * h);
      for (size_t j = 0; j < p; ++j) g[j] += -(ybin[i] * h) * X[i][j];
    }
    for (size_t d = 0; d < dim; ++d) g[d] /= static_cast<double>(n);
    for (size_t j = 0; j < p; ++j) g[j] += w[j] / C_;
  };

  Vec w = adam(w0, epochs_, lr_, loss, grad).w;
  w_.assign(w.begin(), w.begin() + static_cast<ptrdiff_t>(p));
  b_ = w[p];
}

Vec LinearSVM::decision_function(const Mat& X) const {
  if (w_.empty()) throw std::runtime_error("svm: model not fitted");
  validate_features(X, w_.size(), false);
  Vec out(X.size(), b_);
  size_t p = w_.size();
  for (size_t i = 0; i < X.size(); ++i)
    for (size_t j = 0; j < p; ++j) out[i] += w_[j] * X[i][j];
  return out;
}

Vec LinearSVM::predict(const Mat& X) const {
  Vec s = decision_function(X);
  for (double& v : s) v = v > 0.0 ? 1.0 : 0.0;
  return s;
}

// ---------------------------------------------------------------------------
// KernelSVM: Platt's SMO for the RBF kernel.
//
// Solve the dual of the hard/soft margin problem:
//   max_alpha  sum_i alpha_i - 0.5 sum_i sum_j alpha_i alpha_j y_i y_j
//   K(x_i,x_j) s.t.   0 <= alpha_i <= C,  sum_i alpha_i y_i = 0
// with K(x, y) = exp(-gamma ||x - y||^2).  The decision function is
//   f(x) = sum_i alpha_i y_i K(x_i, x) + b
// The SMO loop picks two alpha to optimise each pass (KKT-violating pair),
// clips the update to [L, H], and recomputes b from the marginal vectors.
// ---------------------------------------------------------------------------

KernelSVM::KernelSVM(double C, double gamma, double tol, size_t max_passes)
    : C_(C), gamma_(gamma), tol_(tol), max_passes_(max_passes) {
  validate_kernel_parameters(C_, gamma_, tol_, max_passes_, true);
}

double KernelSVM::rbf_kernel(const Vec& a, const Vec& b, size_t nf) const {
  double d2 = 0.0;
  for (size_t j = 0; j < nf; ++j) {
    double d = a[j] - b[j];
    d2 += d * d;
  }
  return std::exp(-gamma_ * d2);
}

// Decision value f(x) = sum_i alpha_i y_i K(x_i, x) + b.
static Vec svm_decision(const Mat& Xq, const Mat& Xtr, const Vec& ytr,
                        const Vec& alphas, double b, double gamma) {
  size_t n = Xq.size();
  size_t m = Xtr.size();
  size_t p = Xtr.empty() ? 0 : Xtr[0].size();
  Vec out(n, b);
  for (size_t q = 0; q < n; ++q) {
    for (size_t i = 0; i < m; ++i) {
      if (alphas[i] == 0.0) continue;
      double d2 = 0.0;
      for (size_t j = 0; j < p; ++j) {
        double d = Xq[q][j] - Xtr[i][j];
        d2 += d * d;
      }
      out[q] += alphas[i] * ytr[i] * std::exp(-gamma * d2);
    }
  }
  return out;
}

void KernelSVM::fit(const Mat& X, const Vec& y) {
  validate_training(X, y);
  size_t m = X.size();
  n_features_ = X[0].size();
  if (gamma_ <= 0.0) gamma_ = 1.0 / static_cast<double>(n_features_);
  X_train_ = X;

  // Labels to {+1, -1}.
  y_train_.assign(m, 0.0);
  for (size_t i = 0; i < m; ++i) y_train_[i] = y[i] == 1.0 ? 1.0 : -1.0;

  // Precompute the full kernel matrix once (m is small in lessons).
  Mat K(m, Vec(m, 0.0));
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < m; ++j)
      K[i][j] = rbf_kernel(X[i], X[j], n_features_);

  alpha_.assign(m, 0.0);
  b_ = 0.0;
  size_t passes = 0;

  while (passes < max_passes_) {
    size_t changed = 0;
    for (size_t i = 0; i < m; ++i) {
      // Decision value and error at i.
      double fi = b_;
      for (size_t t = 0; t < m; ++t) fi += alpha_[t] * y_train_[t] * K[t][i];
      double ei = fi - y_train_[i];

      // KKT check: pick an alpha_i that violates optimality (alphas in
      // (0, C)) given tolerance.
      if ((y_train_[i] * ei < -tol_ && alpha_[i] < C_) ||
          (y_train_[i] * ei > tol_ && alpha_[i] > 0.0)) {
        // Pick a different index j (first non-i found, deterministic).
        size_t j = (i + 1) % m;
        double fj = b_;
        for (size_t t = 0; t < m; ++t) fj += alpha_[t] * y_train_[t] * K[t][j];
        double ej = fj - y_train_[j];

        double ai_old = alpha_[i], aj_old = alpha_[j];
        double L = 0.0, H = 0.0;
        if (y_train_[i] != y_train_[j]) {
          L = std::max(0.0, aj_old - ai_old);
          H = std::min(C_, C_ + aj_old - ai_old);
        } else {
          L = std::max(0.0, ai_old + aj_old - C_);
          H = std::min(C_, ai_old + aj_old);
        }
        if (L >= H) continue;

        double eta = 2.0 * K[i][j] - K[i][i] - K[j][j];
        if (eta >= 0.0) continue;

        double aj_new = aj_old - y_train_[j] * (ei - ej) / eta;
        aj_new = std::min(H, std::max(L, aj_new));
        if (std::fabs(aj_new - aj_old) < 1e-5) continue;
        double ai_new = ai_old + y_train_[i] * y_train_[j] * (aj_old - aj_new);

        // Update the threshold b from the new margin conditions.
        double b1 = b_ - ei - y_train_[i] * (ai_new - ai_old) * K[i][i] -
                    y_train_[j] * (aj_new - aj_old) * K[i][j];
        double b2 = b_ - ej - y_train_[i] * (ai_new - ai_old) * K[i][j] -
                    y_train_[j] * (aj_new - aj_old) * K[j][j];
        if (ai_new > 0.0 && ai_new < C_)
          b_ = b1;
        else if (aj_new > 0.0 && aj_new < C_)
          b_ = b2;
        else
          b_ = (b1 + b2) / 2.0;

        alpha_[i] = ai_new;
        alpha_[j] = aj_new;
        ++changed;
      }
    }
    if (changed == 0)
      ++passes;
    else
      passes = 0;
  }
}

Vec KernelSVM::decision_function(const Mat& X) const {
  if (X_train_.empty()) throw std::runtime_error("svm: model not fitted");
  validate_features(X, n_features_, false);
  return svm_decision(X, X_train_, y_train_, alpha_, b_, gamma_);
}

Vec KernelSVM::predict(const Mat& X) const {
  Vec s = decision_function(X);
  for (double& v : s) v = v > 0.0 ? 1.0 : 0.0;
  return s;
}

// Load transactionally: rejecting a damaged archive leaves a fitted model
// intact.
void LinearSVM::load(std::istream& in) {
  std::string tag;
  archive::read(in, tag);
  if (tag != "LinearSVM") throw std::runtime_error("Wrong model type");
  LinearSVM candidate;
  archive::read(in, candidate.C_, candidate.lr_, candidate.epochs_,
                candidate.w_, candidate.b_);
  validate_linear_parameters(candidate.C_, candidate.lr_, candidate.epochs_);
  if (candidate.w_.empty() || !std::isfinite(candidate.b_))
    throw std::runtime_error("LinearSVM: invalid fitted archive");
  for (double weight : candidate.w_)
    if (!std::isfinite(weight))
      throw std::runtime_error("LinearSVM: nonfinite archived coefficient");
  *this = candidate;
}

void KernelSVM::load(std::istream& in) {
  std::string tag;
  archive::read(in, tag);
  if (tag != "KernelSVM") throw std::runtime_error("Wrong model type");
  KernelSVM candidate;
  archive::read(in, candidate.C_, candidate.gamma_, candidate.tol_,
                candidate.max_passes_, candidate.alpha_, candidate.b_,
                candidate.X_train_, candidate.y_train_, candidate.n_features_);
  validate_kernel_parameters(candidate.C_, candidate.gamma_, candidate.tol_,
                             candidate.max_passes_, false);
  validate_features(candidate.X_train_, candidate.n_features_, true);
  const size_t rows = candidate.X_train_.size();
  if (!candidate.n_features_ || candidate.alpha_.size() != rows ||
      candidate.y_train_.size() != rows || !std::isfinite(candidate.b_))
    throw std::runtime_error("KernelSVM: inconsistent archive dimensions");
  bool negative = false, positive = false;
  for (size_t i = 0; i < rows; ++i) {
    const double alpha = candidate.alpha_[i], label = candidate.y_train_[i];
    if (!std::isfinite(alpha) || alpha < -1e-8 || alpha > candidate.C_ + 1e-8)
      throw std::runtime_error("KernelSVM: invalid archived dual coefficient");
    if (label == -1.0)
      negative = true;
    else if (label == 1.0)
      positive = true;
    else
      throw std::runtime_error("KernelSVM: invalid archived internal label");
  }
  if (!negative || !positive)
    throw std::runtime_error("KernelSVM: archive requires both classes");
  *this = candidate;
}

}  // namespace ml
