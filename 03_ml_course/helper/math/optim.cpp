#include "helper/math/optim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace ml {

// ---------------------------------------------------------------------------
// Seeded random number generation
//
// All randomness in the ml toolkit flows through one mt19937 engine with a
// seed the lesson chooses explicitly (ml::seed_rng).  Because every stochastic
// algorithm draws from this single stream in a fixed order, lesson output is
// byte-for-byte reproducible.
// ---------------------------------------------------------------------------
namespace {
std::mt19937 g_rng(42);
std::normal_distribution<double> g_normal(0.0, 1.0);
std::uniform_real_distribution<double> g_unit(0.0, 1.0);
}  // namespace

// Reset the engine and clear cached distribution state (Box-Muller can
// otherwise leave internal state between runs).
void seed_rng(std::uint32_t seed) {
  g_rng.seed(seed);
  g_normal.reset();
  g_unit.reset();
}

// Uniform integer in [lo, hi] (for e.g. mini-batch row picks).
std::uint32_t next_rng(std::uint32_t lo, std::uint32_t hi) {
  std::uniform_int_distribution<std::uint32_t> d(lo, hi);
  return d(g_rng);
}

// Uniform double in [0, 1).
double next_unit() { return g_unit(g_rng); }

// Normal deviate via Box-Muller transform on the shared engine.
double next_normal(double mu, double sigma) {
  if (sigma <= 0.0) return mu;
  return mu + sigma * g_normal(g_rng);
}

// In-place Fisher-Yates shuffle of index vector (used for train/test folds).
void shuffle_indices(std::vector<size_t>& idx) {
  std::shuffle(idx.begin(), idx.end(), g_rng);
}

// ---------------------------------------------------------------------------
// Batch gradient descent with a fixed learning rate.
//
// Each epoch computes the full gradient and takes the step w -= lr * grad.
// The loss history (loss[0] = value before the first step) lets a lesson show
// the descent curve.  Stops early once two consecutive losses agree to 1e-9
// (and at least 5 epochs ran) so the reported iteration count stays short.
// ---------------------------------------------------------------------------
OptimResult gradient_descent(const Vec& w0, size_t epochs, double lr,
                             const LossFn& loss, const GradFn& grad) {
  OptimResult r;
  r.w = w0;
  r.loss.push_back(loss(w0));
  long double last = r.loss.back();
  for (size_t ep = 1; ep <= epochs; ++ep) {
    Vec g;
    grad(r.w, g);
    for (size_t j = 0; j < r.w.size(); ++j) r.w[j] -= lr * g[j];
    double cur = loss(r.w);
    r.loss.push_back(cur);
    if (std::fabs(last - cur) < 1e-9 && ep >= 5) {
      // converged; stop the loop to keep output short
      break;
    }
    last = static_cast<long double>(cur);
  }
  return r;
}

// ---------------------------------------------------------------------------
// Adam (Kingma & Ba, 2014).
//
// Maintains per-parameter first moment m (mean of gradients, beta1-decayed)
// and second raw moment v (mean of squared gradients, beta2-decayed).
// Bias correction divides by (1 - beta^t) because m and v start at zero and
// would otherwise under-estimate early gradients.  The effective step is
// lr * mhat / (sqrt(vhat) + eps).  This is the same update the lesson's
// sklearn.stochastic equivalent implements.
// ---------------------------------------------------------------------------
OptimResult adam(const Vec& w0, size_t epochs, double lr, const LossFn& loss,
                 const GradFn& grad, double beta1, double beta2, double eps) {
  OptimResult r;
  size_t n = w0.size();
  r.w = w0;
  Vec m(n, 0.0), v(n, 0.0);
  r.loss.push_back(loss(w0));
  long double last = r.loss.back();
  for (size_t t = 1; t <= epochs; ++t) {
    Vec g;
    grad(r.w, g);
    for (size_t j = 0; j < n; ++j) {
      m[j] = beta1 * m[j] + (1.0 - beta1) * g[j];
      v[j] = beta2 * v[j] + (1.0 - beta2) * g[j] * g[j];
      double mhat = m[j] / (1.0 - std::pow(beta1, static_cast<double>(t)));
      double vhat = v[j] / (1.0 - std::pow(beta2, static_cast<double>(t)));
      r.w[j] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
    double cur = loss(r.w);
    r.loss.push_back(cur);
    if (std::fabs(last - cur) < 1e-9 && t >= 5) break;
    last = static_cast<long double>(cur);
  }
  return r;
}

// ---------------------------------------------------------------------------
// Gradient sanity check: numerical central difference vs. the analytic
// gradient.  Central differences are ~h^2 accurate, so with h = 1e-6 a correct
// analytic gradient yields a mismatch near 1e-9 (well below any lesson
// display threshold).  Returns the worst |fd - analytic| over all parameters.
// ---------------------------------------------------------------------------
double gradient_check(const LossFn& loss, const GradFn& grad, const Vec& w,
                      double h) {
  double worst = 0.0;
  Vec g;
  grad(w, g);
  for (size_t j = 0; j < w.size(); ++j) {
    Vec wp = w, wm = w;
    wp[j] += h;
    wm[j] -= h;
    double fd = (loss(wp) - loss(wm)) / (2.0 * h);
    worst = std::max(worst, std::fabs(fd - g[j]));
  }
  return worst;
}

}  // namespace ml