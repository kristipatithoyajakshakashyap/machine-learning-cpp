#ifndef HELPER_MATH_OPTIM_HPP
#define HELPER_MATH_OPTIM_HPP

// ===========================================================================
// helper/math/optim.hpp  (+ the matching helper/math/optim.cpp)
// ---------------------------------------------------------------------------
// Deterministic optimisers used by the linear, logistic and neural models,
// plus the shared seeded RNG (ml::seed_rng / ml::next_rng / ml::next_normal /
// ml::shuffle_indices).  Helper/eval/cross_validation.cpp and the random
// forests / MLP modules depend on this file.
//
// Full-batch versions are pure functions of (w0, epochs, lr, loss, grad); the
// stochastic variants use the ml::seed_rng() stream so every lesson replays
// the same random decisions.  sklearn equivalents are the SGD/Adam lore in
// sklearn.linear_model.sgd_fast and torch-style Adam updates.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <vector>

#include "helper/math/matrix.hpp"

namespace ml {

// Loss(w) -> scalar to minimise; Grad(w, g) sets g = dLoss/dw.
using LossFn = std::function<double(const Vec&)>;
using GradFn = std::function<void(const Vec&, Vec&)>;

// Result of an optimisation run: final parameters plus the loss recorded
// before each step (loss[0] = initial loss).  Lessons print the last value
// and sometimes the whole curve.
struct OptimResult {
  Vec w;                     // final parameters
  std::vector<double> loss;  // loss per epoch (loss[0] = before first step)
};

// ---------------------------------------------------------------------------
// Seeded RNG (mt19937).
// ---------------------------------------------------------------------------
// Resets the engine (and cached distribution state) to a known seed.
void seed_rng(std::uint32_t seed);
// Uniform integer in [lo, hi].
std::uint32_t next_rng(std::uint32_t lo, std::uint32_t hi);
// Uniform double in [0, 1).
double next_unit();
// Normal deviate with the given mean/std (Box-Muller).
double next_normal(double mu = 0.0, double sigma = 1.0);
// Deterministic in-place Fisher-Yates shuffle of an index vector.
void shuffle_indices(std::vector<size_t>& idx);

// ---------------------------------------------------------------------------
// Optimisers.
// ---------------------------------------------------------------------------
// Full-batch gradient descent with a fixed learning rate: w -= lr * grad.
// Records the loss before each step; stops early once the loss change stays
// under 1e-9 (with a 5-epoch minimum so the trace stays meaningful).
OptimResult gradient_descent(const Vec& w0, size_t epochs, double lr,
                             const LossFn& loss, const GradFn& grad);

// Adam (Kingma & Ba 2014): momentum m and RMS v per parameter, both bias
// corrected by (1 - beta^t).  Effective step: lr*mhat/(sqrt(vhat)+eps).
OptimResult adam(const Vec& w0, size_t epochs, double lr, const LossFn& loss,
                 const GradFn& grad, double beta1 = 0.9, double beta2 = 0.999,
                 double eps = 1e-8);

// Central-difference gradient check.  Returns the largest |analytic - fd|
// over all parameters; a correct analytic gradient yields ~1e-9.
double gradient_check(const LossFn& loss, const GradFn& grad, const Vec& w,
                      double h = 1e-6);

}  // namespace ml

#endif  // ML_OPTIM_HPP