#pragma once
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <stdexcept>
namespace forecast_course {
// Additive Holt-Winters (triple exponential smoothing).
// State after t observations: level, trend and one seasonal offset per
// position of the season.  next() = level + trend + season[t mod period].
class HoltWinters {
public:
  double alpha, beta, gamma;
  size_t period;
  double level = 0, trend = 0;
  ml::Vec seasonal;
  size_t observed = 0;
  explicit HoltWinters(double alpha_ = 0.5, double beta_ = 0.1,
                       double gamma_ = 0.3, size_t season = 12)
      : alpha(alpha_), beta(beta_), gamma(gamma_), period(season) {
    if (!period)
      throw std::invalid_argument("Season length must be positive");
    for (double w : {alpha, beta, gamma})
      if (w < 0 || w > 1)
        throw std::invalid_argument("Smoothing weights must lie in [0,1]");
  }
  // Initial state from the first two seasons: level = first-season mean,
  // trend = per-step change between the season means, seasonal offsets =
  // average deviation from the respective season mean.
  void initialize(const ml::Vec &y) {
    if (y.size() < 2 * period)
      throw std::invalid_argument("Need at least two full seasons");
    double first = 0, second = 0;
    for (size_t i = 0; i < period; ++i) {
      first += y[i] / period;
      second += y[i + period] / period;
    }
    level = first;
    trend = (second - first) / period;
    seasonal.assign(period, 0);
    for (size_t i = 0; i < period; ++i)
      seasonal[i] = ((y[i] - first) + (y[i + period] - second)) / 2;
    observed = 0;
  }
  // Initialise, then run the recursions over the whole series so the state
  // describes the end of the training window.
  void fit(const ml::Vec &y) {
    initialize(y);
    for (double v : y)
      observe(v);
  }
  double next() const {
    if (seasonal.size() != period)
      throw std::runtime_error("Holt-Winters not fitted");
    return level + trend + seasonal[observed % period];
  }
  void observe(double actual) {
    if (seasonal.size() != period)
      throw std::runtime_error("Holt-Winters not fitted");
    const size_t s = observed % period;
    const double previous_level = level;
    level = alpha * (actual - seasonal[s]) +
            (1 - alpha) * (previous_level + trend);
    trend = beta * (level - previous_level) + (1 - beta) * trend;
    seasonal[s] = gamma * (actual - level) + (1 - gamma) * seasonal[s];
    ++observed;
  }
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("HoltWinters_V1"));
    ml::archive::write(out, alpha, beta, gamma, period, level, trend, seasonal,
                       observed);
  }
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("HoltWinters_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, alpha, beta, gamma, period, level, trend, seasonal,
                      observed);
  }
};
} // namespace forecast_course
