// 03_ml_course/04_time_series/01_forecasting_baselines/Model.hpp
// Purpose: the seasonal-naive forecaster, the simplest sensible baseline for a
//          seasonal series: "next month equals the same month one season ago".
//          With period = 1 it degenerates to the last-value (naive) forecast.
// Inputs:  none (header only; included by Workflow.hpp and the lessons).
// Outputs: none (save()/load() stream the period and history for archives).
// Run target: not an executable; compiled into every forecast_* target.
#pragma once
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <stdexcept>
namespace forecast_course {
// Seasonal naive: next() = history[n - period]. Keeps the full history so
// observe() can extend it and next() always looks back exactly one season.
// Invariant after fit(): history.size() >= period >= 1.
class SeasonalNaive {
public:
  size_t period;
  ml::Vec history;
  // Parameters: season - lag in observations (12 for monthly data; 1 = naive).
  explicit SeasonalNaive(size_t season = 12) : period(season) {}
  // "Fitting" only stores the series; throws if shorter than one season.
  void fit(const ml::Vec &y) {
    if (!period || y.size() < period)
      throw std::invalid_argument("History shorter than season");
    history = y;
  }
  // Forecast for the next step: the value observed one season earlier. O(1).
  double next() const {
    if (history.size() < period)
      throw std::runtime_error("Baseline not fitted");
    return history[history.size() - period];
  }
  // Reveal the realised value so the next forecast moves one step forward.
  void observe(double actual) { history.push_back(actual); }
  // Archive format: type tag, then period and history.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("SeasonalNaive_V1"));
    ml::archive::write(out, period, history);
  }
  // Inverse of save(); rejects a file written by a different model type.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("SeasonalNaive_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, period, history);
  }
};
} // namespace forecast_course
