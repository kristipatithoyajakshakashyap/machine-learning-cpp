// 03_ml_course/04_time_series/02_lag_feature_regression/Model.hpp
// Purpose: turn forecasting into supervised regression. Each month t gets six
//          features built only from already observed values (lags 1, 2, 12,
//          the time index and a sin/cos seasonal pair), standardized and fed
//          to a ridge regression (ElasticNet with l1 ratio 0).
// Inputs:  none (header only; included by Workflow.hpp and the lessons).
// Outputs: none (save()/load() stream the regression, preprocessor, history).
// Run target: not an executable; compiled into every forecast_* target.
#pragma once
#include "01_supervised/01_regression/07_elastic_net/Model.hpp"
#include "helper/preprocessing/preprocessor.hpp"
namespace forecast_course {
// Feature row for predicting history[t] from the past.
// Parameters: history - series; t - index to predict, 12 <= t <= size().
// Returns {lag_1, lag_2, lag_12, month_index, season_sin, season_cos}.
// t == history.size() builds the features for the not-yet-observed next step.
// Throws if fewer than twelve earlier months exist. O(1).
inline ml::Vec features(const ml::Vec &history, size_t t) {
  if (t < 12 || t > history.size())
    throw std::invalid_argument("Need twelve observed historical months");
  constexpr double pi = 3.14159265358979323846;
  return {history[t - 1],
          history[t - 2],
          history[t - 12],
          double(t),
          std::sin(2 * pi * t / 12),
          std::cos(2 * pi * t / 12)};
}
// Ridge regression on lag features. The preprocessor (mean/scale) and the
// coefficients are learned once in fit(); observe() only extends the history,
// so held-out forecasts use frozen coefficients on fresh lag values.
// Invariant after fit(): history.size() >= 24.
class LagRegression {
public:
  course::ElasticNet regression;
  ml::Preprocessor preprocessing;
  ml::Vec history;
  // Parameters: alpha - ridge penalty (0 = ordinary least squares).
  explicit LagRegression(double alpha = 0.1) : regression(alpha, 0) {}
  // Build the supervised table for t = 12..n-1, fit scaler then regression.
  // Throws if fewer than two years are supplied. Complexity O(n) rows x
  // ElasticNet coordinate-descent cost.
  void fit(const ml::Vec &y) {
    if (y.size() < 24)
      throw std::invalid_argument("Need two years of training history");
    history = y;
    ml::Mat X;
    ml::Vec target;
    for (size_t t = 12; t < y.size(); ++t) {
      X.push_back(features(y, t));
      target.push_back(y[t]);
    }
    preprocessing.fit(X);
    regression.fit(preprocessing.transform(X), target);
  }
  // One-step forecast from the current end of history. O(p).
  double next() const {
    return regression.predict(
        preprocessing.transform({features(history, history.size())}))[0];
  }
  // Reveal the realised value; coefficients are NOT refitted.
  void observe(double actual) { history.push_back(actual); }
  // Archive format: type tag, regression, preprocessor, history.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("LagRegression_V1"));
    ml::archive::write(out, regression, preprocessing, history);
  }
  // Inverse of save(); rejects a file written by a different model type.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("LagRegression_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, regression, preprocessing, history);
  }
};
} // namespace forecast_course
