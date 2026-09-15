// 03_ml_course/04_time_series/05_arima/Model.hpp
// Purpose: a compact ARIMA(p,d,q) forecaster with a constant. The series is
//          differenced d times, an AR(p) part is fitted by least squares and
//          an MA(q) part by conditional sum of squares (gradient descent);
//          forecasts are integrated back to the original scale.
// Inputs:  none (header only; included by Workflow.hpp and the lessons).
// Outputs: none (save()/load() stream orders, coefficients, history and
//          residuals for archives).
// Run target: not an executable; compiled into every forecast_* target.
#pragma once
#include "helper/math/matrix.hpp"
#include "helper/persistence/archive.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace forecast_course {
// ARIMA(p,d,q) with a constant, estimated in two simplified stages:
//   1. AR coefficients by ordinary least squares on lagged differences.
//   2. MA coefficients by conditional sum of squares with plain gradient
//      descent (fixed iteration budget, |theta| clipped below 0.99).
// State kept for one-step forecasting: the raw history (so differencing can be
// redone) and the residual sequence of the differenced series (for the MA
// terms). Invariants after fit(): phi.size() == p, theta.size() == q,
// history.size() >= d + p + q + 3.
class Arima {
public:
  static constexpr size_t css_iterations = 500;   // gradient steps for theta
  static constexpr double css_learning_rate = 0.05;
  static constexpr double theta_bound = 0.99;     // keeps the MA part invertible
  size_t p, d, q;
  double constant = 0;
  ml::Vec phi, theta, history, residuals;
  // Parameters: p_ AR order, d_ differencing order, q_ MA order.
  explicit Arima(size_t p_ = 1, size_t d_ = 1, size_t q_ = 1)
      : p(p_), d(d_), q(q_) {}
  // Apply first differences `times` times: w_t = y_t - y_{t-1}. Each pass
  // shortens the series by one. Throws when fewer than 2 values remain.
  // Complexity O(times * n).
  static ml::Vec difference(const ml::Vec &y, size_t times) {
    ml::Vec w = y;
    for (size_t k = 0; k < times; ++k) {
      if (w.size() < 2)
        throw std::invalid_argument("Series too short to difference");
      ml::Vec next;
      for (size_t i = 1; i < w.size(); ++i)
        next.push_back(w[i] - w[i - 1]);
      w = next;
    }
    return w;
  }
  // Fit on a chronological series: store it, difference, then AR then MA.
  // Throws if the series cannot support the requested order.
  // Complexity O(n * p^2 + css_iterations * n * q^2).
  void fit(const ml::Vec &y) {
    if (y.size() < d + p + q + 3)
      throw std::invalid_argument("Series too short for ARIMA order");
    history = y;
    const ml::Vec w = difference(y, d);
    fit_autoregression(w);
    fit_moving_average(w);
  }
  // One-step forecast on the original scale: forecast of the d-th difference
  // plus the integration offset. O(d * n + p + q).
  double next() const {
    if (history.size() < d + 1)
      throw std::runtime_error("ARIMA not fitted");
    return forecast_difference() + integration_offset();
  }
  // Reveal the realised value: record the residual of the differenced series
  // (needed by the MA terms) and extend the history. Coefficients stay fixed.
  void observe(double actual) {
    const double predicted = forecast_difference();
    const double actual_difference = actual - integration_offset();
    residuals.push_back(actual_difference - predicted);
    history.push_back(actual);
  }
  // Archive format: type tag, then orders, constant, phi, theta, history,
  // residuals, so next() can be reproduced exactly after reload.
  void save(std::ostream &out) const {
    ml::archive::write(out, std::string("Arima_V1"));
    ml::archive::write(out, p, d, q, constant, phi, theta, history, residuals);
  }
  // Inverse of save(); rejects a file written by a different model type.
  void load(std::istream &in) {
    std::string model_type;
    ml::archive::read(in, model_type);
    if (model_type != std::string("Arima_V1"))
      throw std::runtime_error("Wrong model type/version");
    ml::archive::read(in, p, d, q, constant, phi, theta, history, residuals);
  }

private:
  // Sum of the last value of each partially differenced series: this is
  // what must be added to a forecast of the d-th difference to recover y.
  // (For d = 1 it is simply the last observed value.) O(d * n).
  double integration_offset() const {
    double offset = 0;
    ml::Vec level = history;
    for (size_t k = 0; k < d; ++k) {
      offset += level.back();
      level = difference(level, 1);
    }
    return offset;
  }
  // Forecast of the next d-th difference: constant + sum phi_i w_{t-i}
  // + sum theta_j e_{t-j}. Missing lags (short history) contribute 0.
  double forecast_difference() const {
    const ml::Vec w = difference(history, d);
    double value = constant;
    for (size_t i = 0; i < p; ++i)
      value += i < w.size() ? phi[i] * w[w.size() - 1 - i] : 0;
    for (size_t j = 0; j < q; ++j)
      value += j < residuals.size()
                   ? theta[j] * residuals[residuals.size() - 1 - j]
                   : 0;
    return value;
  }
  // Solve the (p+1)x(p+1) normal equations by Gaussian elimination.
  // Partial pivoting; throws if a pivot is (numerically) zero. O(n^3).
  static ml::Vec solve(ml::Mat A, ml::Vec b) {
    const size_t n = b.size();
    for (size_t c = 0; c < n; ++c) {
      size_t pivot = c;
      for (size_t r = c + 1; r < n; ++r)
        if (std::abs(A[r][c]) > std::abs(A[pivot][c]))
          pivot = r;
      std::swap(A[c], A[pivot]);
      std::swap(b[c], b[pivot]);
      if (std::abs(A[c][c]) < 1e-12)
        throw std::runtime_error("Singular AR normal equations");
      // Gauss-Jordan: eliminate column c from every other row.
      for (size_t r = 0; r < n; ++r) {
        if (r == c)
          continue;
        const double f = A[r][c] / A[c][c];
        for (size_t k = c; k < n; ++k)
          A[r][k] -= f * A[c][k];
        b[r] -= f * b[c];
      }
    }
    ml::Vec x(n);
    for (size_t i = 0; i < n; ++i)
      x[i] = b[i] / A[i][i];
    return x;
  }
  // Stage 1: OLS of w_t on [1, w_{t-1}, ..., w_{t-p}] via accumulated normal
  // equations A beta = b. beta[0] is the constant, the rest are phi.
  void fit_autoregression(const ml::Vec &w) {
    const size_t n = p + 1;
    ml::Mat A = ml::make_mat(n, n);
    ml::Vec b = ml::make_vec(n);
    for (size_t t = p; t < w.size(); ++t) {
      ml::Vec row(n, 1.0);
      for (size_t i = 0; i < p; ++i)
        row[i + 1] = w[t - 1 - i];
      for (size_t r = 0; r < n; ++r) {
        b[r] += row[r] * w[t];
        for (size_t c = 0; c < n; ++c)
          A[r][c] += row[r] * row[c];
      }
    }
    const ml::Vec beta = solve(A, b);
    constant = beta[0];
    phi.assign(beta.begin() + 1, beta.end());
  }
  // Residuals given the AR part and the current theta, plus the exact
  // recursive derivative of each residual with respect to every theta_j.
  // Residuals before index p are fixed at 0 (the "conditional" in CSS).
  // gradient may be null when only the residuals are needed. O(n * q^2).
  ml::Vec css_residuals(const ml::Vec &w, ml::Mat *gradient) const {
    ml::Vec e(w.size(), 0.0);
    ml::Mat de(w.size(), ml::Vec(q, 0.0));
    for (size_t t = p; t < w.size(); ++t) {
      double fitted = constant;
      for (size_t i = 0; i < p; ++i)
        fitted += phi[i] * w[t - 1 - i];
      for (size_t j = 0; j < q; ++j)
        fitted += t > j ? theta[j] * e[t - 1 - j] : 0;
      e[t] = w[t] - fitted;
      // d e_t / d theta_j = -e_{t-1-j} - sum_k theta_k d e_{t-1-k} / d theta_j
      for (size_t j = 0; j < q; ++j) {
        double g = t > j ? -e[t - 1 - j] : 0;
        for (size_t k = 0; k < q; ++k)
          g -= t > k ? theta[k] * de[t - 1 - k][j] : 0;
        de[t][j] = g;
      }
    }
    if (gradient)
      *gradient = de;
    return e;
  }
  // Stage 2: minimise the mean squared residual over theta by gradient descent
  // with the step scaled by the residual variance (a crude normalisation) and
  // theta clipped to (-0.99, 0.99). Leaves `residuals` ready for forecasting.
  void fit_moving_average(const ml::Vec &w) {
    theta.assign(q, 0.0);
    residuals = css_residuals(w, nullptr);
    if (!q)
      return;
    const size_t n = w.size() - p;
    for (size_t iteration = 0; iteration < css_iterations; ++iteration) {
      ml::Mat de;
      const ml::Vec e = css_residuals(w, &de);
      double variance = 1e-12;
      for (size_t t = p; t < w.size(); ++t)
        variance += e[t] * e[t] / n;
      for (size_t j = 0; j < q; ++j) {
        double g = 0;
        for (size_t t = p; t < w.size(); ++t)
          g += 2 * e[t] * de[t][j] / n;
        theta[j] = std::clamp(theta[j] - css_learning_rate * g / variance,
                              -theta_bound, theta_bound);
      }
    }
    residuals = css_residuals(w, nullptr);
  }
};
} // namespace forecast_course
