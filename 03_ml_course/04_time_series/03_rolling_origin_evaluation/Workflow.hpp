// 03_ml_course/04_time_series/03_rolling_origin_evaluation/Workflow.hpp
// Purpose: the one end-to-end forecasting pipeline shared by every time-series
//          module (baselines, lag regression, rolling origin, Holt-Winters,
//          ARIMA): read the series, EDA, rolling-origin candidate selection,
//          final fit, save + reload check, one-step held-out evaluation,
//          metrics, figures, manifest and report. Also the --model inference
//          path used by every forecast_*_predict target.
// Inputs:  DATA_DIR/air_passengers.csv (144 monthly rows, 1949-1960) passed by
//          the caller; command line flags --quick (write under quick/ instead
//          of full/) and --model <file> (reload and emit one forecast).
// Outputs: under <root>/full/ or <root>/quick/: eda/, validation/
//          candidate_scores.csv, model/<kind>_model.txt (+ lag_model.txt and
//          seasonal_model.txt for modules 01-03), model/feature_schema.json,
//          inference/reload_verification.json, evaluation/{predictions.csv,
//          metrics.json,figures/forecast.svg,figures/residuals.svg},
//          eda/autocorrelation.csv, run_manifest.json, report.md,
//          execution.log. With --model: <root>/next_forecast.csv only.
// Run target: not an executable; included by every 04_end_to_end.cpp,
//             predict.cpp and tests/model_test.cpp in 04_time_series.
#pragma once
#include "04_time_series/01_forecasting_baselines/Model.hpp"
#include "04_time_series/02_lag_feature_regression/Model.hpp"
#include "04_time_series/04_exponential_smoothing/Model.hpp"
#include "04_time_series/05_arima/Model.hpp"
#include "helper/math/metrics.hpp"
#include "helper/reporting/artifacts.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
namespace forecast_course {
// Which model family a module reports; selects the candidate grid, the model
// file name and the report wording via spec_for().
enum class Forecaster { SeasonalNaive, LagRegression, HoltWinters, Arima };
// Read the passenger column of a two-column CSV (month,passengers) with a
// header line. Returns the series in file order. Throws if the file is
// missing, a row has no comma, or fewer than 60 rows (five seasons) exist.
// Complexity O(rows).
inline ml::Vec read_passengers(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read passengers data");
  std::string line;
  std::getline(in, line);  // header
  ml::Vec y;
  while (std::getline(in, line)) {
    auto comma = line.find(',');
    if (comma == std::string::npos)
      throw std::runtime_error("Malformed passenger row");
    y.push_back(std::stod(line.substr(comma + 1)));
  }
  if (y.size() < 60)
    throw std::runtime_error("Insufficient chronological data");
  return y;
}
// Uniform one-step forecaster interface shared by every module's model.
// Contract: fit(y) learns from a chronological history; next() forecasts the
// value after the last observed one; observe(actual) appends the realised
// value and updates internal state (never coefficients); save/load persist
// everything next() needs.
class Forecast {
public:
  virtual ~Forecast() = default;
  virtual void fit(const ml::Vec &y) = 0;
  virtual double next() const = 0;
  virtual void observe(double actual) = 0;
  virtual void save_file(const std::string &path) const = 0;
  virtual void load_file(const std::string &path) = 0;
};
// Wraps a concrete model type M (which has the same member names but no
// virtual base) so the workflow can hold any model behind a Forecast pointer.
template <class M> class Adapter : public Forecast {
public:
  M model;
  explicit Adapter(M m) : model(std::move(m)) {}
  void fit(const ml::Vec &y) override { model.fit(y); }
  double next() const override { return model.next(); }
  void observe(double actual) override { model.observe(actual); }
  void save_file(const std::string &path) const override {
    ml::archive::save_file(path, model);
  }
  void load_file(const std::string &path) override {
    ml::archive::load_file(path, model);
  }
};
// Convenience factory: adapt(SeasonalNaive()) -> unique_ptr<Forecast>.
template <class M> std::unique_ptr<Forecast> adapt(M m) {
  return std::make_unique<Adapter<M>>(std::move(m));
}
// Candidate grids are indexed 0..count-1; the factory decodes the index.
// name          - human-readable family name used in reports;
// model_file    - file under model/ that holds the final model;
// columns       - CSV header for the hyper-parameters (e.g. "alpha,beta,gamma");
// phrase_prefix - prefix of the selection sentence in report.md;
// candidate_count - grid size; values(i) renders candidate i's parameters as
// CSV; candidate(i) builds the model scored in rolling-origin selection;
// final_model(i) builds the model that is fitted, saved and evaluated.
struct KindSpec {
  std::string name, model_file, columns, phrase_prefix;
  size_t candidate_count;
  std::string (*values)(size_t);
  std::unique_ptr<Forecast> (*candidate)(size_t);
  std::unique_ptr<Forecast> (*final_model)(size_t);
};
// Lag-regression grid: ridge penalty alpha in {0, 0.1, 10}.
inline double lag_alpha(size_t i) { return ml::Vec{0, 0.1, 10}[i]; }
inline std::string lag_values(size_t i) {
  std::ostringstream s;
  s << lag_alpha(i);
  return s.str();
}
inline std::unique_ptr<Forecast> make_lag(size_t i) {
  return adapt(LagRegression(lag_alpha(i)));
}
// Seasonal naive has no hyper-parameters; the index is ignored.
inline std::unique_ptr<Forecast> make_seasonal(size_t) {
  return adapt(SeasonalNaive());
}
// Holt-Winters grid: 3 x 3 x 3 = 27 combinations of (alpha, beta, gamma), each
// weight in {0.2, 0.5, 0.8}; index i is decoded base 3 (i/9, i/3, i).
inline double hw_weight(size_t k) { return ml::Vec{0.2, 0.5, 0.8}[k % 3]; }
inline std::string hw_values(size_t i) {
  std::ostringstream s;
  s << hw_weight(i / 9) << ',' << hw_weight(i / 3) << ',' << hw_weight(i);
  return s.str();
}
inline std::unique_ptr<Forecast> make_hw(size_t i) {
  return adapt(HoltWinters(hw_weight(i / 9), hw_weight(i / 3), hw_weight(i)));
}
// ARIMA grid: p in {1, 2}, d fixed at 1, q in {1, 2} (4 combinations).
inline std::string arima_values(size_t i) {
  return std::to_string(1 + i / 2) + ",1," + std::to_string(1 + i % 2);
}
inline std::unique_ptr<Forecast> make_arima(size_t i) {
  return adapt(Arima(1 + i / 2, 1, 1 + i % 2));
}
// Map a Forecaster kind to its grid and naming. Throws on an unknown enum.
inline KindSpec spec_for(Forecaster kind) {
  switch (kind) {
  case Forecaster::SeasonalNaive:
    // The baseline module tunes the lag regression as reference and keeps
    // seasonal naive as the reported forecaster.
    return {"seasonal naive", "seasonal_model.txt", "alpha", "Ridge alpha ",
            3,                lag_values,           make_lag, make_seasonal};
  case Forecaster::LagRegression:
    return {"lag regression", "lag_model.txt", "alpha", "Ridge alpha ", 3,
            lag_values,       make_lag,        make_lag};
  case Forecaster::HoltWinters:
    return {"Holt-Winters", "holt_winters_model.txt", "alpha,beta,gamma",
            "Holt-Winters ", 27, hw_values, make_hw, make_hw};
  case Forecaster::Arima:
    return {"ARIMA", "arima_model.txt", "p,d,q", "ARIMA ", 4,
            arima_values, make_arima, make_arima};
  }
  throw std::invalid_argument("Unknown forecaster kind");
}
// Split "a,b,c" into {"a","b","c"} (no quoting; used for the small grids).
inline std::vector<std::string> split_csv(const std::string &text) {
  std::vector<std::string> cells;
  std::stringstream s(text);
  std::string cell;
  while (std::getline(s, cell, ','))
    cells.push_back(cell);
  return cells;
}
// "alpha":0.1  or  "alpha":0.2,"beta":0.5,"gamma":0.8
// JSON fragment for run_manifest.json describing candidate i.
inline std::string manifest_fields(const KindSpec &spec, size_t i) {
  auto keys = split_csv(spec.columns), vals = split_csv(spec.values(i));
  std::string out;
  for (size_t k = 0; k < keys.size(); ++k)
    out += (k ? "," : "") + ml::json_quote(keys[k]) + ":" + vals[k];
  return out;
}
// "Ridge alpha 0.1"  or  "Holt-Winters alpha=0.2, beta=0.5, gamma=0.8"
// Human-readable sentence fragment for report.md describing candidate i.
inline std::string selection_phrase(const KindSpec &spec, size_t i) {
  auto keys = split_csv(spec.columns), vals = split_csv(spec.values(i));
  if (keys.size() == 1)
    return spec.phrase_prefix + vals[0];
  std::string out = spec.phrase_prefix;
  for (size_t k = 0; k < keys.size(); ++k)
    out += (k ? ", " : "") + keys[k] + "=" + vals[k];
  return out;
}
// The whole experiment for one model family.
// Parameters: root - RUN_OUTPUT_DIR of the calling target; path - CSV of the
//             series; kind - model family; argc/argv - flags (see file header).
// Returns 0 on success, 1 after printing the exception message to stderr.
// Stages: parse flags -> (--model shortcut) -> read + EDA -> rolling-origin
// selection -> final fit -> save/reload check -> one-step holdout ->
// metrics/figures -> ACF -> schema/manifest/report.
// Complexity: candidate_count x folds x fit cost, plus one final fit.
inline int workflow(const std::string &root, const std::string &path,
                    Forecaster kind, int argc, char **argv) {
  try {
    const KindSpec spec = spec_for(kind);
    // Modules 01-03 ("legacy") always save both the lag model and the seasonal
    // baseline and describe six lag features in the schema.
    const bool legacy = kind == Forecaster::SeasonalNaive ||
                        kind == Forecaster::LagRegression;
    bool quick = false;
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--quick")
        quick = true;
      else if (arg == "--model" && i + 1 < argc)
        model_path = argv[++i];
      else
        throw std::invalid_argument(
            "Use --quick or --model MODEL_FILE for one-step inference");
    }
    // Inference shortcut (predict targets): reload the saved model, which
    // carries its own history, and forecast the next month. Candidate index 0
    // only picks the type; load_file overwrites every parameter.
    if (!model_path.empty()) {
      auto m = spec.final_model(0);
      m->load_file(model_path);
      ml::Artifacts(root, ".")
          .write("next_forecast.csv",
                 "prediction\n" + std::to_string(m->next()) + "\n");
      return 0;
    }
    // Stage 1 - read the series; --quick only changes the output subfolder so
    // ctest runs never overwrite a full/ run.
    ml::Artifacts a(root, quick ? "quick" : "full");
    auto y = read_passengers(path);
    // Chronological split: first 80% is training, the rest is the holdout.
    const size_t cut = y.size() * 4 / 5;
    ml::Vec training(y.begin(), y.begin() + cut);
    // Stage 2 - EDA on the lag-feature view of the training window (eda/).
    ml::Dataset d;
    d.target_name = "passengers_thousands";
    d.feature_names = {"lag_1",       "lag_2",      "lag_12",
                       "month_index", "season_sin", "season_cos"};
    for (size_t t = 12; t < cut; ++t) {
      d.X.push_back(features(y, t));
      d.y.push_back(y[t]);
    }
    ml::eda(d, a);
    // Stage 3 - rolling-origin selection: for each candidate, expanding
    // windows starting at month 48 and growing by 12; each fold forecasts the
    // next 12 months one step at a time and is scored by RMSE. The candidate
    // with the lowest mean fold RMSE wins. No holdout month is ever used.
    std::ostringstream cv;
    cv << spec.columns << ",origin,validation_end,rmse\n";
    double best = 1e300;
    size_t chosen = 0;
    for (size_t candidate = 0; candidate < spec.candidate_count; ++candidate) {
      double sum = 0;
      size_t folds = 0;
      for (size_t origin = 48; origin + 12 <= cut; origin += 12) {
        auto model = spec.candidate(candidate);
        model->fit(ml::Vec(y.begin(), y.begin() + origin));
        ml::Vec actual, pred;
        for (size_t t = origin; t < origin + 12; ++t) {
          pred.push_back(model->next());
          actual.push_back(y[t]);
          model->observe(y[t]);
        }
        double score = ml::rmse(actual, pred);
        sum += score;
        ++folds;
        cv << spec.values(candidate) << ',' << origin << ',' << origin + 12
           << ',' << score << '\n';
      }
      if (folds && sum / folds < best) {
        best = sum / folds;
        chosen = candidate;
      }
    }
    a.write("validation/candidate_scores.csv", cv.str());
    // Stage 4 - final fit on the full training window, plus the two reference
    // baselines (seasonal naive, period 12; last value, period 1).
    auto model = spec.final_model(chosen);
    model->fit(training);
    SeasonalNaive seasonal;
    seasonal.fit(training);
    SeasonalNaive last(1);
    last.fit(training);
    if (legacy) {
      // Modules 01-03 always persist both the tuned lag model and the
      // seasonal baseline; the chosen kind is one of the two.
      auto lag = spec.candidate(chosen);
      lag->fit(training);
      lag->save_file(a.path("model/lag_model.txt"));
      ml::archive::save_file(a.path("model/seasonal_model.txt"), seasonal);
    }
    // Stage 5 - snapshot and reload check: the reloaded model must give the
    // same first held-out forecast to 1e-12.
    model->save_file(a.path("model/" + spec.model_file));
    auto reloaded = spec.final_model(chosen);
    reloaded->load_file(a.path("model/" + spec.model_file));
    if (std::abs(model->next() - reloaded->next()) > 1e-12)
      throw std::runtime_error("Forecast reload failed");
    a.write("inference/reload_verification.json", "{\"passed\":true}\n");
    // Stage 6 - one-step holdout: forecast month t, then reveal y[t] to every
    // model via observe() before forecasting t+1. Coefficients stay frozen.
    ml::Vec actual, pred, naive, last_pred, time, residual;
    std::ostringstream rows;
    rows << "month_index,split,actual,prediction,seasonal_naive,last_value\n";
    for (size_t t = 0; t < cut; ++t)
      rows << t << ",train," << y[t] << ",,,\n";
    for (size_t t = cut; t < y.size(); ++t) {
      double season = seasonal.next(), estimate = model->next();
      actual.push_back(y[t]);
      pred.push_back(estimate);
      naive.push_back(season);
      last_pred.push_back(last.next());
      time.push_back(double(t));
      residual.push_back(y[t] - estimate);
      rows << t << ",test," << y[t] << ',' << estimate << ',' << season << ','
           << last.next() << '\n';
      model->observe(y[t]);
      seasonal.observe(y[t]);
      last.observe(y[t]);
    }
    a.write("evaluation/predictions.csv", rows.str());
    // Stage 7 - metrics. MASE divides holdout MAE by the in-sample MAE of the
    // seasonal-naive forecast (|y_t - y_{t-12}| averaged over training).
    double denominator = 0;
    for (size_t t = 12; t < cut; ++t)
      denominator += std::abs(y[t] - y[t - 12]) / (cut - 12);
    std::ostringstream metrics;
    metrics << "{\"rmse\":" << ml::rmse(actual, pred)
            << ",\"mae\":" << ml::mae(actual, pred)
            << ",\"r2\":" << ml::r2(actual, pred)
            << ",\"mase\":" << ml::mae(actual, pred) / denominator
            << ",\"seasonal_naive_rmse\":" << ml::rmse(actual, naive)
            << ",\"last_value_rmse\":" << ml::rmse(actual, last_pred) << "}\n";
    a.write("evaluation/metrics.json", metrics.str());
    // Figures: actual vs forecast vs seasonal naive, and the residual trace.
    ml::Plot chart;
    chart.line(time, actual, "actual");
    chart.line(time, pred, "forecast");
    chart.line(time, naive, "seasonal naive");
    chart.title("One-step forecasts with observed history");
    a.figure("evaluation/figures/forecast.svg", chart);
    ml::Plot errors;
    errors.line(time, residual, "residual");
    a.figure("evaluation/figures/residuals.svg", errors);
    // Stage 8 - sample autocorrelation of the training window, lags 1..24,
    // normalised by the total sum of squares (standard ACF estimator).
    std::ostringstream acf;
    acf << "lag,autocorrelation\n";
    double avg = ml::mean(training), ss = 0;
    for (double v : training)
      ss += (v - avg) * (v - avg);
    for (size_t lag = 1; lag <= 24; ++lag) {
      double cov = 0;
      for (size_t i = lag; i < training.size(); ++i)
        cov += (training[i] - avg) * (training[i - lag] - avg);
      acf << lag << ',' << cov / ss << '\n';
    }
    a.write("eda/autocorrelation.csv", acf.str());
    // Stage 9 - schema, manifest and narrative report.
    a.write("model/feature_schema.json",
            legacy ? std::string("{\"features\":[\"lag_1\",\"lag_2\",\"lag_12\","
                                 "\"month_index\",\"season_sin\",\"season_cos\"],"
                                 "\"forecast_horizon\":1,\"season_length\":12}\n")
                   : "{\"features\":[\"univariate_history\"],\"model\":\"" +
                         spec.name +
                         "\",\"forecast_horizon\":1,\"season_length\":12}\n");
    std::ostringstream manifest;
    manifest << "{\"dataset\":\"AirPassengers 1949-1960\",\"holdout_start\":"
             << cut << "," << manifest_fields(spec, chosen)
             << ",\"selection\":\"expanding training "
                "window\",\"training_fingerprint\":\""
             << ml::fingerprint(d) << "\"}\n";
    a.write("run_manifest.json", manifest.str());
    std::ostringstream report;
    report << "# Monthly passenger forecasting\n\nHeld-out RMSE: "
           << ml::rmse(actual, pred)
           << " thousand passengers; seasonal-naive RMSE: "
           << ml::rmse(actual, naive) << ". " << selection_phrase(spec, chosen)
           << " was selected using expanding windows ending before the "
              "holdout.\n\nThis is one-step forecasting: each previous test "
              "observation becomes available before predicting the next month. "
              "Coefficients and scaling stay frozen across the final test. It "
              "is not a 29-month forecast made at a single origin. The "
              "increasing trend and seasonal amplitude violate simple "
              "stationarity assumptions; inspect training autocorrelations and "
              "residual figures. Lag features contain only already observed "
              "values. MASE scales errors by training seasonal-naive MAE. The "
              "saved archive is the pre-holdout model and history, so reload "
              "predicts the first held-out month.\n";
    if (!legacy)
      report << "\nThe " << spec.name
             << " state (not the coefficients) is updated by each observed "
                "holdout month; the candidate grid is listed in "
                "validation/candidate_scores.csv.\n";
    a.write("report.md", report.str());
    a.write("execution.log", "Chronological rolling-origin selection, held-out "
                             "evaluation and reload verification completed.\n");
    std::cout << metrics.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
// Backward-compatible overload used by modules 01-03: `baseline` true selects
// the seasonal-naive family, false the lag regression.
inline int workflow(const std::string &root, const std::string &path,
                    bool baseline, int argc, char **argv) {
  return workflow(root, path,
                  baseline ? Forecaster::SeasonalNaive
                           : Forecaster::LagRegression,
                  argc, argv);
}
} // namespace forecast_course
