// 03_ml_course/helper/pipeline/supervised.hpp
//
// Purpose : the shared end-to-end project for every dense supervised
//           module (01_supervised/*). run_supervised<Factory>() performs
//           holdout split, EDA, training-only cross-validated parameter
//           selection, final fit, holdout evaluation with bootstrap
//           intervals, persistence and reload verification. With --predict
//           it instead serves inference from a saved run.
// Inputs  : original - the ml::Dataset loaded by the caller (typically from
//           DATA_DIR/<name>.csv); root - RUN_OUTPUT_DIR of the caller.
//           CLI: [--quick] | --predict input.csv --model <run>/model.
// Outputs : training run: <root>/full/ (or quick/) with data/, eda/,
//           validation/, evaluation/, model/, inference/, run_manifest.json,
//           report.md, execution.log. Inference run: <root>/predictions.csv.
// Used by : every 01_supervised *_end_to_end and *_predict target, plus
//           tests/numerical/core.cpp (stratified_holdout, Preprocessor).
//           Header-only; no dedicated target.
//
// Model contract (what `make(parameter)` must return):
//   void fit(const Mat &X, const Vec &y);   Vec predict(const Mat &X) const;
//   void save(std::ostream &) const;        void load(std::istream &);
//   optional: Mat predict_proba(const Mat &) const   (enables ROC, log loss,
//             PR and calibration outputs when it yields two columns)
//   optional: Vec decision_function(const Mat &) const (enables ROC AUC only)
// Saved run layout (<run>/model/): model_state.txt, preprocessing_state.txt,
//   target_state.txt, features.txt - all four are needed by --predict.
#pragma once
#include "helper/dataframe/dataframe.hpp"
#include "helper/eval/cross_validation.hpp"
#include "helper/math/metrics.hpp"
#include "helper/math/optim.hpp"
#include "helper/preprocessing/preprocessor.hpp"
#include "helper/reporting/artifacts.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace ml {
// True when "--predict" appears anywhere on the command line. Lets a lesson
// skip dataset loading or extra work when it is only asked for inference.
inline bool inference_requested(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--predict")
      return true;
  return false;
}
// Compile-time detection of the optional model methods (SFINAE on
// predict_proba(Mat) and decision_function(Mat)).
template <class T, class = void> struct has_probability : std::false_type {};
template <class T>
struct has_probability<T, std::void_t<decltype(std::declval<T>().predict_proba(
                              std::declval<Mat>()))>> : std::true_type {};
template <class T, class = void> struct has_decision : std::false_type {};
template <class T>
struct has_decision<T, std::void_t<decltype(std::declval<T>().decision_function(
                           std::declval<Mat>()))>> : std::true_type {};
// Unweighted mean of per-class F1 over every class present in y or pred.
// A class with no true or predicted rows contributes 0. Used as the
// selection metric for classification. Throws on empty/mismatched input.
// Complexity: O(classes * n).
inline double macro_f1(const Vec &y, const Vec &pred) {
  if (y.empty() || y.size() != pred.size())
    throw std::invalid_argument("F1 length mismatch");
  std::map<double, size_t> classes;
  for (double c : y)
    classes[c] = 1;
  for (double c : pred)
    classes[c] = 1;
  double total = 0;
  for (const auto &kv : classes) {
    double tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      if (y[i] == kv.first && pred[i] == kv.first)
        ++tp;
      else if (pred[i] == kv.first)
        ++fp;
      else if (y[i] == kv.first)
        ++fn;
    }
    total += (2 * tp + fp + fn) > 0 ? 2 * tp / (2 * tp + fp + fn) : 0;
  }
  return total / static_cast<double>(classes.size());
}
// Standardises a regression target (fitted on training rows only) so every
// model sees a zero-mean, unit-variance y; predictions are mapped back with
// inverse(). For classification it is the identity (mean 0, scale 1).
// Archived next to the model so --predict returns original units.
struct TargetTransform {
  double mean = 0, scale = 1;
  // Learn mean / std-dev from training y. A constant target (scale ~ 0)
  // falls back to scale 1 so transform() never divides by zero.
  void fit(const Vec &y, bool classification) {
    mean = classification ? 0 : ml::mean(y);
    scale = classification ? 1 : std::sqrt(variance(y, true));
    if (scale < 1e-12)
      scale = 1;
  }
  // y -> (y - mean) / scale, element-wise. O(n).
  Vec transform(Vec y) const {
    for (double &v : y)
      v = (v - mean) / scale;
    return y;
  }
  // Exact inverse of transform(): standardised -> original units. O(n).
  Vec inverse(Vec y) const {
    for (double &v : y)
      v = v * scale + mean;
    return y;
  }
  // Archive format: mean then scale (two doubles).
  void save(std::ostream &o) const { archive::write(o, mean, scale); }
  // Reads the pair back and rejects non-finite or non-positive scales so a
  // corrupt archive cannot silently produce NaN predictions.
  void load(std::istream &i) {
    archive::read(i, mean, scale);
    if (!(scale > 0) || !std::isfinite(scale) || !std::isfinite(mean))
      throw std::runtime_error("Invalid target scale");
  }
};
// Class-stratified train/test split.
// Parameters: y - labels; frac - training fraction per class; seed - RNG
// seed for the per-class shuffle. Returns a Fold whose train/test index
// lists are sorted. Every class keeps at least one row on each side.
// Throws std::invalid_argument for a class with fewer than two rows.
// Complexity: O(n log n).
inline Fold stratified_holdout(const Vec &y, double frac, uint32_t seed) {
  std::map<double, std::vector<size_t>> groups;
  for (size_t i = 0; i < y.size(); ++i)
    groups[y[i]].push_back(i);
  Fold f;
  seed_rng(seed);
  for (auto &kv : groups) {
    auto ids = kv.second;
    if (ids.size() < 2)
      throw std::invalid_argument("Each class needs at least two observations");
    shuffle_indices(ids);
    // Clamp so neither partition of this class is empty.
    size_t cut = std::clamp(static_cast<size_t>(frac * ids.size()), size_t(1),
                            ids.size() - 1);
    f.train.insert(f.train.end(), ids.begin(),
                   ids.begin() + static_cast<ptrdiff_t>(cut));
    f.test.insert(f.test.end(), ids.begin() + static_cast<ptrdiff_t>(cut),
                  ids.end());
  }
  std::sort(f.train.begin(), f.train.end());
  std::sort(f.test.begin(), f.test.end());
  return f;
}
// Run the complete supervised project (or inference).
// Parameters:
//   original       - full dataset as loaded by the caller.
//   root           - results directory for this executable (RUN_OUTPUT_DIR).
//   name           - workflow name for the report and manifest.
//   params         - candidate values of the model's single hyper-parameter.
//   make           - callable: make(double) -> model (see file header).
//   classification - true selects macro F1 and stratified splits; false
//                    selects negative RMSE and plain random splits.
//   argc/argv      - command line; see file header for the accepted flags.
// Returns 0 on success, 1 on any exception (message printed to stderr).
// Protocol: optional quick subsample (400 rows, seed 42) -> 80/20 holdout
// (seed 42) -> EDA on training rows -> k-fold CV on training rows only
// (k = 5, reduced to the smallest class count for classification, seed
// 123) with Preprocessor and TargetTransform refitted in every fold -> refit
// the winner on all training rows -> evaluate the holdout once with a
// 1000-sample bootstrap (seed 2026) -> archive model/preprocessing/target
// and verify a reload reproduces the holdout predictions.
template <class Factory>
int run_supervised(const Dataset &original, const std::string &root,
                   const std::string &name, const Vec &params, Factory make,
                   bool classification, int argc = 0, char **argv = nullptr) {
  try {
    if (params.empty())
      throw std::invalid_argument("Empty parameter grid");
    // ---- command line -----------------------------------------------------
    bool quick = false;
    std::string input, model_path;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--quick")
        quick = true;
      else if (arg == "--predict" && i + 1 < argc)
        input = argv[++i];
      else if (arg == "--model" && i + 1 < argc)
        model_path = argv[++i];
      else
        throw std::invalid_argument(
            "Usage: [--quick] or --predict input.csv --model run/model");
    }
    auto model = make(params.front());
    // ---- inference mode ---------------------------------------------------
    // Restores the four archives from <model_path>, reads the CSV by the
    // saved feature names (missing cells become NaN and are imputed) and
    // writes <root>/predictions.csv in original target units.
    if (!input.empty()) {
      if (model_path.empty())
        throw std::invalid_argument("--model required for inference");
      archive::load_file(model_path + "/model_state.txt", model);
      Preprocessor prep;
      archive::load_file(model_path + "/preprocessing_state.txt", prep);
      TargetTransform target;
      archive::load_file(model_path + "/target_state.txt", target);
      std::vector<std::string> names;
      std::ifstream namesin(model_path + "/features.txt");
      archive::read(namesin, names);
      auto df = read_csv(input);
      if (df.rows() == 0 || names.empty())
        throw std::invalid_argument(
            "Inference needs data rows and a saved feature schema");
      Mat X(df.rows(), Vec(names.size()));
      for (size_t j = 0; j < names.size(); ++j) {
        const auto &col = df.numeric(names[j]);
        for (size_t i = 0; i < df.rows(); ++i)
          X[i][j] = col[i].value_or(std::numeric_limits<double>::quiet_NaN());
      }
      Vec predicted = target.inverse(model.predict(prep.transform(X)));
      Artifacts a(root, ".");
      std::ostringstream out;
      out << std::setprecision(17) << "row_id,prediction\n";
      for (size_t i = 0; i < predicted.size(); ++i)
        out << i << ',' << predicted[i] << '\n';
      a.write("predictions.csv", out.str());
      std::cout << a.path("predictions.csv") << '\n';
      return 0;
    }
    // ---- data and outer holdout -------------------------------------------
    // --quick keeps a seeded random subset of 400 rows for smoke tests.
    Dataset ds =
        quick && original.n() > 400 ? subsample(original, 400, 42) : original;
    if (ds.n() < 10 || ds.y.size() != ds.n())
      throw std::invalid_argument(
          "Workflow requires at least ten labeled observations");
    for (double y : ds.y)
      if (!std::isfinite(y))
        throw std::invalid_argument("Missing or infinite target");
    Artifacts a(root, quick ? "quick" : "full");
    Fold outer = classification ? stratified_holdout(ds.y, .8, 42)
                                : train_test_indices(ds.n(), .8, 42);
    Dataset tr = select_rows(ds, outer.train), te = select_rows(ds, outer.test);
    // EDA is computed on training rows only so the holdout stays untouched.
    eda(tr, a);
    // ---- inner CV folds (training rows only) ------------------------------
    // k = 5, but never more than the rarest class count so every fold can be
    // stratified; fewer than 2 folds is an error.
    size_t k = 5;
    if (classification) {
      std::map<double, size_t> count;
      for (double y : tr.y)
        ++count[y];
      for (const auto &kv : count)
        k = std::min(k, kv.second);
    }
    if (k < 2)
      throw std::invalid_argument("Insufficient class support for CV");
    auto folds =
        classification ? stratified_kfold(tr.y, k, 123) : kfold(tr.n(), k, 123);
    // Row assignment log: which original row went to train/test and to which
    // inner fold, so any number in the report can be traced to source rows.
    std::ostringstream assignments, cv, curve;
    assignments << "row_id,partition,fold\n";
    auto original_id = [&](size_t i) {
      return ds.row_ids.empty() ? i : ds.row_ids.at(i);
    };
    for (size_t i : outer.train)
      assignments << original_id(i) << ",train,-1\n";
    for (size_t i : outer.test)
      assignments << original_id(i) << ",test,-1\n";
    for (size_t f = 0; f < folds.size(); ++f) {
      for (size_t i : folds[f].train)
        assignments << original_id(outer.train.at(i)) << ",cv_train," << f
                    << '\n';
      for (size_t i : folds[f].test)
        assignments << original_id(outer.train.at(i)) << ",cv_validation," << f
                    << '\n';
    }
    a.write("data/row_assignments.csv", assignments.str());
    // Preserve raw holdout inputs so an independent process can replay the
    // model.
    std::ostringstream raw_holdout;
    raw_holdout << std::setprecision(17);
    for (size_t j = 0; j < ds.feature_names.size(); ++j) {
      if (j)
        raw_holdout << ',';
      raw_holdout << csv_quote(ds.feature_names[j]);
    }
    raw_holdout << '\n';
    for (const auto &row : te.X) {
      for (size_t j = 0; j < row.size(); ++j) {
        if (j)
          raw_holdout << ',';
        if (std::isfinite(row[j]))
          raw_holdout << row[j];
      }
      raw_holdout << '\n';
    }
    a.write("data/holdout_features.csv", raw_holdout.str());
    write_split(outer, a.path("data/holdout.split"));
    // ---- model selection: CV score per candidate parameter ---------------
    // Score is macro F1 (higher better) or negative RMSE (so "higher is
    // better" holds for both). Strict '>' keeps the first candidate on ties.
    cv << std::setprecision(17) << "parameter,fold,score\n";
    double best = -std::numeric_limits<double>::infinity(),
           bestp = params.front();
    Vec averages;
    for (double p : params) {
      Vec scores;
      for (size_t f = 0; f < folds.size(); ++f) {
        auto ft = select_rows(tr, folds[f].train),
             fv = select_rows(tr, folds[f].test);
        // Preprocessing and target scaling are refitted inside the fold so
        // validation rows never influence imputation or scaling.
        Preprocessor prep;
        prep.fit(ft.X);
        TargetTransform target;
        target.fit(ft.y, classification);
        // Same seed before every fit: stochastic models are compared on
        // parameters, not on random initialisation.
        seed_rng(42);
        auto m = make(p);
        m.fit(prep.transform(ft.X), target.transform(ft.y));
        auto yp = target.inverse(m.predict(prep.transform(fv.X)));
        double score = classification ? macro_f1(fv.y, yp) : -rmse(fv.y, yp);
        if (!std::isfinite(score))
          throw std::runtime_error("Nonfinite CV score");
        scores.push_back(score);
        cv << p << ',' << f << ',' << score << '\n';
      }
      double avg = ml::mean(scores);
      averages.push_back(avg);
      curve << p << ',' << avg << ',' << std::sqrt(variance(scores, false))
            << '\n';
      if (avg > best) {
        best = avg;
        bestp = p;
      }
    }
    a.write("validation/candidate_scores.csv", cv.str());
    a.write("validation/summary.csv",
            "parameter,mean_score,std_score\n" + curve.str());
    a.write(
        "validation/selected_parameters.json",
        "{\"parameter\":" + std::to_string(bestp) + ",\"selection_metric\":" +
            json_quote(classification ? "macro_f1" : "negative_rmse") + "}\n");
    Plot val;
    val.title("Training-only cross-validation");
    val.line(params, averages);
    a.figure("validation/figures/parameter_curve.svg", val);
    // ---- final fit on all training rows, single holdout evaluation --------
    Preprocessor prep;
    prep.fit(tr.X);
    TargetTransform target;
    target.fit(tr.y, classification);
    seed_rng(42);
    model = make(bestp);
    model.fit(prep.transform(tr.X), target.transform(tr.y));
    Mat Xtest = prep.transform(te.X);
    Vec yp = target.inverse(model.predict(Xtest));
    // Naive baseline: training mean (regression) or majority class
    // (classification, lowest label wins ties).
    Vec baseline(te.n(), ml::mean(tr.y));
    if (classification) {
      std::map<double, size_t> counts;
      for (double y : tr.y)
        ++counts[y];
      double label = counts.begin()->first;
      for (const auto &kv : counts)
        if (kv.second > counts[label])
          label = kv.first;
      std::fill(baseline.begin(), baseline.end(), label);
    }
    std::ostringstream metrics, pred, errors, report;
    metrics << std::setprecision(17) << "{\"n_train\":" << tr.n()
            << ",\"n_test\":" << te.n() << ",\"cv_score\":" << best;
    double score = classification ? macro_f1(te.y, yp) : rmse(te.y, yp);
    if (classification) {
      metrics << ",\"accuracy\":" << accuracy(te.y, yp)
              << ",\"macro_f1\":" << score
              << ",\"baseline_macro_f1\":" << macro_f1(te.y, baseline);
      size_t nc =
          static_cast<size_t>(*std::max_element(ds.y.begin(), ds.y.end())) + 1;
      auto cm = confusion(te.y, yp, nc);
      Plot cp;
      cp.title("Holdout confusion matrix");
      cp.heatmap(cm);
      a.figure("evaluation/figures/confusion.svg", cp);
      std::ostringstream classes;
      classes << "class,precision,recall,f1,support\n";
      for (size_t c = 0; c < nc; ++c) {
        double tp = cm[c][c], fp = 0, fn = 0;
        for (size_t j = 0; j < nc; ++j)
          if (c != j) {
            fp += cm[j][c];
            fn += cm[c][j];
          }
        classes << c << ',' << (tp + fp ? tp / (tp + fp) : 0) << ','
                << (tp + fn ? tp / (tp + fn) : 0) << ','
                << (2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0) << ','
                << tp + fn << '\n';
      }
      a.write("evaluation/per_class.csv", classes.str());
    } else {
      metrics << ",\"rmse\":" << score << ",\"mae\":" << mae(te.y, yp)
              << ",\"r2\":" << r2(te.y, yp)
              << ",\"baseline_rmse\":" << rmse(te.y, baseline);
      Plot p;
      p.title("Holdout actual versus predicted");
      p.scatter(te.y, yp);
      a.figure("evaluation/figures/predictions.svg", p);
      Vec residual(te.n());
      for (size_t i = 0; i < te.n(); ++i)
        residual[i] = te.y[i] - yp[i];
      Plot r;
      r.title("Holdout residual distribution");
      r.histogram(residual, 30);
      a.figure("evaluation/figures/residuals.svg", r);
      Plot rv;
      rv.title("Residual versus fitted value");
      rv.scatter(yp, residual);
      a.figure("evaluation/figures/residual_vs_fitted.svg", rv);
    }
    Vec probabilities;
    Mat all_probabilities;
    if (classification) {
      if constexpr (has_probability<decltype(model)>::value) {
        all_probabilities = model.predict_proba(Xtest);
        if (!all_probabilities.empty() && all_probabilities[0].size() == 2)
          for (const auto &row : all_probabilities)
            probabilities.push_back(row[1]);
      }
      if (!probabilities.empty()) {
        metrics << ",\"roc_auc\":" << auc(te.y, probabilities)
                << ",\"log_loss\":" << binary_logloss(te.y, probabilities);
        auto roc = roc_curve(te.y, probabilities);
        Vec x, y;
        for (const auto &point : roc) {
          x.push_back(point.first);
          y.push_back(point.second);
        }
        Plot r;
        r.title("Holdout ROC");
        r.line(x, y);
        a.figure("evaluation/figures/roc.svg", r);
        std::ostringstream pr, cal;
        pr << "threshold,precision,recall\n";
        Vec recalls, precisions;
        for (int t = 0; t <= 100; ++t) {
          double threshold = t / 100.0;
          Vec hard = probabilities;
          for (double &v : hard)
            v = v >= threshold ? 1 : 0;
          auto s = binary_scores(te.y, hard);
          pr << threshold << ',' << s.precision << ',' << s.recall << '\n';
          recalls.push_back(s.recall);
          precisions.push_back(s.precision);
        }
        a.write("evaluation/pr_curve.csv", pr.str());
        Plot p;
        p.title("Precision recall (fixed threshold grid)");
        p.line(recalls, precisions);
        a.figure("evaluation/figures/pr.svg", p);
        cal << "bin,count,mean_probability,positive_fraction\n";
        Vec cp, cy;
        for (int b = 0; b < 10; ++b) {
          double sp = 0, sy = 0, n = 0;
          for (size_t i = 0; i < te.n(); ++i)
            if (std::min(9, static_cast<int>(probabilities[i] * 10)) == b) {
              sp += probabilities[i];
              sy += te.y[i];
              ++n;
            }
          if (n) {
            cal << b << ',' << n << ',' << sp / n << ',' << sy / n << '\n';
            cp.push_back(sp / n);
            cy.push_back(sy / n);
          }
        }
        a.write("evaluation/calibration.csv", cal.str());
        Plot c;
        c.title("Holdout reliability diagram");
        c.line(cp, cy);
        a.figure("evaluation/figures/calibration.svg", c);
      } else if constexpr (has_decision<decltype(model)>::value) {
        auto scores = model.decision_function(Xtest);
        metrics << ",\"roc_auc\":" << auc(te.y, scores);
      }
    }
    seed_rng(2026);
    Vec boot;
    for (size_t b = 0; b < 1000; ++b) {
      Vec y, p;
      for (size_t i = 0; i < te.n(); ++i) {
        size_t j = next_rng(0, static_cast<uint32_t>(te.n() - 1));
        y.push_back(te.y[j]);
        p.push_back(yp[j]);
      }
      boot.push_back(classification ? macro_f1(y, p) : rmse(y, p));
    }
    std::sort(boot.begin(), boot.end());
    metrics << ",\"bootstrap_ci95\":[" << boot[25] << ',' << boot[974]
            << "]}\n";
    a.write("evaluation/metrics.json", metrics.str());
    pred << std::setprecision(17) << "row_id,actual,prediction";
    if (!all_probabilities.empty())
      for (size_t c = 0; c < all_probabilities[0].size(); ++c)
        pred << ",probability_" << c;
    pred << '\n';
    for (size_t i = 0; i < te.n(); ++i) {
      pred << original_id(outer.test[i]) << ',' << te.y[i] << ',' << yp[i];
      if (!all_probabilities.empty())
        for (double p : all_probabilities[i])
          pred << ',' << p;
      pred << '\n';
    }
    a.write("evaluation/predictions.csv", pred.str());
    archive::save_file(a.path("model/model_state.txt"), model);
    archive::save_file(a.path("model/preprocessing_state.txt"), prep);
    archive::save_file(a.path("model/target_state.txt"), target);
    std::ostringstream names;
    archive::write(names, ds.feature_names);
    a.write("model/features.txt", names.str());
    auto reloaded = make(bestp);
    archive::load_file(a.path("model/model_state.txt"), reloaded);
    Preprocessor rp;
    archive::load_file(a.path("model/preprocessing_state.txt"), rp);
    TargetTransform rt;
    archive::load_file(a.path("model/target_state.txt"), rt);
    auto replay = rt.inverse(reloaded.predict(rp.transform(te.X)));
    double maxdiff = 0;
    for (size_t i = 0; i < yp.size(); ++i)
      maxdiff = std::max(maxdiff, std::fabs(yp[i] - replay[i]));
    if (maxdiff > 1e-9)
      throw std::runtime_error("Reload predictions differ");
    a.write("inference/reload_verification.json",
            "{\"max_absolute_difference\":" + std::to_string(maxdiff) +
                ",\"passed\":true}\n");
    a.write(
        "run_manifest.json",
        "{\"workflow\":" + json_quote(name) +
            ",\"dataset_fingerprint\":" + json_quote(fingerprint(ds)) +
            ",\"seed\":42,\"cv_seed\":123,\"bootstrap_seed\":2026,\"quick\":" +
            (quick ? "true" : "false") +
            ",\"raw_rows\":" + std::to_string(original.n()) +
            ",\"used_rows\":" + std::to_string(ds.n()) + "}\n");
    report << "# " << name << "\n\n"
           << (quick ? "Quick verification uses a documented deterministic "
                       "subset. Run without --quick for the full dataset.\n\n"
                     : "Full supplied dataset run.\n\n")
           << "Training rows: " << tr.n() << "; untouched test rows: " << te.n()
           << ". " << folds.size() << " folds selected parameter " << bestp
           << " using " << (classification ? "macro F1" : "negative RMSE")
           << ".\n\n"
           << "Holdout " << (classification ? "macro F1" : "RMSE") << ": "
           << score << ". Bootstrap 95% percentile interval [" << boot[25]
           << ", " << boot[974]
           << "]. These intervals describe sampling variation conditional on "
              "this fitted model, not repeated model-selection uncertainty.\n\n"
           << "See evaluation/metrics.json for baseline comparison, "
              "evaluation/predictions.csv for every test prediction and "
              "eda/findings.md for data quality. Scaling and median imputation "
              "were fitted separately inside every fold. Test outcomes did not "
              "select parameters. Model and preprocessing reload reproduced "
              "predictions.\n\n"
           << "Limitations: one holdout split; evaluate transfer to new "
              "populations before deployment. Correlations and feature "
              "relations are descriptive.\n";
    a.write("report.md", report.str());
    a.write(
        "evaluation/error_analysis.md",
        classification
            ? "Inspect the per-class report and confusion matrix for "
              "minority-class failures. Per-row errors are available in "
              "predictions.csv; select improvements using training CV, not "
              "this holdout.\n"
            : "Inspect residual plots for heteroscedasticity and systematic "
              "errors. Predictions are in original target units. Select "
              "improvements using training CV, not this holdout.\n");
    a.write("execution.log", report.str() + metrics.str());
    std::cout << name << ": " << a.path("report.md") << '\n' << metrics.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << name << ": " << e.what() << '\n';
    return 1;
  }
}
} // namespace ml
