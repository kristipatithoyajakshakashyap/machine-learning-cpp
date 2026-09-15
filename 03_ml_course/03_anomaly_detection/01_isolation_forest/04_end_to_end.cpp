// 03_ml_course/03_anomaly_detection/01_isolation_forest/04_end_to_end.cpp
// Purpose: the complete isolation-forest experiment on Palmer Penguins: split
//          60/20/20, EDA on train, standardize, fit, calibrate a threshold on
//          the validation scores, score the test rows, save the fitted state
//          and prove it reloads bit-for-bit.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json, eda/,
//          data/row_assignments.csv, validation/{calibration_scores.csv,
//          threshold.json}, evaluation/{predictions.csv,metrics.json,
//          figures/scores.svg}, model/{model_state.txt,preprocessing_state.txt,
//          feature_schema.csv}, inference/reload_verification.json, report.md,
//          execution.log.
// Run target: uif_end_to_end (also ctest -R uif_workflow).
#include <iostream>

#include "02_unsupervised/workflow.hpp"
#include "IsolationForest.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load: keep missing values (true) so the preprocessor learns
  // median imputation. Species labels in ds.y are never used for scoring.
  auto ds = load_penguins(DATA_DIR, true);
  // Feature order the saved preprocessor/model expect; uif_predict relies on it.
  std::ostringstream schema;
  schema << "feature\n";
  for (const auto& name : ds.feature_names) schema << csv_quote(name) << '\n';
  a.write("model/feature_schema.csv", schema.str());
  // Stage 2 - split: seeded shuffle, then 60% train / 20% validation / 20% test.
  // Fix random partitions before any learned preprocessing or threshold
  // selection.
  std::vector<size_t> ids(ds.n());
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937 gen(42);
  std::shuffle(ids.begin(), ids.end(), gen);
  size_t train_end = ds.n() * 6 / 10, val_end = ds.n() * 8 / 10;
  auto train = select_rows(ds, {ids.begin(), ids.begin() + train_end}),
       val = select_rows(ds, {ids.begin() + train_end, ids.begin() + val_end}),
       test = select_rows(ds, {ids.begin() + val_end, ids.end()});
  // Stage 3 - EDA on the training split only (eda/ tables and figures).
  eda(train, a);
  // Stage 4 - preprocessing: impute + z-score fitted on train, applied to all.
  Preprocessor prep;
  prep.fit(train.X);
  auto X = prep.transform(train.X), V = prep.transform(val.X),
       T = prep.transform(test.X);
  // Stage 5 - fit: default forest (100 trees, 256-row subsamples, seed 42).
  IsolationForest model;
  model.fit(X);
  // Stage 6 - calibration: score validation rows and take the 95th percentile
  // as the flag threshold (a review-budget choice, not an estimated prevalence).
  Vec vs = model.score_samples(V);
  std::ostringstream calibration;
  calibration << std::setprecision(17) << "row_id,validation_anomaly_score\n";
  for (size_t i = 0; i < vs.size(); ++i)
    calibration << (val.row_ids.empty() ? ids[train_end + i] : val.row_ids[i])
                << ',' << vs[i] << '\n';
  a.write("validation/calibration_scores.csv", calibration.str());
  std::sort(vs.begin(), vs.end());
  double threshold = vs[size_t(.95 * (vs.size() - 1))];
  model.set_threshold(threshold);
  std::ostringstream threshold_record;
  threshold_record << std::setprecision(17)
                   << "{\"quantile\":0.95,\"threshold\":" << threshold
                   << ",\"calibration_rows\":" << vs.size() << "}\n";
  a.write("validation/threshold.json", threshold_record.str());
  // Stage 7 - evaluation: scores and flags on the untouched test split.
  auto scores = model.score_samples(T);
  auto flags = model.predict(T, threshold);
  // Stage 8 - snapshot: save model + preprocessor as text archives.
  unsup::snapshot(model, prep, a);
  // Stage 9 - verify: a reloaded forest must reproduce the test scores exactly.
  IsolationForest loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  unsup::verify(scores, loaded.score_samples(T), a);
  // Stage 10 - artifacts: per-row test predictions, split membership, metrics,
  // manifest, histogram and the narrative report.
  std::ostringstream rows, splits;
  rows << std::setprecision(17) << "row_id,anomaly_score,flag\n";
  for (size_t i = 0; i < scores.size(); ++i)
    rows << (ds.row_ids.empty() ? ids[val_end + i]
                                : ds.row_ids[ids[val_end + i]])
         << ',' << scores[i] << ',' << flags[i] << '\n';
  a.write("evaluation/predictions.csv", rows.str());
  splits << "row_id,partition\n";
  for (size_t i = 0; i < ids.size(); ++i)
    splits << (ds.row_ids.empty() ? ids[i] : ds.row_ids[ids[i]]) << ','
           << (i < train_end ? "train"
               : i < val_end ? "validation"
                             : "test")
           << '\n';
  a.write("data/row_assignments.csv", splits.str());
  a.write("evaluation/metrics.json",
          "{\"threshold\":" + std::to_string(threshold) +
              ",\"test_flag_rate\":" + std::to_string(mean(flags)) + "}\n");
  a.write("run_manifest.json",
          "{\"seed\":42,\"trees\":100,\"sample_size\":256,\"fingerprint\":" +
              json_quote(fingerprint(ds)) + "}\n");
  Plot h;
  h.title("Held-out anomaly scores");
  h.histogram(scores, 20);
  a.figure("evaluation/figures/scores.svg", h);
  a.write("report.md",
          "# Isolation Forest on Penguins\n\nTraining uses 60%, threshold "
          "calibration 20%, final inspection 20%. The 95th percentile "
          "validation threshold is a review-budget assumption, not an "
          "estimated prevalence. Species labels are not anomaly ground truth; "
          "precision/recall cannot be claimed here. Inspect flagged "
          "measurements and original source records. Score tests on planted "
          "synthetic anomalies are separate algorithm checks.\n");
  a.write("execution.log",
          "Forest trained; validation threshold calibrated; held-out scores "
          "and reload checked.\n");
  std::cout << "Saved anomaly experiment under " << RUN_OUTPUT_DIR << '\n';
}
