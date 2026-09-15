// 03_ml_course/02_unsupervised/06_pca/04_end_to_end.cpp
// Purpose: the complete PCA experiment on Palmer Penguins with a proper
//          train/test split: fit imputation, scaling and PCA on 80% of the
//          rows, choose the dimension by a fixed 95% variance rule, report the
//          hold-out reconstruction error, snapshot and verify the reload.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json,
//          data/row_assignments.csv, eda/ (training rows only),
//          validation/{candidate_scores.csv,selection_rule.md},
//          evaluation/{projections.csv,metrics.json,
//          figures/explained_variance.svg},
//          model/{model_state.txt,preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: upca_end_to_end (also ctest -R upca_workflow); upca_predict then
//             reloads model/ to project new rows.
#include <iostream>

#include "../workflow.hpp"
#include "PCA.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load (missing kept) and record the feature order the saved
  // preprocessor/model expect (upca_predict relies on this order).
  auto ds = load_penguins(DATA_DIR, true);
  std::ostringstream schema;
  schema << "feature\n";
  for (const auto& name : ds.feature_names) schema << csv_quote(name) << '\n';
  a.write("model/feature_schema.csv", schema.str());
  // Stage 2 - split: seeded shuffle (42), first 80% train, rest test. Unlike
  // the clustering lessons, PCA here is evaluated on untouched rows.
  std::vector<size_t> ids(ds.n());
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(ids.begin(), ids.end(), rng);
  size_t n = ds.n() * 8 / 10;
  auto train = select_rows(ds, {ids.begin(), ids.begin() + n}),
       test = select_rows(ds, {ids.begin() + n, ids.end()});
  // Stage 3 - EDA on the training partition only (no peeking at test rows).
  eda(train, a);
  // Stage 4 - preprocessing fitted on train, applied to both partitions.
  Preprocessor prep;
  prep.fit(train.X);
  auto X = prep.transform(train.X), T = prep.transform(test.X);
  // Stage 5 - selection: fit all p axes on train and keep the smallest k whose
  // cumulative explained variance reaches 95%. The rule is fixed in advance;
  // test reconstruction error is reported but never used to pick k.
  PCA full(ds.p());
  full.fit(X);
  auto ratios = full.explained_variance_ratio();
  size_t k = 0;
  double cumulative = 0;
  while (k < ratios.size() && cumulative < .95) cumulative += ratios[k++];
  std::ostringstream selection;
  selection << std::setprecision(17)
            << "components,training_variance_retained,selected\n";
  double retained = 0;
  for (size_t c = 0; c < ratios.size(); ++c) {
    retained += ratios[c];
    selection << c + 1 << ',' << retained << ',' << (c + 1 == k) << '\n';
  }
  a.write("validation/candidate_scores.csv", selection.str());
  a.write("validation/selection_rule.md",
          "Select the smallest dimension retaining at least 95% of training "
          "variance. The threshold is fixed before evaluation; test "
          "reconstruction error does not select components.\n");
  // Stage 6 - final fit with k components; hold-out evaluation = mean squared
  // reconstruction error per cell, in standardized units.
  PCA model(k);
  model.fit(X);
  auto z = model.transform(T), reconstructed = model.inverse_transform(z);
  double error = 0;
  for (size_t i = 0; i < T.size(); ++i)
    for (size_t j = 0; j < ds.p(); ++j)
      error +=
          (T[i][j] - reconstructed[i][j]) * (T[i][j] - reconstructed[i][j]);
  error /= T.size() * ds.p();
  // Stage 7 - snapshot (model + preprocessor) for upca_predict.
  unsup::snapshot(model, prep, a);
  // Stage 8 - verify: reloaded projections of the test rows must match to
  // 1e-10, checked row by row (verify compares vectors, so loop over rows).
  PCA loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  auto again = loaded.transform(T);
  for (size_t i = 0; i < z.size(); ++i) unsup::verify(z[i], again[i], a);
  // Stage 9 - tables: test-row projections keyed by the original row id, and
  // which partition every row landed in.
  std::ostringstream csv, splits;
  csv << std::setprecision(17) << "row_id";
  for (size_t c = 0; c < k; ++c) csv << ",PC" << c + 1;
  csv << '\n';
  for (size_t i = 0; i < z.size(); ++i) {
    csv << (ds.row_ids.empty() ? ids[n + i] : ds.row_ids[ids[n + i]]);
    for (double v : z[i]) csv << ',' << v;
    csv << '\n';
  }
  a.write("evaluation/projections.csv", csv.str());
  splits << "row_id,partition\n";
  for (size_t i = 0; i < ids.size(); ++i)
    splits << (ds.row_ids.empty() ? ids[i] : ds.row_ids[ids[i]]) << ','
           << (i < n ? "train" : "test") << '\n';
  a.write("data/row_assignments.csv", splits.str());
  // Stage 10 - scree figure, metrics, manifest and report.
  std::vector<std::string> names;
  for (size_t j = 0; j < ratios.size(); ++j)
    names.push_back("PC" + std::to_string(j + 1));
  Plot p;
  p.title("Training explained variance ratios");
  p.bar(ratios, names);
  a.figure("evaluation/figures/explained_variance.svg", p);
  a.write("evaluation/metrics.json",
          "{\"components\":" + std::to_string(k) +
              ",\"training_variance_retained\":" + std::to_string(cumulative) +
              ",\"test_reconstruction_mse_standardized\":" +
              std::to_string(error) + "}\n");
  a.write("run_manifest.json",
          "{\"seed\":42,\"variance_threshold\":0.95,\"fingerprint\":" +
              json_quote(fingerprint(ds)) + "}\n");
  a.write("report.md",
          "# PCA project\n\nChoose the smallest component count retaining 95% "
          "of training variance. Both imputation/scaling and projection are "
          "fitted only on training observations. Report reconstruction error "
          "on untouched rows in standardized units. High variance does not "
          "imply predictive relevance; compare downstream performance in a "
          "training-only validation pipeline before adopting PCA.\n");
  a.write("execution.log",
          "Fitted preprocessing and PCA on training data; saved hold-out "
          "reconstruction and checked reloaded projections.\n");
  std::cout << "Saved PCA experiment under " << RUN_OUTPUT_DIR << '\n';
}
