// 03_ml_course/02_unsupervised/01_kmeans/04_end_to_end.cpp
// Purpose: the complete k-means experiment on Palmer Penguins: EDA, standardize,
//          choose k by silhouette, fit, evaluate, check stability, save the
//          fitted state and prove it reloads bit-for-bit.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json, data/, eda/,
//          validation/{candidate_scores.csv,selected_parameters.json,
//          stability.csv}, evaluation/{assignments.csv,metrics.json,
//          cluster_profiles.csv,figures/*.svg}, model/{model_state.txt,
//          preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: ukm_end_to_end (also ctest -R ukm_workflow).
#include <functional>
#include <iostream>

#include "../workflow.hpp"
#include "KMeans.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load: keep missing values (true) so the preprocessor learns
  // median imputation; species labels stay in ds.y for external evaluation only.
  auto ds = load_penguins(DATA_DIR, true);
  unsup::manifest(ds, a, "KMeans");
  // Stage 2 - EDA: descriptive statistics, histograms, correlations under eda/.
  eda(ds, a);
  // Stage 3 - preprocessing: median-impute then z-score every feature so that
  // Euclidean distance is not dominated by body_mass_g (grams vs millimetres).
  Preprocessor prep;
  prep.fit(ds.X);
  auto X = prep.transform(ds.X);
  // Stage 4 - model selection: try k = 2..6 and keep the highest silhouette.
  // Species labels are NOT used here; this is purely internal validation.
  size_t best_k = 2;
  double best = -2;  // silhouette lies in [-1, 1], so -2 is below any value
  std::ostringstream candidates;
  candidates << "k,silhouette,inertia\n";
  for (size_t k = 2; k <= 6; ++k) {
    KMeans m(k);  // defaults: n_init 10, max_iter 300, seed 42
    m.fit(X);
    double s = unsup::valid_silhouette(X, m.labels());
    candidates << k << ',' << s << ',' << m.inertia() << '\n';
    if (s > best) {
      best = s;
      best_k = k;
    }
  }
  a.write("validation/candidate_scores.csv", candidates.str());
  a.write("validation/selected_parameters.json",
          "{\"k\":" + std::to_string(best_k) + ",\"seed\":42}\n");
  // Stage 5 - final fit with the selected k on the full standardized data.
  KMeans model(best_k);
  model.fit(X);
  Vec labels = model.labels();
  // Stage 6 - evaluation: assignments, silhouette figures, profiles, report.
  unsup::report(ds, X, labels, a, "k=" + std::to_string(best_k));
  // Stage 7 - stability: refit on five 80% subsamples and compare by ARI.
  unsup::stability(ds, labels, a, [&](const Mat& z) {
    KMeans m(best_k);
    m.fit(z);
    return m.labels();
  });
  // Stage 8 - snapshot: save model + preprocessor for ukm_predict.
  unsup::snapshot(model, prep, a);
  // Stage 9 - verify: reload from disk and require identical labels (<= 1e-10).
  KMeans loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  unsup::verify(labels, loaded.predict(X), a);

  std::cout << "Saved KMeans experiment under " << RUN_OUTPUT_DIR << "\n";
}
