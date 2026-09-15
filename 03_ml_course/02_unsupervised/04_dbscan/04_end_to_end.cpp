// 03_ml_course/02_unsupervised/04_dbscan/04_end_to_end.cpp
// Purpose: the complete DBSCAN experiment on Palmer Penguins: EDA, standardize,
//          choose eps / min_samples by silhouette subject to a coverage floor,
//          fit, evaluate (noise excluded), check stability, snapshot and verify.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json, data/, eda/,
//          validation/{candidate_scores.csv,selected_parameters.json,
//          stability.csv}, evaluation/{assignments.csv,metrics.json,
//          cluster_profiles.csv,figures/*.svg},
//          model/{model_state.txt,preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: udb_end_to_end (also ctest -R udb_workflow). No predict program:
//             DBSCAN labels only the rows it was fitted on.
#include <functional>
#include <iostream>

#include "../workflow.hpp"
#include "DBSCAN.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load (missing kept for median imputation) and manifest.
  auto ds = load_penguins(DATA_DIR, true);
  unsup::manifest(ds, a, "DBSCAN");
  // Stage 2 - EDA under eda/.
  eda(ds, a);
  // Stage 3 - preprocessing: median-impute + z-score. Essential here because
  // eps is a single radius applied to every feature at once.
  Preprocessor prep;
  prep.fit(ds.X);
  auto X = prep.transform(ds.X);
  // Stage 4 - model selection over 4 eps x 3 min_samples = 12 candidates.
  // Each candidate is scored by the silhouette of its NON-noise rows and by
  // coverage (fraction of rows not labelled noise). A candidate is eligible
  // only if it keeps >= 80% of the rows, otherwise a tiny eps could "win" by
  // discarding everything hard to cluster.
  double best_eps = .5, best = -2;  // -2 is below any silhouette in [-1, 1]
  size_t best_min = 5;
  std::ostringstream candidates;
  candidates << "eps,min_samples,silhouette,coverage\n";
  for (double eps : {.3, .5, .7, 1.0})
    for (size_t minimum : {size_t(4), size_t(8), size_t(12)}) {
      DBSCAN m(eps, minimum);
      m.fit(X);
      // Keep only clustered rows (label >= 0) for the silhouette.
      Mat z;
      Vec y;
      for (size_t i = 0; i < X.size(); ++i)
        if (m.labels()[i] >= 0) {
          z.push_back(X[i]);
          y.push_back(m.labels()[i]);
        }
      double s = unsup::valid_silhouette(z, y),
             coverage = double(z.size()) / X.size();
      candidates << eps << ',' << minimum << ',';
      if (std::isfinite(s)) candidates << s;  // blank when < 2 clusters
      candidates << ',' << coverage << '\n';
      if (coverage >= .8 && std::isfinite(s) && s > best) {
        best = s;
        best_eps = eps;
        best_min = minimum;
      }
    }
  // Guard: fail loudly rather than fit the default parameters silently.
  if (best == -2)
    throw std::runtime_error(
        "No valid DBSCAN candidate with at least 80 percent coverage");
  a.write("validation/candidate_scores.csv", candidates.str());
  a.write("validation/selected_parameters.json",
          "{\"eps\":" + std::to_string(best_eps) + ",\"min_samples\":" +
              std::to_string(best_min) + ",\"minimum_coverage\":0.8}\n");
  // Stage 5 - final fit with the selected parameters.
  DBSCAN model(best_eps, best_min);
  model.fit(X);
  Vec labels = model.labels();
  // Stage 6 - evaluation. report() excludes noise (-1) from the silhouette and
  // records the noise fraction in metrics.json.
  unsup::report(ds, X, labels, a,
                "eps=" + std::to_string(best_eps) +
                    ", min_samples=" + std::to_string(best_min));
  // Stage 7 - stability: refit on five 80% subsamples and compare by ARI.
  // Note that noise (-1) is treated as just another label by the ARI.
  unsup::stability(ds, labels, a, [&](const Mat& z) {
    DBSCAN m(best_eps, best_min);
    m.fit(z);
    return m.labels();
  });
  // Stage 8 - snapshot: parameters, labels, core flags and the fitted rows.
  unsup::snapshot(model, prep, a);
  // Stage 9 - verify: the reloaded object must return the identical labels.
  DBSCAN loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  unsup::verify(labels, loaded.labels(), a);

  std::cout << "Saved DBSCAN experiment under " << RUN_OUTPUT_DIR << "\n";
}
