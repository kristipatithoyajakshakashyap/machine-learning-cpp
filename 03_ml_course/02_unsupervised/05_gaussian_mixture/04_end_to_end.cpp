// 03_ml_course/02_unsupervised/05_gaussian_mixture/04_end_to_end.cpp
// Purpose: the complete Gaussian-mixture experiment on Palmer Penguins: EDA,
//          standardize, choose the component count and seed by BIC, fit by EM,
//          evaluate hard labels, check stability, snapshot and verify.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json, data/, eda/,
//          validation/{candidate_scores.csv,selected_parameters.json,
//          stability.csv}, evaluation/{assignments.csv,metrics.json,
//          cluster_profiles.csv,figures/*.svg},
//          model/{model_state.txt,preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: ugm_end_to_end (also ctest -R ugm_workflow); ugm_predict then
//             reloads model/ to score new rows.
#include <functional>
#include <iostream>

#include "../workflow.hpp"
#include "GaussianMixture.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load (missing kept for median imputation) and manifest.
  auto ds = load_penguins(DATA_DIR, true);
  unsup::manifest(ds, a, "GaussianMixture");
  // Stage 2 - EDA under eda/.
  eda(ds, a);
  // Stage 3 - preprocessing: median-impute + z-score so that the diagonal
  // variances are comparable across features.
  Preprocessor prep;
  prep.fit(ds.X);
  auto X = prep.transform(ds.X);
  // Stage 4 - model selection: k = 2..6 components x 3 seeds = 15 EM runs,
  // each up to 200 iterations. BIC (lower is better) balances fit against the
  // number of parameters; several seeds guard against a poor EM local optimum.
  size_t best_k = 2;
  unsigned best_seed = 42;
  double best = std::numeric_limits<double>::infinity();
  std::ostringstream candidates;
  candidates << "components,seed,aic,bic\n";
  for (size_t k = 2; k <= 6; ++k)
    for (unsigned seed : {42u, 43u, 44u}) {
      GaussianMixture m(k, 200, seed);
      m.fit(X);
      double b = m.bic(X);
      candidates << k << ',' << seed << ',' << m.aic(X) << ',' << b << '\n';
      if (b < best) {
        best = b;
        best_k = k;
        best_seed = seed;
      }
    }
  a.write("validation/candidate_scores.csv", candidates.str());
  a.write("validation/selected_parameters.json",
          "{\"components\":" + std::to_string(best_k) + ",\"seed\":" +
              std::to_string(best_seed) + ",\"covariance\":\"diagonal\"}\n");
  // Stage 5 - final fit; hard labels (arg-max responsibility) feed evaluation.
  GaussianMixture model(best_k, 200, best_seed);
  model.fit(X);
  Vec labels = model.predict(X);
  // Stage 6 - evaluation: assignments, silhouette figures, profiles, report.
  unsup::report(ds, X, labels, a,
                "components=" + std::to_string(best_k) +
                    ", seed=" + std::to_string(best_seed));
  // Stage 7 - stability: refit EM on five 80% subsamples and compare by ARI.
  unsup::stability(ds, labels, a, [&](const Mat& z) {
    GaussianMixture m(best_k, 200, best_seed);
    m.fit(z);
    return m.predict(z);
  });
  // Stage 8 - snapshot: means, variances, weights + preprocessor for ugm_predict.
  unsup::snapshot(model, prep, a);
  // Stage 9 - verify: the reloaded mixture must reproduce the hard labels.
  GaussianMixture loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  unsup::verify(labels, loaded.predict(X), a);

  std::cout << "Saved GaussianMixture experiment under " << RUN_OUTPUT_DIR
            << "\n";
}
