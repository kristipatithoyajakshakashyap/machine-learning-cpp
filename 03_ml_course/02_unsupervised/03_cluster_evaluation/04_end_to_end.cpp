// 03_ml_course/02_unsupervised/03_cluster_evaluation/04_end_to_end.cpp
// Purpose: compare two clustering algorithms (k-means and Ward hierarchical)
//          on the SAME standardized Palmer Penguins rows using only internal
//          metrics, then run the full evaluation for the winning k-means fit.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json, data/, eda/,
//          validation/{candidate_scores.csv,stability.csv},
//          evaluation/{assignments.csv,metrics.json,cluster_profiles.csv,
//          figures/comparison.svg,figures/*.svg},
//          model/{model_state.txt,preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: uce_end_to_end. (No predict program: this lesson is about
//             comparing fits, not about deploying one; use ukm_predict.)
#include <iostream>

#include "../01_kmeans/KMeans.hpp"
#include "../02_agglomerative/Agglomerative.hpp"
#include "../workflow.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load (missing kept) and EDA; species in ds.y are never fitted on.
  auto ds = load_penguins(DATA_DIR, true);
  eda(ds, a);
  unsup::manifest(ds, a, "cluster comparison");
  // Stage 2 - preprocessing: median-impute + z-score, shared by both algorithms
  // so the silhouettes are computed in one identical feature space.
  Preprocessor prep;
  prep.fit(ds.X);
  auto X = prep.transform(ds.X);
  // Stage 3 - candidate grid: for k = 2..6 fit k-means (seed 42, 10 restarts)
  // and Ward agglomerative, record the mean silhouette of each.
  std::ostringstream scores;
  scores << "algorithm,k,silhouette\n";
  Vec ks, km_scores, ag_scores;
  for (size_t k = 2; k <= 6; ++k) {
    KMeans km(k);
    km.fit(X);
    AgglomerativeClustering ag(k, Linkage::Ward);
    ag.fit(X);
    double s = unsup::valid_silhouette(X, km.labels()),
           t = unsup::valid_silhouette(X, ag.labels());
    scores << "kmeans," << k << ',' << s << "\nward," << k << ',' << t << '\n';
    ks.push_back(double(k));
    km_scores.push_back(s);
    ag_scores.push_back(t);
  }
  a.write("validation/candidate_scores.csv", scores.str());
  // Stage 4 - figure: silhouette vs k, one line per algorithm.
  Plot p;
  p.title("Internal cluster evaluation on identical standardized rows");
  p.xlabel("Number of clusters");
  p.ylabel("Average silhouette");
  p.line(ks, km_scores, "K-means");
  p.line(ks, ag_scores, "Ward");
  a.figure("evaluation/figures/comparison.svg", p);
  // Stage 5 - selection: the k with the highest k-means silhouette. The index
  // into km_scores is offset by 2 because the grid started at k = 2.
  size_t k = size_t(std::max_element(km_scores.begin(), km_scores.end()) -
                    km_scores.begin()) +
             2;
  // Stage 6 - final k-means fit, evaluation tables/figures and report.
  KMeans model(k);
  model.fit(X);
  unsup::report(ds, X, model.labels(), a, "K-means k=" + std::to_string(k));
  // Stage 7 - stability: refit on five 80% subsamples and compare by ARI.
  unsup::stability(ds, model.labels(), a, [&](const Mat& z) {
    KMeans m(k);
    m.fit(z);
    return m.labels();
  });
  // Stage 8 - snapshot + Stage 9 - verify a reload reproduces the labels.
  unsup::snapshot(model, prep, a);
  KMeans loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  unsup::verify(model.labels(), loaded.predict(X), a);
  std::cout << "Saved cluster comparison under " << RUN_OUTPUT_DIR << '\n';
}
