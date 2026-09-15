// 03_ml_course/02_unsupervised/02_agglomerative/04_end_to_end.cpp
// Purpose: the complete hierarchical-clustering experiment on Palmer Penguins:
//          EDA, standardize, pick linkage + k by silhouette, fit, evaluate,
//          check stability, save the fitted tree, reload it and draw a dendrogram.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json, data/, eda/,
//          validation/{candidate_scores.csv,selected_parameters.json,
//          stability.csv}, evaluation/{assignments.csv,metrics.json,
//          cluster_profiles.csv,linkage.csv,figures/*.svg,figures/dendrogram.svg},
//          model/{model_state.txt,preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: uag_end_to_end (also ctest -R uag_workflow).
#include <functional>
#include <iostream>

#include "../workflow.hpp"
#include "Agglomerative.hpp"
int main() {
  using namespace ml;
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load: keep missing values so the preprocessor learns median
  // imputation; species labels stay in ds.y for external evaluation only.
  auto ds = load_penguins(DATA_DIR, true);
  unsup::manifest(ds, a, "Agglomerative");
  // Stage 2 - EDA: descriptive statistics, histograms, correlations under eda/.
  eda(ds, a);
  // Stage 3 - preprocessing: median-impute then z-score so that linkage
  // distances are not dominated by body_mass_g (grams vs millimetres).
  Preprocessor prep;
  prep.fit(ds.X);
  auto X = prep.transform(ds.X);
  // Stage 4 - model selection over 4 linkages x k = 2..6 (20 candidates).
  // One fit per linkage builds the whole tree; cut(k) then reads off any k
  // without refitting, which is the main practical advantage of hierarchies.
  size_t best_k = 2;
  Linkage best_link = Linkage::Average;
  double best = -2;  // silhouette lies in [-1, 1], so -2 is below any value
  std::ostringstream candidates;
  candidates << "linkage,k,silhouette\n";
  for (Linkage l :
       {Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward}) {
    AgglomerativeClustering m(2, l);  // n_clusters only affects labels()
    m.fit(X);
    for (size_t k = 2; k <= 6; ++k) {
      double s = unsup::valid_silhouette(X, m.cut(k));
      // Linkage is logged as its enum index: 0 single, 1 complete, 2 average,
      // 3 ward (see Linkage in Agglomerative.hpp).
      candidates << int(l) << ',' << k << ',' << s << '\n';
      if (s > best) {
        best = s;
        best_k = k;
        best_link = l;
      }
    }
  }
  a.write("validation/candidate_scores.csv", candidates.str());
  a.write("validation/selected_parameters.json",
          "{\"k\":" + std::to_string(best_k) +
              ",\"linkage\":" + std::to_string(int(best_link)) + "}\n");
  // Stage 5 - final fit with the selected linkage and k on the full data.
  AgglomerativeClustering model(best_k, best_link);
  model.fit(X);
  Vec labels = model.labels();
  // Stage 6 - evaluation: assignments, silhouette figures, profiles, report.
  unsup::report(ds, X, labels, a,
                "k=" + std::to_string(best_k) +
                    ", linkage=" + std::to_string(int(best_link)));
  // Stage 7 - stability: rebuild the tree on five 80% subsamples, compare by ARI.
  unsup::stability(ds, labels, a, [&](const Mat& z) {
    AgglomerativeClustering m(best_k, best_link);
    m.fit(z);
    return m.labels();
  });
  // Stage 8 - snapshot: the saved state is the merge tree plus fitted labels.
  unsup::snapshot(model, prep, a);
  // Stage 9 - verify: a reloaded tree must reproduce the identical labels.
  // (There is no predict() for new rows; hierarchies only label fitted rows.)
  AgglomerativeClustering loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  unsup::verify(labels, loaded.labels(), a);
  // Stage 10 - export the full linkage matrix (scipy-style: one merge per row,
  // children are leaf ids < n or internal node ids n + row index).
  std::ostringstream linkage;
  linkage << "child_a,child_b,height,count\n";
  for (const auto& r : model.linkage_matrix())
    linkage << r[0] << ',' << r[1] << ',' << r[2] << ',' << r[3] << '\n';
  a.write("evaluation/linkage.csv", linkage.str());
  // Stage 11 - dendrogram. Leaves get x positions in depth-first tree order so
  // that no branch crosses another; an internal node sits midway between its
  // two children at y = merge height. Node ids: 0..n-1 leaves, n.. merges.
  const auto& tree = model.linkage_matrix();
  Vec pos(2 * X.size() - 1), height(2 * X.size() - 1, 0);
  size_t leaf = 0;
  std::function<void(size_t)> visit = [&](size_t u) {
    if (u < X.size()) {  // leaf: next free slot on the x axis
      pos[u] = double(leaf++);
      return;
    }
    const auto& r = tree[u - X.size()];
    size_t l = size_t(r[0]), v = size_t(r[1]);
    visit(l);
    visit(v);
    pos[u] = .5 * (pos[l] + pos[v]);
    height[u] = r[2];
  };
  visit(2 * X.size() - 2);  // the root is the last merge
  Plot dendrogram(1200, 500);
  dendrogram.title("Complete hierarchy (leaves in tree order)");
  dendrogram.xlabel("Observation leaf position");
  dendrogram.ylabel("Linkage distance");
  // Each merge is drawn as a "U": two vertical risers joined by a horizontal bar.
  for (size_t t = 0; t < tree.size(); ++t) {
    size_t l = size_t(tree[t][0]), r = size_t(tree[t][1]);
    double h = tree[t][2];
    dendrogram.line({pos[l], pos[l]}, {height[l], h});
    dendrogram.line({pos[l], pos[r]}, {h, h});
    dendrogram.line({pos[r], pos[r]}, {height[r], h});
  }
  a.figure("evaluation/figures/dendrogram.svg", dendrogram);
  std::cout << "Saved Agglomerative experiment under " << RUN_OUTPUT_DIR
            << "\n";
}
