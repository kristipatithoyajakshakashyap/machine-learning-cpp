// 03_ml_course/02_unsupervised/07_tsne/04_end_to_end.cpp
// Purpose: the complete t-SNE experiment on Palmer Penguins: impute and
//          standardize all rows, embed them in 2-D for three perplexities,
//          select the perplexity by trustworthiness, snapshot the state and
//          prove a reloaded model refits to bit-identical coordinates.
// Inputs:  DATA_DIR/penguins.csv (344 rows, 4 numeric features, missing kept).
// Outputs: results/04_end_to_end_results/ with run_manifest.json,
//          data/row_assignments.csv, eda/,
//          validation/{candidate_scores.csv,selection_rule.md},
//          evaluation/{embedding.csv,metrics.json,figures/embedding.svg},
//          model/{model_state.txt,preprocessing_state.txt,feature_schema.csv},
//          inference/reload_verification.json, report.md, execution.log.
// Run target: utsne_end_to_end (also ctest -R utsne_workflow). No predict
//             program: t-SNE has no out-of-sample transform.
#include <iostream>

#include "../workflow.hpp"
#include "TSNE.hpp"
int main() {
  using namespace ml;
  // kNeighbourhood: k for trustworthiness; kIterations: gradient steps per
  // fit; the perplexity grid spans "local" (5) to "global" (30) structure.
  constexpr size_t kNeighbourhood = 10;
  constexpr size_t kIterations = 500;
  const std::vector<double> perplexities{5, 15, 30};
  // RUN_OUTPUT_DIR is injected by CMake (results/04_end_to_end_results).
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Stage 1 - load (missing kept) and record the feature order of the
  // saved preprocessor.
  auto ds = load_penguins(DATA_DIR, true);
  std::ostringstream schema;
  schema << "feature\n";
  for (const auto& name : ds.feature_names) schema << csv_quote(name) << '\n';
  a.write("model/feature_schema.csv", schema.str());
  // t-SNE has no out-of-sample map, so every row is embedded; there is no
  // hold-out partition.  Preprocessing is fitted on the same rows.
  // Stage 2 - EDA; Stage 3 - median-impute + z-score on all rows.
  eda(ds, a);
  Preprocessor prep;
  prep.fit(ds.X);
  auto X = prep.transform(ds.X);
  // Stage 4 - fit one embedding per perplexity (seed 42 inside TSNE) and score
  // each by trustworthiness: how well the 10 nearest neighbours in the 2-D
  // map match the 10 nearest neighbours in standardized feature space.
  std::vector<TSNE> candidates;
  std::ostringstream scores;
  scores << std::setprecision(17)
         << "perplexity,kl,trustworthiness,selected\n";
  Vec trust;
  for (double perplexity : perplexities) {
    candidates.emplace_back(2, perplexity, kIterations);
    candidates.back().fit(X);
    trust.push_back(
        trustworthiness(X, candidates.back().embedding(), kNeighbourhood));
  }
  // Stage 5 - selection: highest trustworthiness wins (first on ties). KL is
  // logged but NOT used, because P differs per perplexity.
  size_t best = 0;
  for (size_t c = 1; c < trust.size(); ++c)
    if (trust[c] > trust[best]) best = c;
  for (size_t c = 0; c < candidates.size(); ++c)
    scores << perplexities[c] << ',' << candidates[c].kl_divergence() << ','
           << trust[c] << ',' << (c == best) << '\n';
  a.write("validation/candidate_scores.csv", scores.str());
  a.write("validation/selection_rule.md",
          "Select the perplexity with the highest trustworthiness (k = 10 "
          "neighbourhood preservation between standardized features and the "
          "embedding). The final KL divergence is saved for every candidate "
          "but is not the selection metric: P changes with the perplexity, so "
          "KL values of different perplexities measure different objectives "
          "and are not strictly comparable. Species labels never enter "
          "fitting or selection.\n");
  // Stage 6 - snapshot the selected model (parameters, seed, coordinates).
  TSNE& model = candidates[best];
  const Mat Y = model.embedding();
  unsup::snapshot(model, prep, a);
  // Stage 7 - verify. Reload the fitted state and recompute the embedding
  // from the same rows: the saved seed and parameters must reproduce the
  // coordinates exactly. Both n x 2 embeddings are flattened row-major so
  // the shared verify() can compare them as vectors.
  TSNE loaded;
  archive::load_file(a.path("model/model_state.txt"), loaded);
  loaded.fit(X);
  const Mat again = loaded.embedding();
  Vec flat_original, flat_reloaded;
  for (size_t i = 0; i < Y.size(); ++i)
    for (size_t d = 0; d < Y[i].size(); ++d) {
      flat_original.push_back(Y[i][d]);
      flat_reloaded.push_back(again[i][d]);
    }
  unsup::verify(flat_original, flat_reloaded, a);
  // Stage 8 - tables: coordinates per row (species only as a reading aid) and
  // a partition file recording that every row was embedded.
  std::ostringstream csv, splits;
  csv << std::setprecision(17) << "row_id,dim1,dim2,external_species\n";
  for (size_t i = 0; i < Y.size(); ++i)
    csv << (ds.row_ids.empty() ? i : ds.row_ids[i]) << ',' << Y[i][0] << ','
        << Y[i][1] << ',' << ds.y[i] << '\n';
  a.write("evaluation/embedding.csv", csv.str());
  splits << "row_id,partition\n";
  for (size_t i = 0; i < ds.n(); ++i)
    splits << (ds.row_ids.empty() ? i : ds.row_ids[i]) << ",embedded\n";
  a.write("data/row_assignments.csv", splits.str());
  // Stage 9 - figure coloured by species (never used in fitting).
  Plot scatter;
  scatter.title("t-SNE embedding of penguins (colour = species, not used in fitting)");
  scatter.xlabel("dim1");
  scatter.ylabel("dim2");
  scatter.class_labels({"Adelie", "Chinstrap", "Gentoo"});
  scatter.scatter(Y, ds.y, "species");
  a.figure("evaluation/figures/embedding.svg", scatter);
  // Stage 10 - metrics, manifest, report and log.
  std::ostringstream metrics;
  metrics << std::setprecision(17) << "{\"perplexity\":" << perplexities[best]
          << ",\"iterations\":" << kIterations
          << ",\"kl_divergence\":" << model.kl_divergence()
          << ",\"trustworthiness_k10\":" << trust[best]
          << ",\"rows_embedded\":" << Y.size() << "}\n";
  a.write("evaluation/metrics.json", metrics.str());
  a.write("run_manifest.json",
          "{\"algorithm\":\"t-SNE\",\"seed\":42,\"perplexity_grid\":[5,15,30],"
          "\"selection\":\"trustworthiness k=10\",\"fingerprint\":" +
              json_quote(fingerprint(ds)) + "}\n");
  a.write("report.md",
          "# t-SNE project\n\nAll " + std::to_string(Y.size()) +
              " penguins are embedded into two coordinates after imputation "
              "and standardization. Three perplexities were compared and the "
              "one with the highest trustworthiness (k = 10) was kept; KL "
              "divergence is reported per candidate but is not comparable "
              "across perplexities because the target distribution P itself "
              "changes. The figure colours points by species for reading "
              "only: species never entered fitting or selection. Cluster "
              "sizes, distances between clusters and axis directions in a "
              "t-SNE plot carry no quantitative meaning, and the layout "
              "depends on the seed. There is no out-of-sample transform, so "
              "no predict executable exists; the saved state (parameters, "
              "seed, coordinates) reproduces this embedding exactly when "
              "refitted on the same rows, which the reload check "
              "verifies.\n");
  a.write("execution.log",
          "Fitted preprocessing on all rows; embedded three perplexity "
          "candidates; selected by trustworthiness; saved state; refitted "
          "from the reloaded state and compared coordinates.\n");
  std::cout << "Saved t-SNE experiment under " << RUN_OUTPUT_DIR << '\n';
}
