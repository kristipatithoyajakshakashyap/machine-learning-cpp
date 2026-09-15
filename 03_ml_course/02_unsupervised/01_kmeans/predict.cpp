// 03_ml_course/02_unsupervised/01_kmeans/predict.cpp
// Purpose: assign new penguin rows to the clusters learned by ukm_end_to_end,
//          without refitting anything.
// Inputs:  argv[1] - header-less numeric CSV, one row per penguin, columns in the
//          order of model/feature_schema.csv (4 features; blanks/NA allowed).
//          ML_END_TO_END_RESULTS_DIR - results/04_end_to_end_results, where the
//          saved preprocessor and model state are read from.
// Outputs: results/predict_results/new_predictions.csv (row_id,output_0 = cluster).
// Run target: ukm_predict <numeric_csv>  (ctest -R ukm_new_rows).
#include <iostream>

#include "02_unsupervised/predict_input.hpp"
#include "KMeans.hpp"
int main(int argc, char** argv) {
  using namespace ml;
  if (argc != 2)
    throw std::invalid_argument(
        "usage: ukm_predict numeric_csv_without_header");
  Artifacts a(RUN_OUTPUT_DIR, ".");  // where new_predictions.csv is written
  // Reload the fitted state produced by the end_to_end run. The preprocessor
  // carries the training medians/means/scales, so new rows are imputed and
  // standardized exactly as the training rows were.
  Artifacts saved(ML_END_TO_END_RESULTS_DIR, ".");
  Preprocessor prep;
  archive::load_file(saved.path("model/preprocessing_state.txt"), prep);
  KMeans model;
  archive::load_file(saved.path("model/model_state.txt"), model);
  auto X = prep.transform(unsup::read_numeric_rows(argv[1]));
  // Nearest-centroid assignment; one output column holding the cluster id.
  Mat output;
  for (double v : model.predict(X)) output.push_back({v});
  unsup::write_predictions(output, a);
  std::cout << "Saved " << output.size() << " predictions under "
            << RUN_OUTPUT_DIR << '\n';
}
