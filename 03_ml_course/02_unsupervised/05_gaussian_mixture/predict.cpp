// 03_ml_course/02_unsupervised/05_gaussian_mixture/predict.cpp
// Purpose: score new penguin rows with the mixture learned by ugm_end_to_end,
//          returning per-component membership probabilities (not hard labels).
// Inputs:  argv[1] - header-less numeric CSV, one row per penguin, columns in the
//          order of model/feature_schema.csv (4 features; blanks/NA allowed).
//          ML_END_TO_END_RESULTS_DIR - results/04_end_to_end_results, where the
//          saved preprocessor and model state are read from.
// Outputs: results/predict_results/new_predictions.csv
//          (row_id,output_0..output_{k-1} = responsibilities summing to 1).
// Run target: ugm_predict <numeric_csv>  (ctest -R ugm_new_rows).
#include <iostream>

#include "02_unsupervised/predict_input.hpp"
#include "GaussianMixture.hpp"
int main(int argc, char** argv) {
  using namespace ml;
  if (argc != 2)
    throw std::invalid_argument(
        "usage: ugm_predict numeric_csv_without_header");
  Artifacts a(RUN_OUTPUT_DIR, ".");  // where new_predictions.csv is written
  // Reload the fitted state produced by the end_to_end run. The preprocessor
  // carries the training medians/means/scales, so new rows are imputed and
  // standardized exactly as the training rows were; nothing is refitted.
  Artifacts saved(ML_END_TO_END_RESULTS_DIR, ".");
  Preprocessor prep;
  archive::load_file(saved.path("model/preprocessing_state.txt"), prep);
  GaussianMixture model;
  archive::load_file(saved.path("model/model_state.txt"), model);
  auto X = prep.transform(unsup::read_numeric_rows(argv[1]));
  // Soft output: one column per component; arg-max gives the hard label.
  Mat output;
  output = model.predict_proba(X);
  unsup::write_predictions(output, a);
  std::cout << "Saved " << output.size() << " predictions under "
            << RUN_OUTPUT_DIR << '\n';
}
