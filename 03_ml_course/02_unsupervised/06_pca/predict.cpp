// 03_ml_course/02_unsupervised/06_pca/predict.cpp
// Purpose: project new penguin rows onto the principal axes learned by
//          upca_end_to_end, without refitting anything.
// Inputs:  argv[1] - header-less numeric CSV, one row per penguin, columns in the
//          order of model/feature_schema.csv (4 features; blanks/NA allowed).
//          ML_END_TO_END_RESULTS_DIR - results/04_end_to_end_results, where the
//          saved preprocessor and model state are read from.
// Outputs: results/predict_results/new_predictions.csv
//          (row_id,output_0..output_{k-1} = PC scores in standardized units).
// Run target: upca_predict <numeric_csv>  (ctest -R upca_new_rows).
#include <iostream>

#include "02_unsupervised/predict_input.hpp"
#include "PCA.hpp"
int main(int argc, char** argv) {
  using namespace ml;
  if (argc != 2)
    throw std::invalid_argument(
        "usage: upca_predict numeric_csv_without_header");
  Artifacts a(RUN_OUTPUT_DIR, ".");  // where new_predictions.csv is written
  // Reload the fitted state produced by the end_to_end run. The preprocessor
  // was fitted on the training partition only, so new rows are imputed and
  // standardized with the training medians/means/scales.
  Artifacts saved(ML_END_TO_END_RESULTS_DIR, ".");
  Preprocessor prep;
  archive::load_file(saved.path("model/preprocessing_state.txt"), prep);
  PCA model;
  archive::load_file(saved.path("model/model_state.txt"), model);
  auto X = prep.transform(unsup::read_numeric_rows(argv[1]));
  // One output column per kept principal component.
  Mat output;
  output = model.transform(X);
  unsup::write_predictions(output, a);
  std::cout << "Saved " << output.size() << " predictions under "
            << RUN_OUTPUT_DIR << '\n';
}
