// 03_ml_course/03_anomaly_detection/03_one_class_svm/predict.cpp
// Purpose: score new penguin rows with the boundary fitted by
//          uocsvm_end_to_end, without refitting anything.
// Inputs:  argv[1] - header-less numeric CSV, one row per penguin, columns in the
//          order of model/feature_schema.csv (4 features; blanks/NA allowed).
//          ML_END_TO_END_RESULTS_DIR - results/04_end_to_end_results, where the
//          saved preprocessor and model state are read from.
// Outputs: results/predict_results/new_predictions.csv
//          (row_id, output_0 = rho - decision value, output_1 = flag 0/1).
// Run target: uocsvm_predict <numeric_csv>  (ctest -R uocsvm_new_rows).
#include <iostream>

#include "02_unsupervised/predict_input.hpp"
#include "OneClassSVM.hpp"
int main(int argc, char** argv) {
  using namespace ml;
  if (argc != 2)
    throw std::invalid_argument(
        "usage: uocsvm_predict numeric_csv_without_header");
  Artifacts a(RUN_OUTPUT_DIR, ".");  // where new_predictions.csv is written
  // Reload the fitted state produced by the end_to_end run. The preprocessor
  // carries the training medians/means/scales, so new rows are imputed and
  // standardized exactly as the training rows were; the SVM snapshot carries
  // the support vectors, their multipliers, rho, gamma and the threshold.
  Artifacts saved(ML_END_TO_END_RESULTS_DIR, ".");
  Preprocessor prep;
  archive::load_file(saved.path("model/preprocessing_state.txt"), prep);
  OneClassSVM model;
  archive::load_file(saved.path("model/model_state.txt"), model);
  auto X = prep.transform(unsup::read_numeric_rows(argv[1]));
  // Two output columns per row: the score and the thresholded flag.
  Mat output;
  auto scores = model.score_samples(X);
  auto flags = model.predict(X);
  for (size_t i = 0; i < scores.size(); ++i)
    output.push_back({scores[i], flags[i]});
  unsup::write_predictions(output, a);
  std::cout << "Saved " << output.size() << " predictions under "
            << RUN_OUTPUT_DIR << '\n';
}
