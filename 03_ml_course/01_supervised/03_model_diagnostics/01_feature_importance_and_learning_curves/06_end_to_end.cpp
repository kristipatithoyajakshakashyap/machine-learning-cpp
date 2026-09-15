// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/06_end_to_end.cpp
//
// Purpose : Lesson 6 - the shared supervised pipeline (EDA, CV over
//           n_estimators in {20, 50, 100}, holdout evaluation, model export)
//           for a random forest on wine, followed by permutation importance
//           and a learning curve computed on the training rows only.
// Inputs  : ml::load_wine(DATA_DIR); in --predict mode no dataset is loaded.
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/06_end_to_end_results/ for diag_end_to_end).
// Outputs : results/06_end_to_end_results/<full|quick>/
//             the pipeline's data/, validation/, evaluation/ and model/ trees
//             evaluation/permutation_importance.csv : feature, mean, std drop
//             evaluation/figures/learning_curve.svg  : train vs cv macro F1
//           --predict writes predictions.csv directly under RUN_OUTPUT_DIR.
// Run     : target diag_end_to_end; `--quick` subsamples to 400 rows and
//           writes to quick/. predict.cpp #includes this file so diag_predict
//           shares this main() but gets its own RUN_OUTPUT_DIR
//           (results/predict_results/); it is invoked as
//           diag_predict --predict holdout.csv --model <run>/model.
//
// End-to-end: random forest classifier on wine selected over n_estimators,
// then permutation importance and a learning curve on the training rows.
// Full run: executable; smoke run: executable --quick.
#include "Forest.hpp"
#include "helper/eval/diagnostics.hpp"
#include "helper/pipeline/supervised.hpp"
#include <iostream>
#include <sstream>
#include <string>

namespace {
// True when "--quick" appears anywhere on the command line. The pipeline
// parses the same flag; this copy only picks the diagnostics subfolder.
bool quick_requested(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--quick")
      return true;
  return false;
}
// Same holdout the pipeline used (stratified 80/20, seed 42) so the
// diagnostics never touch test rows.
// ds  : the full wine dataset (the pipeline's --quick subsample is internal);
// run : "full" or "quick", the subfolder under RUN_OUTPUT_DIR.
// Returns 0 on success, 1 if any helper throws (message on stderr).
int write_diagnostics(const ml::Dataset &ds, const std::string &run) {
  try {
    auto outer = ml::stratified_holdout(ds.y, .8, 42);
    auto tr = ml::select_rows(ds, outer.train);
    ml::seed_rng(42);
    auto forest = course::make_forest(100);
    forest.fit(tr.X, tr.y);
    // Importance measured on the rows the forest was trained on (10 repeats);
    // lesson 4 shows the out-of-fold variant.
    auto imp = ml::permutation_importance(forest, tr.X, tr.y, ml::macro_f1, 10, 7);
    std::ostringstream csv;
    csv << "feature,mean_drop,std_drop\n";
    for (size_t j = 0; j < imp.size(); ++j)
      csv << ml::csv_quote(ds.feature_names[j]) << ',' << imp[j].mean_drop << ','
          << imp[j].std_drop << '\n';
    ml::Artifacts a(RUN_OUTPUT_DIR, run);
    a.write("evaluation/permutation_importance.csv", csv.str());
    // 5-fold stratified learning curve with fresh 50-tree forests per point.
    auto curve = ml::learning_curve([] { return course::make_forest(50); },
                                    tr.X, tr.y, {0.1, 0.2, 0.4, 0.6, 0.8, 1.0},
                                    5, 42, ml::macro_f1, true);
    ml::Vec n, train, cv;
    for (const auto &pt : curve) {
      n.push_back(static_cast<double>(pt.n_train));
      train.push_back(pt.train_score);
      cv.push_back(pt.cv_score);
    }
    ml::Plot plot;
    plot.title("Training-rows learning curve (macro F1)");
    plot.xlabel("training rows");
    plot.ylabel("macro F1");
    plot.line(n, train, "train");
    plot.line(n, cv, "cross-validation");
    a.figure("evaluation/figures/learning_curve.svg", plot);
    std::cout << "Diagnostics: " << a.path("evaluation/permutation_importance.csv")
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "diagnostics: " << e.what() << '\n';
    return 1;
  }
}
} // namespace

// Runs the shared pipeline first; the extra diagnostics run only after a
// successful training run (never in --predict mode, which loads no data).
int main(int argc, char **argv) {
  const bool inference = ml::inference_requested(argc, argv);
  ml::Dataset ds = inference ? ml::Dataset{} : ml::load_wine(DATA_DIR);
  // Parameter p is n_estimators; the pipeline picks the best by CV macro F1.
  int status = ml::run_supervised(
      ds, RUN_OUTPUT_DIR, "end_to_end", {20, 50, 100},
      [](double p) { return course::make_forest(static_cast<size_t>(p)); }, true,
      argc, argv);
  if (status != 0 || inference)
    return status;
  return write_diagnostics(ds, quick_requested(argc, argv) ? "quick" : "full");
}
