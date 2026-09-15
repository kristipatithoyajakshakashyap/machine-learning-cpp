// 03_ml_course/01_supervised/03_model_diagnostics/02_feature_selection/04_end_to_end.cpp
//
// Purpose : Lesson 4 - the shared supervised pipeline for SelectedForest on
//           breast cancer, tuning k (features kept) over {5, 10, 20, 30} by
//           CV, then extra selection diagnostics on the training rows only:
//           the ANOVA F scores and a greedy forward-selection curve.
// Inputs  : ml::load_breast_cancer(DATA_DIR); nothing is loaded in --predict
//           mode. Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/04_end_to_end_results/ for fsel_end_to_end).
// Outputs : results/04_end_to_end_results/<full|quick>/
//             the pipeline's data/, validation/, evaluation/ and model/ trees
//             validation/filter_scores.csv       : feature, name, anova_f
//             validation/forward_selection.csv   : step, feature, name, cv F1
//             validation/figures/forward_selection.svg : CV macro F1 per step
//           --predict writes predictions.csv directly under RUN_OUTPUT_DIR.
// Run     : target fsel_end_to_end; `--quick` subsamples to 400 rows, uses
//           4 steps of 10-tree forests and writes to quick/. predict.cpp
//           #includes this file so fsel_predict shares this main() but gets
//           its own RUN_OUTPUT_DIR (results/predict_results/); use it as
//           fsel_predict --predict holdout.csv --model <run>/model.
//
// End-to-end: SelectedForest (ANOVA-F top-k + 50-tree forest) on breast
// cancer, selected over k in {5, 10, 20, 30}; then the filter scores and a
// forward-selection curve computed on the training rows only.
// Full run: executable; smoke run: executable --quick.
#include "Model.hpp"
#include "helper/eval/feature_selection.hpp"
#include "helper/pipeline/supervised.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
// True when "--quick" appears on the command line (mirrors the pipeline).
bool quick_requested(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--quick")
      return true;
  return false;
}
// Same holdout the pipeline used (stratified 80/20, seed 42) so the
// selection diagnostics never touch test rows.
// ds    : the full dataset; run : "full" or "quick" subfolder;
// quick : shrink the wrapper search (4 steps, 10 trees) for the smoke test.
// Returns 0 on success, 1 on any exception (message on stderr).
int write_selection(const ml::Dataset &ds, const std::string &run, bool quick) {
  try {
    auto outer = ml::stratified_holdout(ds.y, .8, 42);
    auto tr = ml::select_rows(ds, outer.train);
    ml::Artifacts a(RUN_OUTPUT_DIR, run);
    // Filter scores of every column (the same statistic SelectedForest uses).
    const ml::Vec scores = ml::filter_scores(tr.X, tr.y, true);
    std::ostringstream fcsv;
    fcsv << std::setprecision(10) << "feature,name,anova_f\n";
    for (size_t j = 0; j < scores.size(); ++j)
      fcsv << j << ',' << ml::csv_quote(ds.feature_names[j]) << ',' << scores[j]
           << '\n';
    a.write("validation/filter_scores.csv", fcsv.str());
    // Greedy forward selection with 3-fold macro F1; each step adds the
    // single column that improves the CV score the most.
    const size_t k_max = quick ? 4 : 8;
    auto steps = ml::forward_selection(
        [quick] { return course::make_forest(quick ? 10 : 20); }, tr.X, tr.y,
        k_max, 3, 42, ml::macro_f1, true);
    std::ostringstream wcsv;
    wcsv << std::setprecision(10) << "step,feature,name,cv_macro_f1\n";
    ml::Vec x, yv;
    for (size_t s = 0; s < steps.size(); ++s) {
      wcsv << s + 1 << ',' << steps[s].feature << ','
           << ml::csv_quote(ds.feature_names[steps[s].feature]) << ','
           << steps[s].cv_score << '\n';
      x.push_back(static_cast<double>(s + 1));
      yv.push_back(steps[s].cv_score);
    }
    a.write("validation/forward_selection.csv", wcsv.str());
    ml::Plot plot;
    plot.title("Forward selection (training rows, 3-fold macro F1)");
    plot.xlabel("features kept");
    plot.ylabel("CV macro F1");
    plot.line(x, yv, "forward");
    a.figure("validation/figures/forward_selection.svg", plot);
    std::cout << "Selection: " << a.path("validation/forward_selection.csv")
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "selection: " << e.what() << '\n';
    return 1;
  }
}
} // namespace

// Pipeline first (grid over k); the extra diagnostics only after a
// successful training run, never in --predict mode.
int main(int argc, char **argv) {
  const bool inference = ml::inference_requested(argc, argv);
  ml::Dataset ds = inference ? ml::Dataset{} : ml::load_breast_cancer(DATA_DIR);
  // Parameter p is k, the number of features SelectedForest keeps.
  int status = ml::run_supervised(
      ds, RUN_OUTPUT_DIR, "end_to_end", {5, 10, 20, 30},
      [](double p) { return course::SelectedForest(p); }, true, argc, argv);
  if (status != 0 || inference)
    return status;
  const bool quick = quick_requested(argc, argv);
  return write_selection(ds, quick ? "quick" : "full", quick);
}
