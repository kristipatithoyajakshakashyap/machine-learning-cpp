// 03_ml_course/01_supervised/03_model_diagnostics/03_hyperparameter_search/04_end_to_end.cpp
//
// Purpose : Lesson 4 - the shared supervised pipeline for TunedForest on
//           wine. Its 1-D grid {0..8} indexes the 2-D (max_depth, min_leaf)
//           grid, so the pipeline's CV is itself an exhaustive grid search.
//           Random search and nested CV then run on the training rows only,
//           so the honest (nested) and naive (best CV) scores can be compared.
// Inputs  : ml::load_wine(DATA_DIR); nothing is loaded in --predict mode.
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/04_end_to_end_results/ for hps_end_to_end).
// Outputs : results/04_end_to_end_results/<full|quick>/
//             the pipeline's data/, validation/, evaluation/ and model/ trees
//             validation/random_search.csv : draw, params, mean, std, best
//             validation/nested_cv.csv     : per outer fold + nested/naive rows
//             validation/figures/search_comparison.svg : the three curves
//           --predict writes predictions.csv directly under RUN_OUTPUT_DIR.
// Run     : target hps_end_to_end; `--quick` subsamples to 400 rows, uses
//           3 draws / 3 folds and writes to quick/. predict.cpp #includes this
//           file so hps_predict shares this main() but gets its own
//           RUN_OUTPUT_DIR (results/predict_results/); use it as
//           hps_predict --predict holdout.csv --model <run>/model.
//
// End-to-end: the pipeline's 1-D parameter grid {0..8} indexes the 2-D
// (max_depth x min_leaf) forest grid, so training-only CV IS an exhaustive
// grid search.  Afterwards random search and nested CV run on the same
// training rows for contrast.  Full run: executable; smoke: --quick.
#include "Model.hpp"
#include "helper/eval/search.hpp"
#include "helper/pipeline/supervised.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
// Named form of the grid TunedForest indexes, for the search helpers.
const ml::ParamSpace kSpace{{{"max_depth", {4, 8, 16}}, {"min_leaf", {1, 2, 5}}}};
// True when "--quick" appears on the command line (mirrors the pipeline).
bool quick_requested(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--quick")
      return true;
  return false;
}
// Factory for the search helpers: one 100-tree forest per Params.
ml::RandomForest make(const ml::Params &p) {
  return course::make_forest(
      course::TunedForest::kTrees,
      static_cast<size_t>(ml::param_value(p, "max_depth")),
      static_cast<size_t>(ml::param_value(p, "min_leaf")));
}
// Same stratified 80/20 holdout (seed 42) as the pipeline: the search below
// never touches the test rows.
// ds : the full dataset; quick : smaller search (3 draws, 3 folds) and the
// quick/ subfolder. Returns 0 on success, 1 on any exception.
int write_search(const ml::Dataset &ds, bool quick) {
  try {
    auto outer = ml::stratified_holdout(ds.y, .8, 42);
    auto tr = ml::select_rows(ds, outer.train);
    // Standardise on the training rows, as the pipeline does.
    ml::Preprocessor prep;
    prep.fit(tr.X);
    const ml::Mat X = prep.transform(tr.X);
    const size_t n_iter = quick ? 3 : 5, k = quick ? 3 : 5;
    // random: n_iter distinct draws scored by k-fold macro F1 (seed 11).
    // nested: (5|3) outer folds, 3 inner folds, n_iter inner draws each.
    const auto random = ml::random_search(make, X, tr.y, kSpace, n_iter, k, 11, ml::macro_f1, true);
    const auto nested = ml::nested_cv(make, X, tr.y, kSpace, quick ? 3 : 5, 3, n_iter, 11, ml::macro_f1, true);
    ml::Artifacts a(RUN_OUTPUT_DIR, quick ? "quick" : "full");
    std::ostringstream rcsv, ncsv;
    rcsv << std::setprecision(10) << "draw,max_depth,min_leaf,mean_score,std_score,best_so_far\n";
    ml::Vec rx, rbest;
    double best = -1e300;
    // Running best over the draws doubles as the plotted curve.
    for (size_t i = 0; i < random.size(); ++i) {
      best = std::max(best, random[i].mean);
      rcsv << i + 1 << ',' << ml::param_value(random[i].params, "max_depth") << ','
           << ml::param_value(random[i].params, "min_leaf") << ',' << random[i].mean
           << ',' << random[i].std << ',' << best << '\n';
      rx.push_back(double(i + 1));
      rbest.push_back(best);
    }
    a.write("validation/random_search.csv", rcsv.str());
    // One row per outer fold, then the honest mean/std and the naive best.
    ncsv << std::setprecision(10) << "fold,max_depth,min_leaf,inner_score,outer_score\n";
    for (const auto &f : nested.folds)
      ncsv << f.fold << ',' << ml::param_value(f.params, "max_depth") << ','
           << ml::param_value(f.params, "min_leaf") << ',' << f.inner_score << ','
           << f.outer_score << '\n';
    ncsv << "nested_mean,,," << nested.nested_mean << ',' << nested.nested_std << '\n'
         << "naive," << ml::param_value(nested.naive_params, "max_depth") << ','
         << ml::param_value(nested.naive_params, "min_leaf") << ',' << nested.naive_score << ",\n";
    a.write("validation/nested_cv.csv", ncsv.str());
    // The gap between the "naive best CV" and "nested CV mean" lines is the
    // optimistic selection bias lesson 2 estimates in closed form.
    ml::Plot fig;
    fig.title("Random search best-so-far versus nested CV (training rows)");
    fig.xlabel("random draws");
    fig.ylabel("macro F1");
    fig.line(rx, rbest, "random search best-so-far");
    fig.line({1.0, rx.back()}, {nested.nested_mean, nested.nested_mean}, "nested CV mean");
    fig.line({1.0, rx.back()}, {nested.naive_score, nested.naive_score}, "naive best CV");
    a.figure("validation/figures/search_comparison.svg", fig);
    std::cout << "Search: nested " << nested.nested_mean << " naive " << nested.naive_score
              << " -> " << a.path("validation/nested_cv.csv") << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "search: " << e.what() << '\n';
    return 1;
  }
}
} // namespace

// Pipeline first (its CV over indices 0..8 is the grid search); the extra
// searches only after a successful training run, never in --predict mode.
int main(int argc, char **argv) {
  const bool inference = ml::inference_requested(argc, argv);
  ml::Dataset ds = inference ? ml::Dataset{} : ml::load_wine(DATA_DIR);
  int status = ml::run_supervised(
      ds, RUN_OUTPUT_DIR, "end_to_end", {0, 1, 2, 3, 4, 5, 6, 7, 8},
      [](double p) { return course::TunedForest(p); }, true, argc, argv);
  if (status != 0 || inference)
    return status;
  return write_search(ds, quick_requested(argc, argv));
}
