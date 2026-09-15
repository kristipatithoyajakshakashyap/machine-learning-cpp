// 03_ml_course/01_supervised/03_model_diagnostics/03_hyperparameter_search/03_implementation.cpp
//
// Purpose : Lesson 3 - grid search, random search and nested CV from
//           helper/eval/search.hpp on the full (standardised) wine dataset,
//           all tuning the same 3 x 3 (max_depth, min_leaf) forest grid with
//           macro F1, and a plot of best-so-far score versus evaluations.
// Inputs  : ml::load_wine(DATA_DIR), standardised with ml::standardize.
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/03_implementation_results/).
// Outputs : results/03_implementation_results/
//             grid_search.csv       : every (max_depth, min_leaf) with CV mean/std
//             random_search.csv     : the sampled subset, in draw order
//             nested_cv.csv         : per outer fold selected params + outer score
//             search_comparison.svg : best-so-far CV macro F1 versus evaluations
// Run     : target hps_implementation (no arguments).
#include "Model.hpp"
#include "helper/eval/search.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
// Artifacts -> results/03_implementation_results/
//   grid_search.csv       : every (max_depth, min_leaf) with CV mean/std
//   random_search.csv     : the sampled subset, in draw order
//   nested_cv.csv         : per outer fold selected params + outer score
//   search_comparison.svg : best-so-far CV macro F1 versus evaluations
namespace {
// The same 3 x 3 grid TunedForest indexes; here it is searched by name.
const ml::ParamSpace kSpace{{{"max_depth", {4, 8, 16}}, {"min_leaf", {1, 2, 5}}}};
// Factory handed to the search helpers: one 100-tree forest per Params.
ml::RandomForest make(const ml::Params &p) {
  return course::make_forest(
      course::TunedForest::kTrees,
      static_cast<size_t>(ml::param_value(p, "max_depth")),
      static_cast<size_t>(ml::param_value(p, "min_leaf")));
}
// One CSV line per evaluated configuration in visit order, with a running
// best_so_far column so the curve can be plotted straight from the file.
std::string rows_csv(const std::vector<ml::SearchRow> &rows) {
  std::ostringstream out;
  out << std::setprecision(10) << "evaluation,max_depth,min_leaf,mean_score,std_score,best_so_far\n";
  double best = -1e300;
  for (size_t i = 0; i < rows.size(); ++i) {
    best = std::max(best, rows[i].mean);
    out << i + 1 << ',' << ml::param_value(rows[i].params, "max_depth") << ','
        << ml::param_value(rows[i].params, "min_leaf") << ',' << rows[i].mean
        << ',' << rows[i].std << ',' << best << '\n';
  }
  return out.str();
}
// Running maximum of the mean CV score, i.e. the y-values of the plot.
ml::Vec best_curve(const std::vector<ml::SearchRow> &rows) {
  ml::Vec out;
  double best = -1e300;
  for (const auto &r : rows) {
    best = std::max(best, r.mean);
    out.push_back(best);
  }
  return out;
}
} // namespace
int main() {
  try {
    std::cout
        << R"LESSON(Grid search visits all nine (max_depth, min_leaf) forests, random search visits a seeded subset without replacement, and nested CV wraps an inner random search inside outer folds so that the reported score never saw the selection.)LESSON"
        << "\n";
    const ml::Dataset wine = ml::load_wine(DATA_DIR);
    const ml::Mat X = ml::standardize(wine.X);
    // grid: all 9 cells, 5 stratified folds, seed 7.
    // random: 5 distinct draws from the 9 cells, same folds/seed.
    // nested: 3 outer x 3 inner folds, 5 inner draws per outer fold.
    const auto grid = ml::grid_search(make, X, wine.y, kSpace, 5, 7, ml::macro_f1, true);
    const auto random = ml::random_search(make, X, wine.y, kSpace, 5, 5, 7, ml::macro_f1, true);
    const auto nested = ml::nested_cv(make, X, wine.y, kSpace, 3, 3, 5, 7, ml::macro_f1, true);
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("grid_search.csv", rows_csv(grid));
    a.write("random_search.csv", rows_csv(random));
    // nested_cv.csv: one row per outer fold, then a summary row with the
    // honest mean/std and a "naive" row with the plain (optimistic) best CV.
    std::ostringstream ncsv;
    ncsv << std::setprecision(10) << "fold,max_depth,min_leaf,inner_score,outer_score\n";
    for (const auto &f : nested.folds)
      ncsv << f.fold << ',' << ml::param_value(f.params, "max_depth") << ','
           << ml::param_value(f.params, "min_leaf") << ',' << f.inner_score
           << ',' << f.outer_score << '\n';
    ncsv << "nested_mean,,," << nested.nested_mean << ',' << nested.nested_std << '\n'
         << "naive," << ml::param_value(nested.naive_params, "max_depth") << ','
         << ml::param_value(nested.naive_params, "min_leaf") << ',' << nested.naive_score << ",\n";
    a.write("nested_cv.csv", ncsv.str());
    // Plot: grid and random best-so-far curves plus the nested estimate as a
    // horizontal reference line spanning the grid's evaluation count.
    ml::Plot fig;
    fig.title("Best-so-far CV macro F1 versus evaluations (wine)");
    fig.xlabel("model evaluations (x5 folds)");
    fig.ylabel("macro F1");
    ml::Vec gx(grid.size()), rx(random.size());
    for (size_t i = 0; i < gx.size(); ++i) gx[i] = double(i + 1);
    for (size_t i = 0; i < rx.size(); ++i) rx[i] = double(i + 1);
    fig.line(gx, best_curve(grid), "grid (exhaustive)");
    fig.line(rx, best_curve(random), "random (5 draws)");
    fig.line({1.0, gx.back()}, {nested.nested_mean, nested.nested_mean}, "nested CV estimate");
    a.figure("search_comparison.svg", fig);
    const auto &gb = ml::best_row(grid);
    std::cout << std::setprecision(4) << "grid best " << ml::params_to_string(gb.params)
              << " F1 " << gb.mean << "\nrandom best "
              << ml::params_to_string(ml::best_row(random).params) << " F1 "
              << ml::best_row(random).mean << "\nnested honest F1 " << nested.nested_mean
              << " +/- " << nested.nested_std << " vs naive " << nested.naive_score << "\n"
              << "wrote " << a.path("search_comparison.svg") << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "implementation: " << e.what() << '\n';
    return 1;
  }
}
