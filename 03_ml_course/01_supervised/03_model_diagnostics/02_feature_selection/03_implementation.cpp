// 03_ml_course/01_supervised/03_model_diagnostics/02_feature_selection/03_implementation.cpp
//
// Purpose : Lesson 3 - the three families of feature selection side by side
//           on the breast-cancer training rows: filter (ANOVA F), wrapper
//           (greedy forward selection with a 20-tree forest) and embedded
//           (lasso regularisation path). Ranks from all three are tabulated.
// Inputs  : ml::load_breast_cancer(DATA_DIR), stratified 80/20 split seed 42;
//           only the training partition is used.
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/03_implementation_results/).
// Outputs : results/03_implementation_results/
//             rankings.csv : feature, name, filter score/rank, forward step,
//                            l1 score/rank
//             rankings.svg : filter rank versus l1 rank per feature
//             l1_path.csv  : lasso coefficient of every feature at every lambda
// Run     : target fsel_implementation (no arguments).
//
// Lesson 3: filter, wrapper and embedded rankings on breast cancer.
// Artifacts -> results/03_implementation_results/
//   rankings.csv  : feature, name, filter score/rank, forward step, l1 score/rank
//   rankings.svg  : filter rank versus l1 rank per feature (agreement plot)
//   l1_path.csv   : lasso coefficient of every feature at every lambda
#include "Model.hpp"
#include "helper/eval/feature_selection.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
namespace {
// Dense 1-based ranks: rank[j] = 1 for the highest score. Ties keep column
// order (stable sort). O(p log p).
std::vector<size_t> ranks(const ml::Vec &scores) {
  std::vector<size_t> order(scores.size()), rank(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return scores[a] > scores[b]; });
  for (size_t r = 0; r < order.size(); ++r)
    rank[order[r]] = r + 1;
  return rank;
}
} // namespace
int main() {
  try {
    std::cout
        << std::setprecision(4)
        << R"LESSON(filter_scores(X, y, classification) -> one score per column.
forward_selection(make, X, y, k_max, folds, seed, metric) -> k_max steps of
{feature, cv_score}. l1_path(X, y, lambdas) -> lasso slopes per lambda on
standardized columns; l1_scores ranks by the lambda at which each enters.)LESSON"
        << "\n\n";
    ml::Dataset ds = ml::load_breast_cancer(DATA_DIR);
    // Same split as the end-to-end run so the rankings are comparable.
    ml::Fold split = ml::stratified_holdout(ds.y, 0.8, 42);
    ml::Dataset tr = ml::select_rows(ds, split.train);
    const size_t p = tr.X[0].size();

    // Filter: ANOVA F of each column against the class labels.
    const ml::Vec filter = ml::filter_scores(tr.X, tr.y, true);
    const auto filter_rank = ranks(filter);
    // Wrapper: 8 greedy steps, each picking the column that most improves
    // 3-fold macro F1 of a 20-tree forest. forward_step[j] = 0 means unused.
    const size_t k_max = 8;
    auto steps = ml::forward_selection([] { return course::make_forest(20); },
                                       tr.X, tr.y, k_max, 3, 42, ml::macro_f1,
                                       true);
    std::vector<size_t> forward_step(p, 0);
    for (size_t s = 0; s < steps.size(); ++s)
      forward_step[steps[s].feature] = s + 1;
    // Embedded: lasso path from strong to weak penalty; a feature scores by
    // the largest lambda at which its coefficient is still non-zero.
    const ml::Vec lambdas = {0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    const ml::Mat path = ml::l1_path(tr.X, tr.y, lambdas);
    const ml::Vec l1 = ml::l1_scores(path, lambdas);
    const auto l1_rank = ranks(l1);

    std::ostringstream csv;
    csv << std::setprecision(10)
        << "feature,name,filter_score,filter_rank,forward_step,l1_score,l1_rank\n";
    ml::Vec fr, lr;
    for (size_t j = 0; j < p; ++j) {
      csv << j << ',' << ml::csv_quote(ds.feature_names[j]) << ',' << filter[j]
          << ',' << filter_rank[j] << ',' << forward_step[j] << ',' << l1[j]
          << ',' << l1_rank[j] << '\n';
      fr.push_back(static_cast<double>(filter_rank[j]));
      lr.push_back(static_cast<double>(l1_rank[j]));
    }
    std::cout << "top filter feature: " << ds.feature_names[ml::top_k(filter, 1)[0]]
              << "\nforward step 1: " << ds.feature_names[steps[0].feature]
              << " (cv macro F1 " << steps[0].cv_score << ")\n"
              << "top l1 feature: " << ds.feature_names[ml::top_k(l1, 1)[0]]
              << "\n";
    // l1_path.csv: one row per lambda, one column per feature (wide format).
    std::ostringstream pcsv;
    pcsv << std::setprecision(10) << "lambda";
    for (size_t j = 0; j < p; ++j)
      pcsv << ',' << ml::csv_quote(ds.feature_names[j]);
    pcsv << '\n';
    for (size_t l = 0; l < lambdas.size(); ++l) {
      pcsv << lambdas[l];
      for (double w : path[l])
        pcsv << ',' << w;
      pcsv << '\n';
    }
    // Agreement plot: points near the diagonal are ranked alike by both.
    ml::Plot fig(640, 480);
    fig.title("Filter rank versus L1 rank (breast cancer, 30 features)");
    fig.xlabel("ANOVA F rank (1 = best)");
    fig.ylabel("lasso entry rank (1 = best)");
    fig.scatter(fr, lr, "feature");
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("rankings.csv", csv.str());
    a.figure("rankings.svg", fig);
    a.write("l1_path.csv", pcsv.str());
    std::cout << "wrote " << a.path("rankings.csv") << "\nwrote "
              << a.path("rankings.svg") << "\nwrote " << a.path("l1_path.csv")
              << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "03_implementation: " << e.what() << '\n';
    return 1;
  }
}
