// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/04_permutation_importance.cpp
//
// Purpose : Lesson 4 - out-of-fold permutation importance of a 100-tree random
//           forest on wine. One pure-noise column is appended; the lesson
//           checks that it ranks below every feature with a detectable drop
//           and returns exit code 1 if it does not.
// Inputs  : ml::load_wine(DATA_DIR) plus one appended N(0,1) column (seed 99).
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/04_permutation_importance_results/).
// Outputs : results/04_permutation_importance_results/
//             permutation_importance.csv : rank, feature, mean_drop, std_drop
//             permutation_importance.svg : bar chart of the ranked drops
// Run     : target diag_04_permutation_importance (no arguments).
//
// Lesson 4: permutation importance of a random forest on wine, with one
// appended pure-noise feature that must rank last.
#include "Forest.hpp"
#include "helper/data/datasets.hpp"
#include "helper/eval/diagnostics.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/reporting/artifacts.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
int main() {
  try {
    std::cout << std::setprecision(4);
    auto data = ml::load_wine(DATA_DIR);
    // Append a 14th column of pure noise; a sound importance measure must put
    // it at (statistically) zero.
    std::mt19937 rng(99);
    std::normal_distribution<double> noise(0.0, 1.0);
    for (auto &row : data.X)
      row.push_back(noise(rng));
    data.feature_names.push_back("pure_noise");
    // Out-of-fold importance: fit on each fold's training rows and shuffle
    // columns of the held-out rows, then average the drops over the folds.
    // A single 45-row holdout is too small to rank fourteen features.
    const size_t k = 5, repeats = 15;
    auto folds = ml::stratified_kfold(data.y, k, 11);
    std::vector<ml::ImportanceRow> imp(data.p());
    double holdout_accuracy = 0;
    for (size_t f = 0; f < folds.size(); ++f) {
      auto tr = ml::select_rows(data, folds[f].train);
      auto te = ml::select_rows(data, folds[f].test);
      // Distinct forest seed per fold so the trees differ between folds.
      ml::seed_rng(42 + static_cast<unsigned>(f));
      auto forest = course::make_forest(100);
      forest.fit(tr.X, tr.y);
      holdout_accuracy += ml::accuracy(te.y, forest.predict(te.X)) / k;
      auto fold_imp = ml::permutation_importance(forest, te.X, te.y, ml::macro_f1,
                                                 repeats, 7 + static_cast<unsigned>(f));
      // Running mean over folds (divide by k as we go instead of at the end).
      for (size_t j = 0; j < imp.size(); ++j) {
        imp[j].mean_drop += fold_imp[j].mean_drop / k;
        imp[j].std_drop += fold_imp[j].std_drop / k;
      }
    }
    std::cout << "Mean out-of-fold accuracy: " << holdout_accuracy << '\n';
    // order[r] = index of the feature with the r-th largest mean drop.
    std::vector<size_t> order(imp.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return imp[a].mean_drop > imp[b].mean_drop;
    });
    std::ostringstream csv;
    csv << "rank,feature,mean_drop,std_drop\n";
    ml::Vec heights;
    std::vector<std::string> labels;
    for (size_t r = 0; r < order.size(); ++r) {
      const size_t j = order[r];
      csv << r + 1 << ',' << ml::csv_quote(data.feature_names[j]) << ','
          << imp[j].mean_drop << ',' << imp[j].std_drop << '\n';
      std::cout << std::setw(2) << r + 1 << ' ' << std::setw(30)
                << data.feature_names[j] << ' ' << imp[j].mean_drop << " +- "
                << imp[j].std_drop << '\n';
      heights.push_back(imp[j].mean_drop);
      labels.push_back(data.feature_names[j]);
    }
    // Several real wine features are never consulted by the forest and tie
    // with the noise column at (statistically) zero, so the check is: the
    // noise drop is within two standard deviations of zero and ranks below
    // every feature whose drop is detectable (> kDetectable).
    const double kDetectable = 0.01;
    const size_t noise_index = imp.size() - 1;
    const auto &noise_row = imp[noise_index];
    bool noise_last =
        std::abs(noise_row.mean_drop) <= 2 * noise_row.std_drop + 1e-12;
    for (size_t j = 0; j < noise_index; ++j)
      if (imp[j].mean_drop > kDetectable &&
          imp[j].mean_drop <= noise_row.mean_drop)
        noise_last = false;
    std::cout << "pure_noise ranks " << (noise_last ? "last" : "NOT last")
              << " among features with a detectable drop\n";
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("permutation_importance.csv", csv.str());
    ml::Plot plot(900, 420);
    plot.title("Permutation importance (out-of-fold macro F1 drop)");
    plot.ylabel("mean drop");
    plot.bar(heights, labels);
    a.figure("permutation_importance.svg", plot);
    std::cout << "Wrote " << a.path("permutation_importance.csv") << '\n';
    // The lesson doubles as a sanity check: a misranked noise column fails it.
    return noise_last ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "04_permutation_importance: " << e.what() << '\n';
    return 1;
  }
}
