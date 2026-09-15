// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/03_implementation.cpp
//
// Purpose : Lesson 3 - both helpers from helper/eval/diagnostics.hpp on a
//           synthetic 3-class sample (two informative + two noise columns):
//           permutation importance of a 30-tree forest and a 4-fold learning
//           curve, each printed and written as CSV + SVG.
// Inputs  : no dataset loader; the 240-row fixture is generated here (seed 5).
//           Defines used: RUN_OUTPUT_DIR (results/03_implementation_results/).
// Outputs : results/03_implementation_results/
//             importance.csv     : feature, mean_drop, std_drop (accuracy drop)
//             importance.svg     : bar chart of mean_drop per feature
//             learning_curve.csv : n_train, train_score, cv_score (macro F1)
//             learning_curve.svg : train and cv curves against n_train
// Run     : target diag_implementation (no arguments).
//
// Lesson 3: both helpers from helper/eval/diagnostics.hpp on synthetic data.
#include "Forest.hpp"
#include "helper/eval/diagnostics.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
int main() {
  try {
    std::cout << std::setprecision(3)
              << R"LESSON(permutation_importance(model, X, y, metric, repeats, seed) returns one
{mean_drop, std_drop} per column. learning_curve(make, X, y, fractions, k,
seed, metric, stratified) returns {n_train, train_score, cv_score} per
fraction. Both take any metric(y_true, y_pred) that is higher-is-better.)LESSON"
              << "\n\n";
    // Synthetic 3-class data: two informative features, two noise features.
    // Column 0 tracks the class index, column 1 singles out class 1, columns
    // 2-3 are pure noise scaled by 3 so they look "large" but carry no signal.
    std::mt19937 rng(5);
    std::normal_distribution<double> noise(0.0, 0.6);
    ml::Mat X;
    ml::Vec y;
    for (int i = 0; i < 240; ++i) {
      const int c = i % 3; // round-robin: 80 rows per class
      X.push_back({c + noise(rng), (c == 1 ? 2.0 : 0.0) + noise(rng),
                   noise(rng) * 3, noise(rng) * 3});
      y.push_back(c);
    }
    // seed_rng fixes the forest's bootstrap/feature sampling for reproducibility.
    ml::seed_rng(7);
    auto forest = course::make_forest(30);
    forest.fit(X, y);
    // Importance = accuracy lost when one column is shuffled; 8 repeats give a
    // mean and a standard deviation per feature (seed 21 fixes the shuffles).
    std::cout << "Permutation importance (accuracy drop, 8 repeats):\n";
    auto imp = ml::permutation_importance(forest, X, y, ml::accuracy, 8, 21);
    for (size_t j = 0; j < imp.size(); ++j)
      std::cout << "  feature " << j << ": " << imp[j].mean_drop << " +- "
                << imp[j].std_drop << '\n';
    // Learning curve: refit a fresh 30-tree forest on 10/25/50/100% of each
    // fold's training rows and score train vs held-out fold with macro F1.
    std::cout << "Learning curve (macro F1, 4 stratified folds):\n";
    auto curve = ml::learning_curve([] { return course::make_forest(30); }, X, y,
                                    {0.1, 0.25, 0.5, 1.0}, 4, 9, ml::macro_f1,
                                    true);
    for (const auto &pt : curve)
      std::cout << "  n_train " << pt.n_train << ": train " << pt.train_score
                << " cv " << pt.cv_score << '\n';

    // Artifacts: importance.csv + importance.svg (bars), learning_curve.csv
    // + learning_curve.svg (train vs cv lines).
    std::ostringstream imp_csv;
    imp_csv << std::setprecision(10) << "feature,mean_drop,std_drop\n";
    ml::Vec heights;
    std::vector<std::string> names;
    for (size_t j = 0; j < imp.size(); ++j) {
      imp_csv << j << "," << imp[j].mean_drop << "," << imp[j].std_drop << "\n";
      heights.push_back(imp[j].mean_drop);
      names.push_back("f" + std::to_string(j));
    }
    ml::Plot imp_fig(640, 400);
    imp_fig.title("permutation importance (accuracy drop)");
    imp_fig.xlabel("feature"); imp_fig.ylabel("mean accuracy drop");
    imp_fig.bar(heights, names, "importance");

    std::ostringstream lc_csv;
    lc_csv << std::setprecision(10) << "n_train,train_score,cv_score\n";
    ml::Vec ns, tr, cv;
    for (const auto &pt : curve) {
      lc_csv << pt.n_train << "," << pt.train_score << "," << pt.cv_score << "\n";
      ns.push_back(static_cast<double>(pt.n_train));
      tr.push_back(pt.train_score);
      cv.push_back(pt.cv_score);
    }
    ml::Plot lc_fig(640, 400);
    lc_fig.title("learning curve (macro F1, 4 folds)");
    lc_fig.xlabel("n_train"); lc_fig.ylabel("macro F1");
    lc_fig.line(ns, tr, "train");
    lc_fig.line(ns, cv, "cv");

    // "." keeps files directly under RUN_OUTPUT_DIR (no full/quick subfolder).
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("importance.csv", imp_csv.str());
    a.figure("importance.svg", imp_fig);
    a.write("learning_curve.csv", lc_csv.str());
    a.figure("learning_curve.svg", lc_fig);
    std::cout << "wrote " << a.path("importance.csv") << "\n"
              << "wrote " << a.path("importance.svg") << "\n"
              << "wrote " << a.path("learning_curve.csv") << "\n"
              << "wrote " << a.path("learning_curve.svg") << "\n";
    return 0;
  } catch (const std::exception &e) {
    // Any helper/forest error ends the lesson with a message and exit code 1.
    std::cerr << "03_implementation: " << e.what() << '\n';
    return 1;
  }
}
