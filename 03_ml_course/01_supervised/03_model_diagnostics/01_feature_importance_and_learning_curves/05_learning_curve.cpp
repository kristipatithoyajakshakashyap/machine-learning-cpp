// 03_ml_course/01_supervised/03_model_diagnostics/01_feature_importance_and_learning_curves/05_learning_curve.cpp
//
// Purpose : Lesson 5 - learning curve of a 50-tree random forest on wine:
//           train and 5-fold CV macro F1 at 10..100% of the training rows,
//           to show whether more data would still help.
// Inputs  : ml::load_wine(DATA_DIR).
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/05_learning_curve_results/).
// Outputs : results/05_learning_curve_results/
//             learning_curve.csv : fraction, n_train, train_score, cv_score
//             learning_curve.svg : train and cross-validation macro F1 lines
// Run     : target diag_05_learning_curve (no arguments).
//
// Lesson 5: learning curve of a random forest on wine.
#include "Forest.hpp"
#include "helper/data/datasets.hpp"
#include "helper/eval/diagnostics.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
int main() {
  try {
    std::cout << std::setprecision(4);
    auto data = ml::load_wine(DATA_DIR);
    // Fractions of each fold's training partition used for the refits.
    const std::vector<double> fractions = {0.1, 0.2, 0.4, 0.6, 0.8, 1.0};
    // Stratified 5-fold (seed 42); the factory builds a fresh forest per point.
    auto curve = ml::learning_curve([] { return course::make_forest(50); },
                                    data.X, data.y, fractions, 5, 42,
                                    ml::macro_f1, true);
    std::ostringstream csv;
    csv << "fraction,n_train,train_score,cv_score\n";
    ml::Vec n, train, cv;
    std::cout << "fraction n_train train_macro_f1 cv_macro_f1\n";
    for (size_t i = 0; i < curve.size(); ++i) {
      const auto &pt = curve[i];
      csv << fractions[i] << ',' << pt.n_train << ',' << pt.train_score << ','
          << pt.cv_score << '\n';
      std::cout << fractions[i] << ' ' << pt.n_train << ' ' << pt.train_score
                << ' ' << pt.cv_score << '\n';
      n.push_back(static_cast<double>(pt.n_train));
      train.push_back(pt.train_score);
      cv.push_back(pt.cv_score);
    }
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("learning_curve.csv", csv.str());
    ml::Plot plot;
    plot.title("Random forest learning curve on wine (5-fold, macro F1)");
    plot.xlabel("training rows");
    plot.ylabel("macro F1");
    plot.line(n, train, "train");
    plot.line(n, cv, "cross-validation");
    a.figure("learning_curve.svg", plot);
    std::cout << "Wrote " << a.path("learning_curve.csv") << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "05_learning_curve: " << e.what() << '\n';
    return 1;
  }
}
