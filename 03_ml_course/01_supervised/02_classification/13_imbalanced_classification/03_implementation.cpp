// 03_ml_course/01_supervised/02_classification/13_imbalanced_classification/03_implementation.cpp
//
// Purpose : Lesson 3 - sweep the positive-class weight w in {1, 2, 4, 8} on one
//           fixed stratified split of the deliberately imbalanced breast-cancer
//           subset and record accuracy / recall / precision / macro-F1.
// Inputs  : ml::make_imbalanced(ml::load_breast_cancer(DATA_DIR), 1.0, 30, 42)
//           (all 357 benign rows + a seeded 30-row subset of malignant rows).
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR.
// Outputs : results/03_implementation_results/
//             weight_sweep.csv     : positive_weight, accuracy, recall, precision,
//                                    macro_f1 on the 25% holdout
//             recall_vs_weight.svg : holdout recall as a function of w
// Run     : target cimb_implementation (no arguments).
//
// Lesson 3: sweep the positive-class weight on the deliberately imbalanced
// breast_cancer subset and record accuracy/recall/precision/macro-F1.
#include "Model.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/preprocessing/preprocessor.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
int main() {
  try {
    std::cout << std::setprecision(4)
              << R"LESSON(WeightedLogistic stores weights and a bias, fits by full-batch gradient
descent on the weighted log-loss and clips logits before the sigmoid. The
pipeline standardises features first; this lesson does the same on a fixed
stratified split and sweeps the positive weight.)LESSON"
              << "\n\n";
    // Same construction as the end-to-end run (seed 42 -> same 30 positives).
    auto data =
        ml::make_imbalanced(ml::load_breast_cancer(DATA_DIR), 1.0, 30, 42);
    // Stratified 75/25 split (seed 7) keeps roughly 8% positives on each side.
    auto split = ml::stratified_holdout(data.y, 0.75, 7);
    auto tr = ml::select_rows(data, split.train);
    auto te = ml::select_rows(data, split.test);
    // Standardise on training rows only, then apply to both partitions.
    ml::Preprocessor prep;
    prep.fit(tr.X);
    auto Xtr = prep.transform(tr.X), Xte = prep.transform(te.X);
    std::ostringstream csv;
    csv << "positive_weight,accuracy,recall,precision,macro_f1\n";
    ml::Vec ws = {1, 2, 4, 8}, recalls;
    std::cout << "weight accuracy recall precision macro_f1\n";
    // One fit per weight; only the loss weighting changes between runs.
    for (double w : ws) {
      course::WeightedLogistic m(w);
      m.fit(Xtr, tr.y);
      auto pred = m.predict(Xte);
      auto s = ml::binary_scores(te.y, pred);
      double f1 = ml::macro_f1(te.y, pred);
      recalls.push_back(s.recall);
      std::cout << w << ' ' << s.accuracy << ' ' << s.recall << ' '
                << s.precision << ' ' << f1 << '\n';
      csv << w << ',' << s.accuracy << ',' << s.recall << ',' << s.precision
          << ',' << f1 << '\n';
    }
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("weight_sweep.csv", csv.str());
    ml::Plot plot;
    plot.title("Holdout recall versus positive-class weight");
    plot.xlabel("positive weight");
    plot.ylabel("recall");
    plot.line(ws, recalls, "recall");
    a.figure("recall_vs_weight.svg", plot);
    std::cout << "Wrote " << a.path("weight_sweep.csv") << '\n';
    return 0;
  } catch (const std::exception &e) {
    // Data loading or fitting errors are reported on stderr with exit code 1.
    std::cerr << "03_implementation: " << e.what() << '\n';
    return 1;
  }
}
