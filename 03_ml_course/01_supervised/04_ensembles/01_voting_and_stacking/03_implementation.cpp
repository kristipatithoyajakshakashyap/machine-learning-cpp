// 03_ml_course/01_supervised/04_ensembles/01_voting_and_stacking/03_implementation.cpp
//
// Purpose : Lesson 3 - every member (logistic, 5-NN, 50-tree forest) and
//           every ensemble mode (hard, soft, stacking) scored on one fixed
//           stratified wine holdout, plus the stacking meta-learner's weights.
// Inputs  : ml::load_wine(DATA_DIR), stratified 80/20 split seed 42,
//           standardised on the training rows.
//           Defines used: DATA_DIR, RUN_OUTPUT_DIR
//           (results/03_implementation_results/).
// Outputs : results/03_implementation_results/
//             comparison.csv   : model, accuracy, macro_f1 on the holdout
//             macro_f1.svg     : bar chart of holdout macro-F1
//             meta_weights.csv : stacking meta-learner weights
//                                (meta_class, base, base_class, weight)
// Run     : target ens_implementation (no arguments).
//
// Lesson 3: each base learner and each ensemble mode on one fixed wine split.
// Artifacts -> results/03_implementation_results/
//   comparison.csv    : model, accuracy, macro_f1 on the holdout
//   macro_f1.svg      : bar chart of holdout macro-F1
//   meta_weights.csv  : stacking meta-learner weights (class x base x class)
#include "Model.hpp"
#include "helper/pipeline/supervised.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
namespace {
// One line of the comparison table.
struct Row {
  std::string name;
  double accuracy, f1;
};
// Score any model with predict(X) on the holdout; M is a member or ensemble.
template <class M>
Row score(const std::string &name, const M &m, const ml::Mat &X,
          const ml::Vec &y) {
  const ml::Vec yp = m.predict(X);
  return {name, ml::accuracy(y, yp), ml::macro_f1(y, yp)};
}
} // namespace
int main() {
  try {
    std::cout
        << std::setprecision(4)
        << R"LESSON(VotingStacking(mode) owns three members. fit() trains them on the full training rows; in stacking mode it first builds out-of-fold member probabilities with 5 stratified folds and fits a multinomial logistic meta-learner on them. predict_proba() returns vote fractions, the probability average, or the meta-learner output.)LESSON"
        << "\n\n";
    ml::Dataset ds = ml::load_wine(DATA_DIR);
    ml::Fold split = ml::stratified_holdout(ds.y, 0.8, 42);
    ml::Dataset tr = ml::select_rows(ds, split.train),
                te = ml::select_rows(ds, split.test);
    // Standardise on training rows only (5-NN and logistic need it).
    ml::Preprocessor prep;
    prep.fit(tr.X);
    const ml::Mat Xtr = prep.transform(tr.X), Xte = prep.transform(te.X);
    std::vector<Row> rows;
    const std::vector<std::string> modes = {"hard_voting", "soft_voting",
                                            "stacking"};
    course::VotingStacking stack(course::VotingStacking::kStacking);
    // One ensemble per mode, each re-seeded so the forests are identical
    // across modes; the members are scored once, from the hard-voting fit.
    for (int mode = 0; mode < 3; ++mode) {
      ml::seed_rng(42);
      course::VotingStacking ens(mode);
      ens.fit(Xtr, tr.y);
      if (mode == 0) {
        rows.push_back(score("logistic", ens.logistic(), Xte, te.y));
        rows.push_back(score("knn5", ens.knn(), Xte, te.y));
        rows.push_back(score("forest50", ens.forest(), Xte, te.y));
      }
      rows.push_back(score(modes[static_cast<size_t>(mode)], ens, Xte, te.y));
      if (mode == 2)
        stack = ens; // keep the stacking fit for the meta-weight dump
    }
    std::ostringstream csv;
    csv << std::setprecision(10) << "model,accuracy,macro_f1\n";
    ml::Vec heights;
    std::vector<std::string> names;
    for (const auto &r : rows) {
      csv << r.name << ',' << r.accuracy << ',' << r.f1 << '\n';
      std::cout << std::left << std::setw(14) << r.name << " accuracy "
                << r.accuracy << " macro-F1 " << r.f1 << '\n';
      heights.push_back(r.f1);
      names.push_back(r.name);
    }
    // Meta-learner weights: for output class c, feature j of the stacked
    // input is member j / k, class j % k (layout [lr | knn | forest]).
    std::ostringstream meta;
    meta << std::setprecision(10) << "meta_class,base,base_class,weight\n";
    const std::vector<std::string> bases = {"logistic", "knn5", "forest50"};
    const size_t k = stack.n_classes();
    for (size_t c = 0; c < k; ++c) {
      const ml::Vec &w = stack.meta().feature_weights(c);
      for (size_t j = 0; j < w.size(); ++j)
        meta << c << ',' << bases[j / k] << ',' << j % k << ',' << w[j] << '\n';
    }
    ml::Plot fig(720, 400);
    fig.title("Holdout macro-F1: members versus ensembles (wine)");
    fig.xlabel("model");
    fig.ylabel("macro F1");
    fig.bar(heights, names, "macro_f1");
    ml::Artifacts a(RUN_OUTPUT_DIR, ".");
    a.write("comparison.csv", csv.str());
    a.figure("macro_f1.svg", fig);
    a.write("meta_weights.csv", meta.str());
    std::cout << "wrote " << a.path("comparison.csv") << "\nwrote "
              << a.path("macro_f1.svg") << "\nwrote "
              << a.path("meta_weights.csv") << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "03_implementation: " << e.what() << '\n';
    return 1;
  }
}
