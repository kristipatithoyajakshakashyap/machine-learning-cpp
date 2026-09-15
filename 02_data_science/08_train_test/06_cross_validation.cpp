// 06_cross_validation.cpp
// Lesson: one hold-out split gives one noisy score. k-fold cross-validation
// rotates the test fold across the data so every row is tested exactly once,
// and the mean +/- std of the k scores is a far steadier estimate. Stratified
// folds keep the class mix identical in every fold, which matters most for
// small or imbalanced classes.
// Toolkit: dsts::kfold_indices() and dsts::stratified_kfold_indices().
// Equivalent: sklearn.model_selection.KFold / StratifiedKFold + cross_val_score.
//
// Data: iris.csv (150 rows, 3 species x 50). Model: nearest class mean on
// the four measurements (fit = per-class means; predict = closest mean).
//
// EXPECTED OUTPUT:
// ## plain k-fold (k=5, seed=7): fold sizes and class counts
//   fold 0: n=30  setosa 8  versicolor 10  virginica 12
//   fold 4: n=30  setosa 16  versicolor 7  virginica 7   (folds 1-3 omitted)
// ## stratified k-fold (k=5, seed=7)
//   fold 0..4: n=30  setosa 10  versicolor 10  virginica 10
// ## nearest-class-mean accuracy per fold
//   plain     : 0.9333333 0.8666667 0.9333333 0.9 0.9666667  mean 0.92 std 0.03800585
//   stratified: 0.9 0.8666667 1 0.9 0.9  mean 0.9133333 std 0.0505525
// wrote folds.csv and fold_scores.csv

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/sampling.hpp"

namespace {

const std::vector<std::string> kFeatures{"sepal_length", "sepal_width", "petal_length",
                                         "petal_width"};
const std::vector<std::string> kClasses{"setosa", "versicolor", "virginica"};

using Matrix = std::vector<std::vector<double>>;

Matrix feature_matrix(const dsts::DataFrame& df) {
  Matrix x(df.rows(), std::vector<double>(kFeatures.size(), 0.0));
  for (size_t j = 0; j < kFeatures.size(); ++j) {
    const auto& col = df.numeric(kFeatures[j]);
    for (size_t r = 0; r < df.rows(); ++r) x[r][j] = col[r].value_or(0.0);
  }
  return x;
}

// Fit: mean feature vector per class over the training rows.
std::map<std::string, std::vector<double>> fit_class_means(
    const Matrix& x, const std::vector<std::string>& y, const std::vector<size_t>& train) {
  std::map<std::string, std::vector<double>> sums;
  std::map<std::string, size_t> counts;
  for (size_t r : train) {
    auto& s = sums[y[r]];
    if (s.empty()) s.assign(kFeatures.size(), 0.0);
    for (size_t j = 0; j < kFeatures.size(); ++j) s[j] += x[r][j];
    ++counts[y[r]];
  }
  for (auto& [cls, s] : sums) {
    for (double& v : s) v /= static_cast<double>(counts[cls]);
  }
  return sums;
}

std::string predict(const std::map<std::string, std::vector<double>>& means,
                    const std::vector<double>& row) {
  std::string best;
  double best_d = 1e300;
  for (const auto& [cls, m] : means) {
    double d = 0.0;
    for (size_t j = 0; j < row.size(); ++j) d += (row[j] - m[j]) * (row[j] - m[j]);
    if (d < best_d) {
      best_d = d;
      best = cls;
    }
  }
  return best;
}

// Accuracy on fold `f` after fitting on all other folds.
double fold_accuracy(const Matrix& x, const std::vector<std::string>& y,
                     const std::vector<std::vector<size_t>>& folds, size_t f) {
  std::vector<size_t> train;
  for (size_t g = 0; g < folds.size(); ++g) {
    if (g != f) train.insert(train.end(), folds[g].begin(), folds[g].end());
  }
  const auto means = fit_class_means(x, y, train);
  size_t correct = 0;
  for (size_t r : folds[f]) correct += predict(means, x[r]) == y[r];
  return static_cast<double>(correct) / static_cast<double>(folds[f].size());
}

void print_fold_balance(const std::vector<std::vector<size_t>>& folds,
                        const std::vector<std::string>& y) {
  for (size_t f = 0; f < folds.size(); ++f) {
    std::map<std::string, size_t> counts;
    for (size_t r : folds[f]) ++counts[y[r]];
    std::cout << "  fold " << f << ": n=" << folds[f].size();
    for (const auto& c : kClasses) std::cout << "  " << c << " " << counts[c];
    std::cout << "\n";
  }
}

struct Summary {
  std::vector<double> scores;
  double mean = 0.0, sd = 0.0;
};

Summary evaluate(const Matrix& x, const std::vector<std::string>& y,
                 const std::vector<std::vector<size_t>>& folds) {
  Summary s;
  for (size_t f = 0; f < folds.size(); ++f) s.scores.push_back(fold_accuracy(x, y, folds, f));
  for (double v : s.scores) s.mean += v;
  s.mean /= static_cast<double>(s.scores.size());
  for (double v : s.scores) s.sd += (v - s.mean) * (v - s.mean);
  s.sd = std::sqrt(s.sd / static_cast<double>(s.scores.size() - 1));
  return s;
}

void print_summary(const std::string& label, const Summary& s) {
  std::cout << "  " << label << ":";
  for (double v : s.scores) std::cout << " " << dsts::fmt(v);
  std::cout << "  mean " << dsts::fmt(s.mean) << " std " << dsts::fmt(s.sd) << "\n";
}

}  // namespace

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const Matrix x = feature_matrix(iris);
  const std::vector<std::string>& y = iris.strings("species");
  const size_t k = 5;
  const uint32_t seed = 7;

  const auto plain = dsts::kfold_indices(iris.rows(), k, seed);
  const auto strat = dsts::stratified_kfold_indices(y, k, seed);

  std::cout << "## plain k-fold (k=" << k << ", seed=" << seed
            << "): fold sizes and class counts\n";
  print_fold_balance(plain, y);
  std::cout << "## stratified k-fold (k=" << k << ", seed=" << seed << ")\n";
  print_fold_balance(strat, y);

  const Summary plain_s = evaluate(x, y, plain);
  const Summary strat_s = evaluate(x, y, strat);
  std::cout << "## nearest-class-mean accuracy per fold\n";
  print_summary("plain     ", plain_s);
  print_summary("stratified", strat_s);
  std::cout << "  every row is tested exactly once; the std across folds is the\n"
            << "  wobble a single hold-out split would hide. Stratified folds all\n"
            << "  hold 10 of each species, so no fold can miss a class entirely.\n";

  // --- files ---------------------------------------------------------------
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  dsts::DataFrame folds_df;
  std::vector<dsts::OptD> row_idx, plain_fold(iris.rows()), strat_fold(iris.rows());
  for (size_t r = 0; r < iris.rows(); ++r) row_idx.push_back(static_cast<double>(r));
  for (size_t f = 0; f < k; ++f) {
    for (size_t r : plain[f]) plain_fold[r] = static_cast<double>(f);
    for (size_t r : strat[f]) strat_fold[r] = static_cast<double>(f);
  }
  folds_df.add_numeric("row", row_idx);
  folds_df.add_string("species", y);
  folds_df.add_numeric("plain_fold", plain_fold);
  folds_df.add_numeric("stratified_fold", strat_fold);
  dsts::write_csv(std::string(RUN_OUTPUT_DIR) + "/folds.csv", folds_df);

  dsts::DataFrame scores_df;
  std::vector<dsts::OptD> fold_id, acc_plain, acc_strat;
  for (size_t f = 0; f < k; ++f) {
    fold_id.push_back(static_cast<double>(f));
    acc_plain.push_back(plain_s.scores[f]);
    acc_strat.push_back(strat_s.scores[f]);
  }
  scores_df.add_numeric("fold", fold_id);
  scores_df.add_numeric("accuracy_plain", acc_plain);
  scores_df.add_numeric("accuracy_stratified", acc_strat);
  dsts::write_csv(std::string(RUN_OUTPUT_DIR) + "/fold_scores.csv", scores_df);
  std::cout << "wrote folds.csv and fold_scores.csv\n";
  return 0;
}
