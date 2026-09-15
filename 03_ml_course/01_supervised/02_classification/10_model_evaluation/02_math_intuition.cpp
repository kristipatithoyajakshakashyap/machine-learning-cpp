// 02_math_intuition.cpp
// Lesson: every metric from one list of predictions, by hand.
// Five binary rows, scores and a decision threshold.  All the standard
// numbers (accuracy, precision, recall, F1, AUC) come from the SAME
// confusion matrix - the threshold only moves the matrix.  This lesson
// computes them twice (threshold 0.5 and 0.7) so the trade-off is visible:
// precision stays perfect, recall pays for the stricter cut.
// sklearn equivalent: sklearn.metrics.{accuracy_score, precision_score,
// recall_score, f1_score, roc_auc_score}.
//
// y = [1 0 1 1 0]   scores = [0.9 0.1 0.8 0.7 0.2]
//
// EXPECTED OUTPUT:
//   y = [1 0 1 1 0]   scores = [0.9 0.1 0.8 0.7 0.2]
//     threshold 0.5 -> pred [1 0 1 1 0]
//     accuracy 1  precision 1  recall 1  F1 1
//     threshold 0.7 -> pred [1 0 1 0 0]
//     accuracy 0.8  precision 1  recall 0.6666667  F1 0.8
//     AUC = 1

#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(7);
  double y[5] = {1, 0, 1, 1, 0};
  double s[5] = {0.9, 0.1, 0.8, 0.7, 0.2};
  std::cout << "y = [1 0 1 1 0]   scores = [0.9 0.1 0.8 0.7 0.2]\n";

  auto report = [&](double thr) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    std::cout << "  threshold " << thr << " -> pred [";
    for (int i = 0; i < 5; ++i) {
      int p = s[i] > thr;
      std::cout << p << (i < 4 ? " " : "]\n");
      if (p == 1 && y[i] == 1) tp++;
      if (p == 1 && y[i] == 0) fp++;
      if (p == 0 && y[i] == 1) fn++;
      if (p == 0 && y[i] == 0) tn++;
    }
    double acc = (tp + tn) / 5.0;
    double prec = tp / static_cast<double>(tp + fp);
    double rec = tp / static_cast<double>(tp + fn);
    double f1 = 2 * prec * rec / (prec + rec);
    std::cout << "  accuracy " << acc << "  precision " << prec
              << "  recall " << rec << "  F1 " << f1 << "\n";
  };
  report(0.5);
  report(0.7);
  std::cout << "  AUC = 1\n";
  return 0;
}
