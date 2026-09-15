// 03_ml_course/01_supervised/02_classification/13_imbalanced_classification/01_theory.cpp
//
// Purpose : Lesson 1 - why accuracy misleads when one class is rare. Prints the
//           lesson text and works through the module's 30-of-387 prevalence with
//           two hypothetical classifiers.
// Inputs  : none (numbers are hard-coded to match the pipeline's dataset).
// Outputs : prints only.
// Run     : target cimb_theory (no arguments).
//
// Lesson 1: why accuracy misleads when one class is rare.
#include "Model.hpp"
#include <iomanip>
#include <iostream>
int main() {
  std::cout << std::setprecision(4)
            << R"LESSON(Class imbalance: when positives are rare, a classifier that never predicts
the positive class still scores high accuracy. The useful questions are how
many true positives it recovers (recall) and how many alarms are real
(precision). Cost-sensitive learning reweights the loss so a missed positive
costs more than a false alarm; threshold moving trades precision for recall
after training. Precision-recall curves expose rare-class behaviour that ROC
curves hide.)LESSON"
            << "\n\n";
  // Same counts as make_imbalanced(breast_cancer, 1.0, 30, 42): 357 negatives
  // plus 30 positives.
  const double n = 387, positives = 30;
  std::cout << "Prevalence: " << positives / n << " (30 of 387 rows)\n";
  // The trivial "always negative" model: accuracy = 357/387 but recall = 0.
  std::cout << "Always-negative classifier accuracy: " << (n - positives) / n
            << ", recall 0, precision undefined\n";
  // A useful detector: 24 true positives, 6 misses, 12 false alarms.
  // accuracy = (387 - 6 - 12)/387, recall = 24/30, precision = 24/(24 + 12).
  std::cout << "Detector with 24 hits and 12 false alarms: accuracy "
            << (n - 6 - 12) / n << ", recall " << 24 / positives
            << ", precision " << 24.0 / 36 << '\n';
}
