// 01_random_split.cpp
// Lesson: hold out a chunk of rows so evaluation happens on data the model
// never saw. train_test_split draws row indices without replacement with a
// fixed seed, so the very same split comes back every run.
// Equivalent: sklearn.model_selection.train_test_split(random_state=42).
//
// Data: titanic.csv (real data, 891 passengers).
//
// EXPECTED OUTPUT:
// full dataset: 891 rows
// train_frac 0.5 -> train 445 test 446  survived train 0.3775281 test 0.3901345
// train_frac 0.7 -> train 623 test 268  survived train 0.3868379 test 0.3768657
// train_frac 0.8 -> train 712 test 179  survived train 0.380618 test 0.396648
// train_frac 0.9 -> train 801 test 90  survived train 0.3795256 test 0.4222222
// seed=7 twice -> identical split: yes

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/sampling.hpp"
#include "dsts/series.hpp"

double survived_rate(const dsts::DataFrame& df) {
  const dsts::Series s("s", df.numeric("Survived"));
  return s.mean();
}

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");
  std::cout << "full dataset: " << t.rows() << " rows\n";

  const std::vector<double> fracs = {0.5, 0.7, 0.8, 0.9};
  for (double f : fracs) {
    const auto [train, test] = dsts::train_test_split(t, f, 42);
    std::cout << "train_frac " << f << " -> train " << train.rows()
              << " test " << test.rows() << "  survived train "
              << dsts::fmt(survived_rate(train)) << " test "
              << dsts::fmt(survived_rate(test)) << "\n";
  }

  // Same seed, same rows every time.
  const auto [t1, e1] = dsts::train_test_split(t, 0.8, 7);
  const auto [t2, e2] = dsts::train_test_split(t, 0.8, 7);
  bool same = t1.cell(0, "PassengerId") == t2.cell(0, "PassengerId") &&
              t1.rows() == t2.rows() && e1.rows() == e2.rows();
  std::cout << "seed=7 twice -> identical split: " << (same ? "yes" : "no")
            << "\n";
  return 0;
}