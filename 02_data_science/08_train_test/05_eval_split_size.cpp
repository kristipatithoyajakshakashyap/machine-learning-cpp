// 05_eval_split_size.cpp
// Lesson: a split size is a bias/variance trade. Bigger trains = more stable
// estimates; smaller trains = more honest but noisier tests. Repeating the
// split over several seeds shows how much the number is allowed to wobble -
// the intuition behind k-fold cross-validation.
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// full-data mean tip = 2.998279
//
// frac  seed   train   test
//  0.6    1   3.02137   2.963878  (146/98)
//  0.6    2   3.069315   2.892449  (146/98)
//  0.6    3   2.999589   2.996327  (146/98)
//  0.6    4   2.963219   3.05051  (146/98)
//  0.6    5   3.010959   2.979388  (146/98)
//
//  0.8    1   2.974205   3.094082  (195/49)
//  0.8    2   3.013744   2.936735  (195/49)
//  0.8    3   3.020513   2.909796  (195/49)
//  0.8    4   3.053897   2.776939  (195/49)
//  0.8    5   2.997231   3.002449  (195/49)

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/sampling.hpp"
#include "dsts/series.hpp"

double mean_tip(const dsts::DataFrame& df) {
  return dsts::Series("t", df.numeric("tip")).mean();
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const double full = mean_tip(tips);
  std::cout << "full-data mean tip = " << dsts::fmt(full) << "\n\n";
  std::cout << "frac  seed   train   test\n";
  for (double f : {0.6, 0.8}) {
    for (uint32_t seed : {1u, 2u, 3u, 4u, 5u}) {
      const auto [tr, te] = dsts::train_test_split(tips, f, seed);
      std::cout << " " << f << "    " << seed << "   " << dsts::fmt(mean_tip(tr))
                << "   " << dsts::fmt(mean_tip(te)) << "  (" << tr.rows() << "/"
                << te.rows() << ")\n";
    }
    std::cout << "\n";
  }
  return 0;
}