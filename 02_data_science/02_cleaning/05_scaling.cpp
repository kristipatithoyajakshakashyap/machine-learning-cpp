// 05_scaling.cpp
// Lesson: features live on different scales (bill in $, size as 2-4 people).
// MinMax puts every value into [0,1]; z-score makes mean 0 and sd 1. Both
// keep the shape, so after scaling we can head to distances and models.
// Equivalent: sklearn MinMaxScaler / StandardScaler.
//
// Data: tips.csv (real data).
//
// EXPECTED OUTPUT:
// ## total_bill -> [0,1] by min-max
//   0    0.2915794
//   1    0.1522832
//   2    0.3757855
//   3    0.4317134
//   4    0.450775
// Name: total_bill, dtype: float64
// range check: min = 0, max = 1
// first value  = 0.2915794  (bill 16.99)
// largest bill = 50.81 must map to 1.0
//
// ## total_bill -> z-scores
//   0    -0.3147113
//   1    -1.063235
//   2    0.1377799
//   3    0.4383151
//   4    0.5407447
// Name: total_bill, dtype: float64
// mean after = -4.70366e-16
// sd   after = 1.002056
//
// ## before scaling (raw units)
// total_bill mean = 19.78594
// size       mean = 2.569672
// ## after z-scaling
// total_bill meanZ = -4.70366e-16
// size       meanZ = -3.549074e-17
// both are now comparable; a tip does not outweigh party size

#include <iostream>

#include "dsts/cleaning.hpp"
#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::Series bill = tips.series("total_bill");

  const dsts::Series mm = dsts::minmax_scale(bill);
  std::cout << "## total_bill -> [0,1] by min-max\n"
            << mm.head(5) << "\n"
            << "range check: min = " << dsts::fmt(mm.min())
            << ", max = " << dsts::fmt(mm.max()) << "\n"
            << "first value  = " << dsts::to_text(mm.at(0)) << "  (bill 16.99)\n"
            << "largest bill = " << dsts::fmt(bill.max())
            << " must map to 1.0\n\n";

  const dsts::Series zs = dsts::zscore_scale(bill);
  std::cout << "## total_bill -> z-scores\n"
            << zs.head(5) << "\n"
            << "mean after = " << dsts::fmt(zs.mean()) << "\n"
            << "sd   after = " << dsts::fmt(zs.stddev()) << "\n\n";

  // Two features of very different magnitudes, side by side.
  const dsts::Series size = tips.series("size");
  std::cout << "## before scaling (raw units)\n"
            << "total_bill mean = " << dsts::fmt(bill.mean()) << "\n"
            << "size       mean = " << dsts::fmt(size.mean()) << "\n"
            << "## after z-scaling\n"
            << "total_bill meanZ = " << dsts::fmt(dsts::zscore_scale(bill).mean())
            << "\n"
            << "size       meanZ = " << dsts::fmt(dsts::zscore_scale(size).mean())
            << "\n";
  std::cout << "both are now comparable; a tip does not outweigh party size\n";

  return 0;
}