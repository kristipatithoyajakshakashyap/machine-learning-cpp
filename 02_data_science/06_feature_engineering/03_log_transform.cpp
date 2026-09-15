// 03_log_transform.cpp
// Lesson: log1p() maps an exponential series onto a line. Air-passenger
// counts grow roughly exponentially over 1949-1960, so log(1 + passengers)
// has an almost perfect linear trend: the correlation with the row index
// jumps towards 1. The lesson compares the Pearson correlation before and
// after the transform.
// Equivalent: numpy.log1p.
//
// Data: flights.csv (real data, 144 monthly values).
//
// EXPECTED OUTPUT:
// first passengers: 112, 118, 432
// first log1p values: 4.727388, 4.779123, 6.070738
// pearson(sequence, passengers)      = 0.9239254
// pearson(sequence, log1p(passengers)) = 0.9494922

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/features.hpp"
#include "dsts/series.hpp"
#include "dsts/stats.hpp"

int main() {
  const dsts::DataFrame flights = dsts::read_csv(DATA_DIR "/flights.csv");
  const dsts::Series passengers("passengers", flights.numeric("passengers"));
  const dsts::Series logp = dsts::log1p(passengers);

  std::vector<dsts::OptD> row_idx;
  row_idx.reserve(passengers.size());
  for (size_t i = 0; i < passengers.size(); ++i) row_idx.push_back(double(i));
  const dsts::Series sequence("sequence", row_idx);

  const double r_raw = dsts::pearson(sequence, passengers);
  const double r_log = dsts::pearson(sequence, logp);

  std::cout << "first passengers: " << dsts::fmt(*passengers.at(0)) << ", "
            << dsts::fmt(*passengers.at(11)) << ", "
            << dsts::fmt(*passengers.at(143)) << "\n";
  std::cout << "first log1p values: " << dsts::fmt(*logp.at(0)) << ", "
            << dsts::fmt(*logp.at(11)) << ", " << dsts::fmt(*logp.at(143))
            << "\n";
  std::cout << "pearson(sequence, passengers)      = " << dsts::fmt(r_raw)
            << "\n";
  std::cout << "pearson(sequence, log1p(passengers)) = " << dsts::fmt(r_log)
            << "\n";
  return 0;
}