// 02_seasonality.cpp
// Lesson: the seasonal profile is every January-December average across the
// years. Adding 12 months to any "same month last year" forecast idea - the
// peak clearly falls in July/August.
// Equivalent: groupby("month").mean() on a monthly series.
//
// Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960).
//
// EXPECTED OUTPUT:
// month  avg      vs overall
//    1     241.75  -38.54861
//    2        235  -45.29861
//    3   270.1667  -10.13194
//    4   267.0833  -13.21528
//    5   271.8333  -8.465278
//    6   311.6667   31.36806
//    7   351.3333   71.03472
//    8   351.0833   70.78472
//    9   302.4167   22.11806
//   10   266.5833  -13.71528
//   11   232.8333  -47.46528
//   12   261.8333  -18.46528
//
// overall monthly mean 280.2986

#include <iomanip>
#include <iostream>
#include <sstream>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"

int month_of(const std::string& m) {
  // "1949-01" -> 1
  const std::string mm = m.substr(5, 2);
  std::istringstream is(mm);
  int v = 0;
  is >> v;
  return v;
}

int main() {
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const dsts::Series p("passengers", ap.numeric("Passengers"));

  // Accumulate month-of-year sums and counts over all 12 years.
  double sums[12] = {0.0};
  int counts[12] = {0};
  for (size_t r = 0; r < ap.rows(); ++r) {
    const int mo = month_of(ap.strings("Month")[r]) - 1;
    sums[mo] += *p.values()[r];
    ++counts[mo];
  }

  const double grand = p.mean();
  std::cout << "month  avg      vs overall\n";
  for (int mo = 0; mo < 12; ++mo) {
    const double avg = sums[mo] / static_cast<double>(counts[mo]);
    std::cout << "  " << std::setw(2) << mo + 1 << "  " << std::setw(9)
              << dsts::fmt(avg) << "  " << std::setw(9)
              << dsts::fmt(avg - grand) << "\n";
  }
  std::cout << "\noverall monthly mean " << dsts::fmt(grand) << "\n";
  return 0;
}