// 05_acf.cpp
// Lesson: the autocorrelation function plots corr(y[t], y[t-lag]) against
// lag. A repeating bump every 12 steps is the fingerprint of seasonality;
// here lag 12 beats every other lag.
// Equivalent: pandas.Series.autocorr(lag) for lags 1..13.
//
// Data: air_passengers.csv (real data, 144 monthly totals).
//
// EXPECTED OUTPUT:
// lag  autocorr
//    1   0.9601946
//    2   0.8956753
//    3   0.8373948
//    4   0.7977347
//    5   0.7859431
//    6   0.7839188
//    7   0.7845921
//    8   0.792215
//    9   0.8278519
//   10   0.8827128
//   11   0.949702
//   12   0.9905274
//   13   0.9481066
//
// peak lag 12 (autocorr 0.9905274) -> strong 12-month seasonality

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/series.hpp"
#include "dsts/timeseries.hpp"

int main() {
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const dsts::Series p("passengers", ap.numeric("Passengers"));

  std::cout << "lag  autocorr\n";
  double best = -2.0;
  size_t best_lag = 0;
  for (size_t lag = 1; lag <= 13; ++lag) {
    const double a = dsts::autocorr(p, lag);
    std::cout << "  " << (lag < 10 ? " " : "") << lag << "   " << dsts::fmt(a)
              << "\n";
    if (a > best) {
      best = a;
      best_lag = lag;
    }
  }
  std::cout << "\npeak lag " << best_lag << " (autocorr " << dsts::fmt(best)
            << ") -> strong 12-month seasonality\n";
  return 0;
}