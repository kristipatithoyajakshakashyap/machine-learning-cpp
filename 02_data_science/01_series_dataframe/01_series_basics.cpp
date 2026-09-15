// 01_series_basics.cpp
// Lesson: a Series is one named column of numbers, like pandas.Series.
// Every cell is a dsts::OptD = std::optional<double>; a missing value is
// std::nullopt and is ignored by all statistics.
//
// Data: first 10 sepal_length values of the iris flower dataset (real data).
//
// EXPECTED OUTPUT:
// ## The series itself
// 0    5.1
// 1    4.9
// 2    4.7
// 3    4.6
// 4    5
// 5    5.4
// 6    4.6
// 7    5
// 8    4.4
// 9    4.9
// Name: sepal_length, dtype: float64
//
// size       = 10
// count      = 10 (non-missing)
// missing    = 0
//
// ## Descriptive statistics
// mean       = 4.86
// median     = 4.9
// variance   = 0.08488889
// stddev     = 0.291357
// min        = 4.4
// max        = 5.4
// Q1 (25%)   = 4.625
// Q3 (75%)   = 5
//
// ## describe() in one call
// count    10
// mean     4.86
// std      0.291357
// min      4.4
// 25%      4.625
// 50%      4.9
// 75%      5
// max      5.4
// Name: sepal_length, dtype: float64
//
// ## First three values
// 0    5.1
// 1    4.9
// 2    4.7
// Name: sepal_length, dtype: float64
//
// ## Last three values
// 7    5
// 8    4.4
// 9    4.9
// Name: sepal_length, dtype: float64

#include <iostream>
#include <vector>

#include "dsts/series.hpp"

int main() {
  // 10 real iris sepal lengths, in centimetres (rows 1-10 of iris.csv).
  dsts::Series sepal_length("sepal_length",
                            {5.1, 4.9, 4.7, 4.6, 5.0,
                             5.4, 4.6, 5.0, 4.4, 4.9});

  std::cout << "## The series itself\n"
            << sepal_length.to_string(10) << "\n\n";

  std::cout << "size       = " << sepal_length.size() << "\n"
            << "count      = " << sepal_length.count() << " (non-missing)\n"
            << "missing    = " << sepal_length.missing() << "\n\n";

  std::cout << "## Descriptive statistics\n"
            << "mean       = " << dsts::fmt(sepal_length.mean()) << "\n"
            << "median     = " << dsts::fmt(sepal_length.median()) << "\n"
            << "variance   = " << dsts::fmt(sepal_length.variance()) << "\n"
            << "stddev     = " << dsts::fmt(sepal_length.stddev()) << "\n"
            << "min        = " << dsts::fmt(sepal_length.min()) << "\n"
            << "max        = " << dsts::fmt(sepal_length.max()) << "\n"
            << "Q1 (25%)   = " << dsts::fmt(sepal_length.quantile(0.25)) << "\n"
            << "Q3 (75%)   = " << dsts::fmt(sepal_length.quantile(0.75)) << "\n\n";

  std::cout << "## describe() in one call\n" << sepal_length.describe() << "\n\n";

  std::cout << "## First three values\n" << sepal_length.head(3) << "\n\n";
  std::cout << "## Last three values\n" << sepal_length.tail(3) << "\n";

  return 0;
}