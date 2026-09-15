// 04_pivot.cpp
// Lesson: pivot_table() turns three columns into a grid: one row per value of
// the index column, one column per category of the columns column, cells hold
// the mean of a numeric value column. Two real pivots are built: penguin bill
// length by species x island, and tips by day x meal time.
// Equivalent: pandas.pivot_table(..., aggfunc="mean").
//
// Data: penguins.csv (real data, 344 penguins) and tips.csv (244 tips).
//
// EXPECTED OUTPUT:
// ## mean bill length (mm) by species x island
//      species  Torgersen    Biscoe     Dream
// 0     Adelie   38.95098    38.975  38.50179
// 1  Chinstrap         NA        NA  48.83382
// 2     Gentoo         NA  47.50488        NA
//
// ## mean total bill by day x time of day
//     day    Dinner     Lunch
// 0   Sun     21.41        NA
// 1   Sat  20.44138        NA
// 2  Thur     18.78  17.66475
// 3   Fri  19.66333  12.84571

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/join.hpp"

int main() {
  const dsts::DataFrame penguins = dsts::read_csv(DATA_DIR "/penguins.csv");
  const dsts::DataFrame pv = dsts::pivot_table(
      penguins, "species", "island", "bill_length_mm");
  std::cout << "## mean bill length (mm) by species x island\n"
            << pv.to_string() << "\n";

  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::DataFrame pv2 =
      dsts::pivot_table(tips, "day", "time", "total_bill");
  std::cout << "## mean total bill by day x time of day\n"
            << pv2.to_string() << "\n";
  return 0;
}