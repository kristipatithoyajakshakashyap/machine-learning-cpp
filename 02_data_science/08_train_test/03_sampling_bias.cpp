// 03_sampling_bias.cpp
// Lesson: which rows you keep changes the story. The first 40 tips rows are
// dominated by Sunday; a proper random sample keeps every weekday at
// (roughly) its true weight. Same lesson as "don't take the first page".
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// first 40 rows   (n 40): Thur 0 Fri 0 Sat 21 Sun 19
// random 40 seed 1 (n 40): Thur 11 Fri 4 Sat 16 Sun 9
// random 40 seed 7 (n 40): Thur 9 Fri 4 Sat 15 Sun 12

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/sampling.hpp"

void day_share(const dsts::DataFrame& df, const std::string& label) {
  std::cout << label << " (n " << df.rows() << "):";
  for (const char* d : {"Thur", "Fri", "Sat", "Sun"}) {
    size_t c = 0;
    for (size_t r = 0; r < df.rows(); ++r) {
      if (df.strings("day")[r] == d) ++c;
    }
    std::cout << " " << d << " " << int(c);
  }
  std::cout << "\n";
}

int main() {
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");

  std::vector<size_t> first40;
  for (size_t i = 0; i < 40; ++i) first40.push_back(i);
  day_share(tips.select_rows(first40), "first 40 rows  ");

  for (uint32_t seed : {1u, 7u}) {
    const auto [s, _] = dsts::train_test_split(tips, 40.0 / 244.0, seed);
    day_share(s, "random 40 seed " + std::to_string(seed));
  }
  return 0;
}