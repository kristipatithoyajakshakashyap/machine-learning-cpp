// 03_bar_chart.cpp
// Lesson: a bar chart shows one scalar value per category. Here the average
// total bill per day of the week is aggregated and drawn as bars.
// Equivalent: matplotlib.pyplot.bar, seaborn.barplot.
//
// Data: tips.csv (real data, 244 restaurant tips).
//
// EXPECTED OUTPUT:
// Fri  mean 17.15158  n 19
// Sat  mean 20.44138  n 87
// Sun  mean 21.41  n 76
// Thur  mean 17.68274  n 62
// wrote 03_bar_tips.svg (2847 bytes)

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/eda.hpp"
#include "dsts/plots.hpp"
#include <filesystem>

size_t file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? static_cast<size_t>(f.tellg()) : 0;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const auto groups = dsts::group_mean(tips, "day", "total_bill");

  std::vector<std::string> days;
  std::vector<double> means;
  for (const auto& g : groups) {
    days.push_back(g.group);
    means.push_back(g.mean);
    std::cout << g.group << "  mean " << dsts::fmt(g.mean) << "  n " << g.count
              << "\n";
  }

  const std::string name = "03_bar_tips.svg";
  const std::string out = std::string(RUN_OUTPUT_DIR) + "/" + name;
  dsts::write_svg_bar(out.c_str(), "Average bill by day", "Day",
                      "Average total bill ($)", days, means);

  std::cout << "wrote " << name << " (" << file_bytes(out) << " bytes)\n";
  return 0;
}