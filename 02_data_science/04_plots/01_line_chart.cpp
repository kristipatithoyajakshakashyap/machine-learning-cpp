// 01_line_chart.cpp
// Lesson: a line chart connects observations by their index/order. Here the
// monthly air-passenger series from the flights dataset is drawn as a line.
// Equivalent: matplotlib.pyplot.plot, seaborn.lineplot.
//
// Data: flights.csv (real data, 1949-1960).
//
// EXPECTED OUTPUT:
// rows 144  min 104  max 622
// wrote 01_line_flights.svg (28910 bytes)

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/plots.hpp"
#include <filesystem>

size_t file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? static_cast<size_t>(f.tellg()) : 0;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame f = dsts::read_csv(DATA_DIR "/flights.csv");
  const dsts::Series passengers("passengers", f.numeric("passengers"));

  std::vector<std::string> months;
  months.reserve(passengers.size());
  for (const std::string& m : f.strings("month")) months.push_back(m);

  std::cout << "rows " << passengers.size() << "  min " << passengers.min()
            << "  max " << passengers.max() << "\n";

  const std::string name = "01_line_flights.svg";
  const std::string out = std::string(RUN_OUTPUT_DIR) + "/" + name;
  dsts::write_svg_line(out.c_str(), "Monthly air passengers", "Month",
                       "Passengers", months, passengers);

  std::cout << "wrote " << name << " (" << file_bytes(out) << " bytes)\n";
  return 0;
}