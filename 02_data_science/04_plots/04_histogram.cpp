// 04_histogram.cpp
// Lesson: a histogram groups a numeric column into equal-width bins and
// counts the observations per bin, revealing the shape of the distribution.
// Equivalent: matplotlib.pyplot.hist, seaborn.histplot.
//
// Data: tips.csv and iris.csv (real data).
//
// EXPECTED OUTPUT:
// total_bill   min 3.07  max 50.81
// petal_length min 1  max 6.9
// wrote 04_hist_tips.svg (3232 bytes) and 04_hist_iris.svg (3359 bytes)

#include <fstream>
#include <iostream>
#include <string>

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
  const dsts::Series total_bill("total_bill", tips.numeric("total_bill"));
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::Series petal_length("petal_length",
                                  iris.numeric("petal_length"));

  std::cout << "total_bill   min " << total_bill.min() << "  max "
            << total_bill.max() << "\n";
  std::cout << "petal_length min " << petal_length.min() << "  max "
            << petal_length.max() << "\n";

  const std::string name1 = "04_hist_tips.svg";
  const std::string out1 = std::string(RUN_OUTPUT_DIR) + "/" + name1;
  dsts::write_svg_histogram(out1, "Restaurant bill sizes", "Total bill ($)",
                            "Count", total_bill, 8);

  const std::string name2 = "04_hist_iris.svg";
  const std::string out2 = std::string(RUN_OUTPUT_DIR) + "/" + name2;
  dsts::write_svg_histogram(out2, "Iris petal length", "Petal length (cm)",
                            "Count", petal_length, 10);

  std::cout << "wrote " << name1 << " (" << file_bytes(out1)
            << " bytes) and " << name2 << " (" << file_bytes(out2)
            << " bytes)\n";
  return 0;
}