// 05_box_plot.cpp
// Lesson: a box (whisker) plot compresses a whole distribution into
// quartiles, medians and outliers; drawing one per species lets us compare
// them compactly. Fences use the 1.5*IQR rule for whiskers and outliers.
// Equivalent: matplotlib.pyplot.boxplot, seaborn.boxplot.
//
// Data: iris.csv (real data, 150 flowers of three species).
//
// EXPECTED OUTPUT:
// medians: setosa=1.5  versicolor=4.35  virginica=5.55
// wrote 05_box_iris.svg (4083 bytes)

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/plots.hpp"
#include "dsts/series.hpp"
#include <filesystem>

size_t file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? static_cast<size_t>(f.tellg()) : 0;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const auto& petal = iris.numeric("petal_length");
  const auto& species = iris.strings("species");

  std::vector<std::string> group;
  std::vector<dsts::OptD> values;
  for (size_t i = 0; i < iris.rows(); ++i) {
    group.push_back(species[i]);
    values.push_back(petal[i]);
  }

  std::cout << "medians: ";
  for (const char* g : {"setosa", "versicolor", "virginica"}) {
    std::string gs(g);
    std::vector<dsts::OptD> vals;
    for (size_t i = 0; i < group.size(); ++i) {
      if (group[i] == gs) vals.push_back(values[i]);
    }
    std::cout << gs << "=" << dsts::Series(gs, vals).quantile(0.5) << "  ";
  }
  std::cout << "\n";

  const std::string name = "05_box_iris.svg";
  const std::string out = std::string(RUN_OUTPUT_DIR) + "/" + name;
  dsts::write_svg_box(out.c_str(), "Iris petal length by species",
                      "Petal length (cm)", group, values);

  std::cout << "wrote " << name << " (" << file_bytes(out) << " bytes)\n";
  return 0;
}