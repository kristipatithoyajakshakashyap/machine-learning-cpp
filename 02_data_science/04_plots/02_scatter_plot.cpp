// 02_scatter_plot.cpp
// Lesson: a scatter plot shows the relation between two numeric columns; the
// colouring by species makes each group visible at a glance.
// Equivalent: matplotlib.pyplot.scatter, seaborn.scatterplot.
//
// Data: iris.csv (real data, 150 flowers of three species).
//
// EXPECTED OUTPUT:
// pearson(petal_length, petal_width) = 0.962865
// wrote 02_scatter_iris.svg (10591 bytes)

#include <fstream>
#include <iostream>
#include <string>

#include "dsts/csv.hpp"
#include "dsts/plots.hpp"
#include "dsts/stats.hpp"
#include <filesystem>

size_t file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? static_cast<size_t>(f.tellg()) : 0;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const dsts::Series petal_length("petal_length", iris.numeric("petal_length"));
  const dsts::Series petal_width("petal_width", iris.numeric("petal_width"));

  std::vector<std::string> species;
  species.reserve(iris.rows());
  for (const std::string& s : iris.strings("species")) species.push_back(s);

  std::cout << "pearson(petal_length, petal_width) = "
            << dsts::pearson(petal_length, petal_width) << "\n";

  const std::string name = "02_scatter_iris.svg";
  const std::string out = std::string(RUN_OUTPUT_DIR) + "/" + name;
  dsts::write_svg_scatter(out.c_str(), "Iris petal measurements", "Petal length",
                          "Petal width", petal_length, petal_width, species);

  std::cout << "wrote " << name << " (" << file_bytes(out) << " bytes)\n";
  return 0;
}