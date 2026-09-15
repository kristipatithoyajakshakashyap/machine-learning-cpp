// 06_heatmap.cpp
// Lesson: a correlation heatmap colours the Pearson correlation matrix so
// strong relations jump out immediately (red negative, blue positive).
// Equivalent: seaborn.heatmap(df.corr()).
//
// Data: iris.csv and tips.csv (real data).
//
// EXPECTED OUTPUT:
// iris correlation petal_width vs petal_length = 0.962865
// wrote 06_heatmap_iris.svg (4060 bytes) and 06_heatmap_tips.svg (2686 bytes)

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
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const auto iris_matrix = dsts::correlation_matrix(iris);
  const std::vector<std::string> iris_cols = {
      "sepal_length", "sepal_width", "petal_length", "petal_width"};

  std::cout << "iris correlation petal_width vs petal_length = "
            << iris_matrix[3][2] << "\n";

  const std::string name1 = "06_heatmap_iris.svg";
  const std::string out1 = std::string(RUN_OUTPUT_DIR) + "/" + name1;
  dsts::write_svg_heatmap(out1, "Iris correlations", iris_cols, iris_matrix);

  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  const dsts::DataFrame tips_num = tips.select_columns({"total_bill", "tip", "size"});
  const auto tips_matrix = dsts::correlation_matrix(tips_num);

  const std::string name2 = "06_heatmap_tips.svg";
  const std::string out2 = std::string(RUN_OUTPUT_DIR) + "/" + name2;
  dsts::write_svg_heatmap(out2, "Tips correlations",
                          tips_num.columns(), tips_matrix);

  std::cout << "wrote " << name1 << " (" << file_bytes(out1)
            << " bytes) and " << name2 << " (" << file_bytes(out2)
            << " bytes)\n";
  return 0;
}