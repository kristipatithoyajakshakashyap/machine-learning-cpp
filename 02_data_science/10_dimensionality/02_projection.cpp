// 02_projection.cpp
// Lesson: dropping to PC1/PC2 compresses 4 dimensions into 2 while keeping
// almost all of the signal. Plotting the scores (row coordinates on the new
// axes) makes the three iris species pop as separate clouds.
// Equivalent: PCA().fit_transform(...) then a scatter.pcolormesh-view.
//
// Data: iris.csv (real measurements, 150 flowers).
//
// EXPECTED OUTPUT:
// first 6 rows scored on PC1, PC2:
//    species  PC1       PC2
//    setosa  -2.684126  0.3193972
//    setosa  -2.714142  -0.1770012
//    setosa  -2.888991  -0.1449494
//    setosa  -2.745343  -0.318299
//    setosa  -2.728717  0.3267545
//    setosa  -2.28086  0.7413304
// wrote <results/02_projection_results>/m10_02_pca_scatter.svg

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/pca.hpp"
#include "dsts/plots.hpp"
#include "dsts/series.hpp"
#include <filesystem>

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const std::vector<std::string> cols = {"sepal_length", "sepal_width",
                                         "petal_length", "petal_width"};
  std::vector<std::vector<double>> X(150, std::vector<double>(4));
  for (size_t r = 0; r < 150; ++r) {
    for (size_t j = 0; j < 4; ++j) X[r][j] = *iris.numeric(cols[j])[r];
  }
  const dsts::PcaResult p = dsts::pca(X);

  std::cout << "first 6 rows scored on PC1, PC2:\n";
  std::cout << "   species  PC1       PC2\n";
  for (size_t r = 0; r < 6; ++r) {
    std::cout << "   " << iris.strings("species")[r] << "  "
              << dsts::fmt(p.scores[r][0]) << "  " << dsts::fmt(p.scores[r][1])
              << "\n";
  }

  std::vector<dsts::OptD> pc1(150), pc2(150);
  for (size_t r = 0; r < 150; ++r) {
    pc1[r] = p.scores[r][0];
    pc2[r] = p.scores[r][1];
  }
  dsts::write_svg_scatter(RUN_OUTPUT_DIR "/m10_02_pca_scatter.svg",
                          "Iris in PC1-PC2 space", "PC1", "PC2",
                          dsts::Series("PC1", pc1), dsts::Series("PC2", pc2),
                          iris.strings("species"));
  std::cout << "wrote " RUN_OUTPUT_DIR "/m10_02_pca_scatter.svg\n";
  return 0;
}