// 01_explained_variance.cpp
// Lesson: PCA rotates a dataset onto directions that maximally spread the
// points. The fraction of total variance captured by each axis (eigenvalue /
// sum) decides how many axes you need; iris boils down to the first two.
// Equivalent: sklearn.decomposition.PCA().fit(iris_X).explained_variance_ratio_.
//
// Data: iris.csv (real measurements, 150 flowers, 4 traits).
//
// EXPECTED OUTPUT:
// PC  eigenvalue   explained   cumulative
//   1   4.228242   92.46187%   92.46187%
//   2   0.2426707   5.306648%   97.76852%
//   3   0.0782095   1.710261%   99.47878%
//   4   0.02383509   0.5212184%   100%
//
// wrote <results/01_explained_variance_results>/m10_01_scree.svg

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/pca.hpp"
#include "dsts/plots.hpp"
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

  double cum = 0.0;
  std::cout << "PC  eigenvalue   explained   cumulative\n";
  for (size_t k = 0; k < 4; ++k) {
    cum += p.explained_ratio[k];
    std::cout << "  " << k + 1 << "   " << dsts::fmt(p.eigenvalues[k]) << "   "
              << dsts::fmt(p.explained_ratio[k] * 100.0) << "%   "
              << dsts::fmt(cum * 100.0) << "%\n";
  }

  std::vector<double> ratios(4);
  std::vector<std::string> names(4);
  for (size_t k = 0; k < 4; ++k) {
    ratios[k] = p.explained_ratio[k] * 100.0;
    names[k] = "PC" + std::to_string(k + 1);
  }
  dsts::write_svg_bar(RUN_OUTPUT_DIR "/m10_01_scree.svg",
                      "Iris explained variance per PC", "component", "%",
                      names, ratios);
  std::cout << "\nwrote " RUN_OUTPUT_DIR "/m10_01_scree.svg\n";
  return 0;
}