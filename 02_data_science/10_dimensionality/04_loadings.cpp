// 04_loadings.cpp
// Lesson: loadings are the weights of each original variable on a principal
// component. PC1 leans almost entirely on the petal traits - i.e. most of
// iris' spread lives on petal size - while PC2 separates on sepal shape
// ("how wide for its length").
// Equivalent: sklearn PCA.components_ (rows = PCs, columns = variables).
//
// Data: iris.csv (real measurements, 150 flowers).
//
// EXPECTED OUTPUT:
// loadings (row = PC, col = sepal_length, sepal_width, petal_length, petal_width)
//   PC1: 0.3613866 -0.08452251 0.8566706 0.3582892
//   PC2: 0.6565888 0.7301614 -0.1733727 -0.07548102
//   PC3: 0.5820299 -0.5979108 -0.07623608 -0.5458314
//   PC4: 0.3154872 -0.3197231 -0.479839 0.7536574

#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/pca.hpp"

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const std::vector<std::string> cols = {"sepal_length", "sepal_width",
                                         "petal_length", "petal_width"};
  std::vector<std::vector<double>> X(150, std::vector<double>(4));
  for (size_t r = 0; r < 150; ++r) {
    for (size_t j = 0; j < 4; ++j) X[r][j] = *iris.numeric(cols[j])[r];
  }
  const dsts::PcaResult pc = dsts::pca(X);

  std::cout << "loadings (row = PC, col = " << cols[0] << ", " << cols[1]
            << ", " << cols[2] << ", " << cols[3] << ")\n";
  for (size_t i = 0; i < 4; ++i) {
    std::cout << "  PC" << i + 1 << ":";
    for (size_t j = 0; j < 4; ++j) {
      std::cout << " " << dsts::fmt(pc.loadings[i][j]);
    }
    std::cout << "\n";
  }
  return 0;
}