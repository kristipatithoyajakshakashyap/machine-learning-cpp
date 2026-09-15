// 03_reconstruction.cpp
// Lesson: PCA is lossy compression. Keeping k axes and mapping back to iris
// space reproduces the flowers imperfectly; the mean absolute error shows how
// fast the signal shrinks with k. Reconstructed values are scores * loadings
// plus the column means.
// Equivalent: PCA(...).inverse_transform on the first k scores.
//
// Data: iris.csv (real measurements, 150 flowers, 4 traits).
//
// EXPECTED OUTPUT:
// k=1  RMSE 0.2925821  on original scale
// k=2  RMSE 0.1591888  on original scale
// k=3  RMSE 0.07693535  on original scale
// k=4  RMSE 4.240658e-16  on original scale

#include <cmath>
#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/pca.hpp"

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const std::vector<std::string> cols = {"sepal_length", "sepal_width",
                                         "petal_length", "petal_width"};
  const size_t n = 150, p = cols.size();
  std::vector<std::vector<double>> X(n, std::vector<double>(p));
  for (size_t r = 0; r < n; ++r) {
    for (size_t j = 0; j < p; ++j) X[r][j] = *iris.numeric(cols[j])[r];
  }
  const dsts::PcaResult pc = dsts::pca(X);

  for (size_t k = 1; k <= p; ++k) {
    double sse = 0.0;
    size_t pairs = 0;
    for (size_t r = 0; r < n; ++r) {
      std::vector<double> hat(p, 0.0);
      for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < p; ++j) {
          hat[j] += pc.scores[r][i] * pc.loadings[i][j];
        }
      }
      for (size_t j = 0; j < p; ++j) {
        double mu = 0.0;
        for (size_t r2 = 0; r2 < n; ++r2) mu += X[r2][j];
        mu /= static_cast<double>(n);
        const double e = X[r][j] - (hat[j] + mu);
        sse += e * e;
        ++pairs;
      }
    }
    std::cout << "k=" << k << "  RMSE " << dsts::fmt(std::sqrt(sse / pairs))
              << "  on original scale\n";
  }
  return 0;
}