// 05_standardize.cpp
// Lesson: PCA maximizes variance, so variables with big units dominate. When
// the tasks really treat all traits as equal, z-score first: the first axis
// then explains much less but no single variable rules the ranking.
// Equivalent: StandardScaler().fit_transform(...) then PCA().
//
// Data: iris.csv (real measurements, 150 flowers; all four traits are in mm).
//
// EXPECTED OUTPUT:
// raw (raw measurement scale):
//   PC1  92.46187%  (eig 4.228242)
//   PC2  5.306648%  (eig 0.2426707)
//   PC3  1.710261%  (eig 0.0782095)
//   PC4  0.5212184%  (eig 0.02383509)
//
// standardized (all traits equal vote):
//   PC1  72.96245%  (eig 2.938085)
//   PC2  22.85076%  (eig 0.9201649)
//   PC3  3.668922%  (eig 0.1477418)
//   PC4  0.5178709%  (eig 0.02085386)

#include <cmath>
#include <iostream>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/pca.hpp"

void report(const char* label, const dsts::PcaResult& p) {
  std::cout << label << ":\n";
  for (size_t k = 0; k < 4; ++k) {
    std::cout << "  PC" << k + 1 << "  " << dsts::fmt(p.explained_ratio[k] * 100.0)
              << "%  (eig " << dsts::fmt(p.eigenvalues[k]) << ")\n";
  }
}

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  const std::vector<std::string> cols = {"sepal_length", "sepal_width",
                                         "petal_length", "petal_width"};
  std::vector<std::vector<double>> X(150, std::vector<double>(4));
  std::vector<double> mu(4, 0.0), sd(4, 0.0);
  for (size_t r = 0; r < 150; ++r) {
    for (size_t j = 0; j < 4; ++j) X[r][j] = *iris.numeric(cols[j])[r];
  }
  for (size_t j = 0; j < 4; ++j) {
    for (size_t r = 0; r < 150; ++r) mu[j] += X[r][j];
    mu[j] /= 150.0;
    for (size_t r = 0; r < 150; ++r) sd[j] += (X[r][j] - mu[j]) * (X[r][j] - mu[j]);
    sd[j] = std::sqrt(sd[j] / 150.0);  // population std, like StandardScaler
  }
  std::vector<std::vector<double>> Z(150, std::vector<double>(4));
  for (size_t r = 0; r < 150; ++r) {
    for (size_t j = 0; j < 4; ++j) Z[r][j] = (X[r][j] - mu[j]) / sd[j];
  }

  report("raw (raw measurement scale)", dsts::pca(X));
  report("\nstandardized (all traits equal vote)", dsts::pca(Z));
  return 0;
}