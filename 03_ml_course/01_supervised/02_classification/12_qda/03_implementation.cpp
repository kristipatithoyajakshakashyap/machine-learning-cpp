// 03_ml_course/01_supervised/02_classification/12_qda/03_implementation.cpp
//
// Purpose : Lesson 3 of the QDA module. Fits course::QDA on a 1-D fixture, then
//           on a seeded 2-D three-class sample whose classes have *different*
//           spreads (the situation QDA models and LDA cannot), and recovers each
//           class covariance from the stored Cholesky factor.
// Inputs  : no dataset loader; fixtures are generated here.
//           Defines used: RUN_OUTPUT_DIR (results/03_implementation_results/).
// Outputs : results/03_implementation_results/
//             discriminant_scores.csv : (x1, x2, y, p_class0..2, y_hat) per point
//             class_covariance.csv    : (class, row, col, value) of each cov_c
//             scatter.svg             : the 2-D points coloured by predicted class
// Run     : target qda_implementation (no arguments).
#include "Model.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
// Artifacts -> results/03_implementation_results/
//   discriminant_scores.csv : (x1, x2, y, p_class0..2, y_hat) on 2-D data
//   class_covariance.csv    : per-class covariance entries (class, row, col, value)
//   scatter.svg             : the 2-D points coloured by predicted class
int main() {
  std::cout
      << R"LESSON(QDA reuses the canonical Gaussian discriminant implementation with class-specific maximum-likelihood covariances. It retains separate Cholesky factors and log determinants for every class. Save/load includes label mapping and all fitted state.)LESSON"
      << "\n";
  // 1-D sanity check on a separable two-class fixture.
  course::QDA model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";

  // 2-D synthetic 3-class Gaussians with class-specific spread (QDA's model).
  // sx/sy scale a unit normal per axis, so each class has its own diagonal
  // covariance; seed 23 keeps the artifacts reproducible.
  std::mt19937 rng(23);
  std::normal_distribution<double> gauss(0.0, 1.0);
  const double cx[3] = {0.0, 3.0, 1.5}, cy[3] = {0.0, 0.0, 3.0};
  const double sx[3] = {0.4, 0.9, 0.6}, sy[3] = {0.9, 0.4, 0.6};
  ml::Mat X2;
  ml::Vec y2;
  for (int i = 0; i < 150; ++i) {
    const int c = i % 3; // 50 points per class
    X2.push_back({cx[c] + sx[c] * gauss(rng), cy[c] + sy[c] * gauss(rng)});
    y2.push_back(c);
  }
  course::QDA qda;
  qda.fit(X2, y2);
  ml::Mat proba = qda.predict_proba(X2);
  ml::Vec pred = qda.predict(X2);
  std::ostringstream csv;
  csv << std::setprecision(10) << "x1,x2,y,p_class0,p_class1,p_class2,y_hat\n";
  size_t correct = 0;
  for (size_t i = 0; i < X2.size(); ++i) {
    csv << X2[i][0] << "," << X2[i][1] << "," << y2[i];
    for (double p : proba[i]) csv << "," << p;
    csv << "," << pred[i] << "\n";
    if (pred[i] == y2[i]) ++correct;
  }
  std::cout << "2-D demo train accuracy " << correct << "/" << X2.size() << "\n";
  // Per-class covariance recovered from the stored Cholesky factor L L^T.
  // Entry (r, k) = sum_j L[r][j] * L[k][j]; the ridge added in fit() is included.
  std::ostringstream cov;
  cov << std::setprecision(10) << "class,row,col,value\n";
  for (size_t c = 0; c < qda.factors.size(); ++c) {
    const ml::Mat &L = qda.factors[c];
    for (size_t r = 0; r < L.size(); ++r)
      for (size_t k = 0; k < L.size(); ++k) {
        double v = 0;
        for (size_t j = 0; j < L.size(); ++j) v += L[r][j] * L[k][j];
        cov << qda.labels[c] << "," << r << "," << k << "," << v << "\n";
      }
  }
  ml::Plot fig(640, 400);
  fig.title("QDA on 2-D Gaussians (coloured by predicted class)");
  fig.xlabel("x1"); fig.ylabel("x2");
  fig.class_labels({"class 0", "class 1", "class 2"});
  fig.scatter(X2, pred);
  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  a.write("discriminant_scores.csv", csv.str());
  a.write("class_covariance.csv", cov.str());
  a.figure("scatter.svg", fig);
  std::cout << "wrote " << a.path("discriminant_scores.csv") << "\n"
            << "wrote " << a.path("class_covariance.csv") << "\n"
            << "wrote " << a.path("scatter.svg") << "\n";
}
