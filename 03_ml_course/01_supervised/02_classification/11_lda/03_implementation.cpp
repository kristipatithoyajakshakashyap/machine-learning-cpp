// 03_ml_course/01_supervised/02_classification/11_lda/03_implementation.cpp
//
// Purpose : Lesson 3 of the LDA module. Fits course::LDA on a 1-D separable
//           fixture, then on a seeded 2-D three-class Gaussian sample that obeys
//           the LDA assumption (shared covariance) and writes the scores.
// Inputs  : no dataset loader; both fixtures are generated in this file.
//           Defines used: RUN_OUTPUT_DIR (results/03_implementation_results/).
// Outputs : results/03_implementation_results/
//             discriminant_scores.csv : (x1, x2, y, p_class0..2, y_hat) per point
//             scatter.svg             : the 2-D points coloured by predicted class
// Run     : target lda_implementation (no arguments).
#include "Model.hpp"
#include "helper/plot/plot_svg.hpp"
#include "helper/reporting/artifacts.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
// Artifacts -> results/03_implementation_results/
//   discriminant_scores.csv : (x1, x2, y, p_class0..2, y_hat) on 2-D data
//   scatter.svg             : the 2-D points coloured by predicted class
int main() {
  std::cout
      << R"LESSON(The implementation estimates class means, empirical priors and pooled covariance. Cholesky triangular solves compute Mahalanobis distances without explicitly inverting covariance. Log-sum-exp normalization produces stable probabilities. Labels need not be consecutive integers.)LESSON"
      << "\n";
  // 1-D sanity check: two classes on either side of zero must be recovered.
  course::LDA model;
  ml::Mat X = {{-2}, {-1}, {1}, {2}};
  ml::Vec y = {0, 0, 1, 1};
  model.fit(X, y);
  std::cout << ml::print_vec(model.predict(X)) << "\n";

  // 2-D synthetic 3-class Gaussians sharing one covariance (LDA's model).
  // Fixed seed 17 keeps the CSV/SVG reproducible run to run.
  std::mt19937 rng(17);
  std::normal_distribution<double> gauss(0.0, 0.7);
  const double cx[3] = {0.0, 3.0, 1.5}, cy[3] = {0.0, 0.0, 2.5};
  ml::Mat X2;
  ml::Vec y2;
  for (int i = 0; i < 150; ++i) {
    const int c = i % 3; // round-robin gives 50 points per class
    X2.push_back({cx[c] + gauss(rng), cy[c] + gauss(rng)});
    y2.push_back(c);
  }
  course::LDA lda;
  lda.fit(X2, y2);
  ml::Mat proba = lda.predict_proba(X2);
  ml::Vec pred = lda.predict(X2);
  // One CSV row per point: coordinates, true label, three posteriors, argmax.
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
  // Scatter coloured by *predicted* class so boundary mistakes are visible.
  ml::Plot fig(640, 400);
  fig.title("LDA on 2-D Gaussians (coloured by predicted class)");
  fig.xlabel("x1"); fig.ylabel("x2");
  fig.class_labels({"class 0", "class 1", "class 2"});
  fig.scatter(X2, pred);
  ml::Artifacts a(RUN_OUTPUT_DIR, ".");
  a.write("discriminant_scores.csv", csv.str());
  a.figure("scatter.svg", fig);
  std::cout << "wrote " << a.path("discriminant_scores.csv") << "\n"
            << "wrote " << a.path("scatter.svg") << "\n";
}
