// 03_ml_course/03_anomaly_detection/03_one_class_svm/03_implementation.cpp
// Purpose: lesson showing the one-class SVM on one 2-D Gaussian blob with four
//          planted outliers: ranks scores, reports support vectors and the
//          training flag rate, and draws the learned decision region on a grid.
// Inputs:  none from disk (points are generated in code, seed 7);
//          nu = 0.1, gamma = 0.5, 41 x 41 grid over [-7, 7]^2.
// Outputs: results/03_implementation_results/{scores.csv, decision_grid.csv,
//          figures/decision_boundary.svg, figures/scores_scatter.svg,
//          figures/scores_histogram.svg}; also prints the top five scores.
// Run target: uocsvm_implementation.
// sklearn equivalent: sklearn.svm.OneClassSVM(nu=0.1, gamma=0.5).
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "OneClassSVM.hpp"
#include "helper/math/metrics.hpp"
#include "helper/reporting/artifacts.hpp"
int main() {
  using namespace ml;
  constexpr double kNu = .1, kGamma = .5;
  constexpr int kGrid = 41;  // grid points per axis for the boundary figure
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // One 2-D Gaussian blob plus four planted outliers appended at the end.
  std::mt19937 gen(7);
  std::normal_distribution<double> noise(0, 1);
  Mat X;
  for (int i = 0; i < 120; ++i) X.push_back({noise(gen), noise(gen)});
  const size_t first_outlier = X.size();
  X.push_back({5, 5});
  X.push_back({-5, 4});
  X.push_back({6, -1});
  X.push_back({0, -6});
  OneClassSVM model(kNu, kGamma);
  model.fit(X);
  const Vec scores = model.score_samples(X);  // rho - decision (> 0 outside)
  const Vec flags = model.predict(X);
  // Rank rows by descending score and print the five most anomalous. The
  // training flag rate should be close to nu (the outlier budget).
  std::vector<size_t> order(X.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t p, size_t q) { return scores[p] > scores[q]; });
  std::cout << std::setprecision(4) << "support vectors " << model.support_count()
            << " of " << X.size() << ", rho " << model.rho()
            << ", training flag rate " << mean(flags) << " (nu = " << kNu
            << ")\ntop five anomaly scores\n";
  for (size_t r = 0; r < 5; ++r)
    std::cout << "row " << order[r] << " (" << X[order[r]][0] << ", "
              << X[order[r]][1] << ") score " << scores[order[r]]
              << (order[r] >= first_outlier ? " planted outlier" : "") << '\n';
  // Success criterion: the four planted rows occupy the four top ranks.
  bool top_four_planted = true;
  for (size_t r = 0; r < 4; ++r)
    top_four_planted = top_four_planted && order[r] >= first_outlier;
  std::cout << (top_four_planted
                    ? "The four planted outliers own the four largest scores.\n"
                    : "Warning: a blob point outranks a planted outlier.\n");
  std::ostringstream csv;
  csv << std::setprecision(17) << "row,x,y,score,flag\n";
  for (size_t i = 0; i < X.size(); ++i)
    csv << i << ',' << X[i][0] << ',' << X[i][1] << ',' << scores[i] << ','
        << flags[i] << '\n';
  a.write("scores.csv", csv.str());
  // Decision boundary on a regular grid: score = 0 is the learned frontier.
  // Row-major grid, x varies fastest; inside = 1 where score <= 0.
  Mat grid;
  for (int r = 0; r < kGrid; ++r)
    for (int c = 0; c < kGrid; ++c)
      grid.push_back({-7 + 14.0 * c / (kGrid - 1), -7 + 14.0 * r / (kGrid - 1)});
  const Vec grid_scores = model.score_samples(grid);
  std::ostringstream g;
  g << std::setprecision(17) << "x,y,score,inside\n";
  Vec inside(grid.size(), 0);
  for (size_t i = 0; i < grid.size(); ++i) {
    inside[i] = grid_scores[i] <= 0 ? 1 : 0;
    g << grid[i][0] << ',' << grid[i][1] << ',' << grid_scores[i] << ','
      << inside[i] << '\n';
  }
  a.write("decision_grid.csv", g.str());
  // Figures: grid region + flagged rows, flagged rows alone, score histogram.
  Plot boundary;
  boundary.title("One-class SVM region (grid: 1 = inside) and flagged rows");
  boundary.xlabel("x");
  boundary.ylabel("y");
  boundary.scatter(grid, inside, "grid inside");
  boundary.scatter(X, flags, "flag");
  a.figure("figures/decision_boundary.svg", boundary);
  Plot scatter;
  scatter.title("Flagged rows (nu = 0.1, gamma = 0.5)");
  scatter.xlabel("x");
  scatter.ylabel("y");
  scatter.scatter(X, flags, "flag");
  a.figure("figures/scores_scatter.svg", scatter);
  Plot h;
  h.title("Distribution of anomaly scores (rho - decision)");
  h.histogram(scores, 20);
  a.figure("figures/scores_histogram.svg", h);
  std::cout << "Saved scores.csv, decision_grid.csv and figures under "
            << RUN_OUTPUT_DIR << '\n';
}
