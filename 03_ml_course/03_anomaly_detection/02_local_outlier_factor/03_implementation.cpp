// 03_ml_course/03_anomaly_detection/02_local_outlier_factor/03_implementation.cpp
// Purpose: lesson showing LOF on two 2-D Gaussian clusters of different spread
//          with three planted outliers; ranks the factors, flags LOF > 1.5,
//          saves scores and figures, then demonstrates novelty scoring.
// Inputs:  none from disk (points are generated in code, seed 7).
// Outputs: results/03_implementation_results/{scores.csv,
//          figures/lof_scatter.svg, figures/lof_histogram.svg};
//          also prints the top five factors and three novelty scores.
// Run target: ulof_implementation.
// sklearn equivalent: sklearn.neighbors.LocalOutlierFactor(n_neighbors=10).
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "LocalOutlierFactor.hpp"
#include "helper/reporting/artifacts.hpp"
int main() {
  using namespace ml;
  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Two Gaussian clusters of different spread plus three planted outliers.
  // (3,3) sits between the clusters: not extreme in any coordinate, but in a
  // locally sparse region, which is exactly what LOF is designed to catch.
  std::mt19937 gen(7);
  std::normal_distribution<double> tight(0, .3), wide(0, 1);
  Mat X;
  for (int i = 0; i < 60; ++i) X.push_back({tight(gen), tight(gen)});
  for (int i = 0; i < 60; ++i) X.push_back({6 + wide(gen), 6 + wide(gen)});
  const size_t first_outlier = X.size();
  X.push_back({3, 3});
  X.push_back({-4, 5});
  X.push_back({12, 0});
  LocalOutlierFactor model(10);
  model.fit(X);
  Vec lof = model.fitted_scores();
  // Rank rows by descending factor and print the five most anomalous.
  std::vector<size_t> order(X.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t p, size_t q) { return lof[p] > lof[q]; });
  std::cout << std::setprecision(4) << "top five LOF scores\n";
  for (size_t r = 0; r < 5; ++r)
    std::cout << "row " << order[r] << " (" << X[order[r]][0] << ", "
              << X[order[r]][1] << ") lof " << lof[order[r]]
              << (order[r] >= first_outlier ? " planted outlier" : "")
              << '\n';
  // Success criterion: the three planted rows occupy the three top ranks.
  bool top_three_planted = true;
  for (size_t r = 0; r < 3; ++r)
    top_three_planted = top_three_planted && order[r] >= first_outlier;
  std::cout << (top_three_planted ? "The three planted outliers own the three "
                                    "largest factors.\n"
                                  : "Warning: a cluster point outranks a "
                                    "planted outlier.\n");
  // Simple rule for the lesson: factor above 1.5 is flagged.
  Vec flags(X.size(), 0);
  for (size_t i = 0; i < X.size(); ++i) flags[i] = lof[i] > 1.5 ? 1 : 0;
  std::ostringstream csv;
  csv << std::setprecision(17) << "row,x,y,lof,flag\n";
  for (size_t i = 0; i < X.size(); ++i)
    csv << i << ',' << X[i][0] << ',' << X[i][1] << ',' << lof[i] << ','
        << flags[i] << '\n';
  a.write("scores.csv", csv.str());
  // Figures: scatter coloured by flag, histogram of all factors.
  Plot scatter;
  scatter.title("LOF > 1.5 flagged (k = 10)");
  scatter.xlabel("x");
  scatter.ylabel("y");
  scatter.scatter(X, flags, "flag");
  a.figure("figures/lof_scatter.svg", scatter);
  Plot h;
  h.title("Distribution of LOF scores");
  h.histogram(lof, 20);
  a.figure("figures/lof_histogram.svg", h);
  // Novelty mode: a fresh query is scored against the stored rows only.
  // Cluster centres should score near 1; the between-cluster point higher.
  auto novel = model.score_samples({{0, 0}, {6, 6}, {3, 3}});
  std::cout << "novelty scores centre-A " << novel[0] << " centre-B "
            << novel[1] << " between " << novel[2] << '\n';
  std::cout << "Saved scores and figures under " << RUN_OUTPUT_DIR << '\n';
}
