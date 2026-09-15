// 03_ml_course/03_anomaly_detection/01_isolation_forest/03_implementation.cpp
// Purpose: lesson showing the isolation forest at work. A 1-D line plus one far
//          point is the sanity check; then, mirroring the LOF lesson, two
//          Gaussian clusters with three planted outliers are scored and the
//          scores and figures are saved.
// Inputs:  none from disk (both datasets are generated in code, seed 7).
// Outputs: results/03_implementation_results/{scores.csv,
//          figures/isolation_scatter.svg, figures/isolation_histogram.svg};
//          also prints the top five scores.
// Run target: uif_implementation.
// sklearn equivalent: sklearn.ensemble.IsolationForest(random_state=42).

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "IsolationForest.hpp"
#include "helper/reporting/artifacts.hpp"

int main() {
  using namespace ml;
  // Part 1 - 100 points on a line y=0 and one point at (20, 20). The far point
  // should score well above the middle-of-the-line point.
  Mat X;
  for (int i = 0; i < 100; ++i) X.push_back({double(i) / 100, 0});
  X.push_back({20, 20});
  IsolationForest m;
  m.fit(X);
  auto s = m.score_samples(X);
  std::cout << "ordinary " << s[50] << " outlier " << s.back() << '\n';

  // Part 2 - two 2-D Gaussian blobs (tight at the origin, wide around (6, 6))
  // plus three hand-placed outliers appended at the end.
  Artifacts a(RUN_OUTPUT_DIR, ".");
  std::mt19937 gen(7);
  std::normal_distribution<double> tight(0, .3), wide(0, 1);
  Mat S;
  for (int i = 0; i < 60; ++i) S.push_back({tight(gen), tight(gen)});
  for (int i = 0; i < 60; ++i) S.push_back({6 + wide(gen), 6 + wide(gen)});
  const size_t first_outlier = S.size();
  S.push_back({3, 3});
  S.push_back({-4, 5});
  S.push_back({12, 0});
  // 100 trees on 64-row subsamples; small subsamples keep trees shallow.
  IsolationForest forest(100, 64, 42);
  forest.fit(S);
  const Vec score = forest.score_samples(S);
  // Rank rows by descending score and print the five most anomalous.
  std::vector<size_t> order(S.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t p, size_t q) { return score[p] > score[q]; });
  std::cout << std::setprecision(4) << "top five isolation scores\n";
  for (size_t r = 0; r < 5; ++r)
    std::cout << "row " << order[r] << " (" << S[order[r]][0] << ", "
              << S[order[r]][1] << ") score " << score[order[r]]
              << (order[r] >= first_outlier ? " planted outlier" : "") << '\n';
  // Success criterion: the three planted rows occupy the three top ranks.
  bool top_three_planted = true;
  for (size_t r = 0; r < 3; ++r)
    top_three_planted = top_three_planted && order[r] >= first_outlier;
  std::cout << (top_three_planted
                    ? "The three planted outliers own the three largest "
                      "scores.\n"
                    : "Warning: a cluster point outranks a planted outlier.\n");

  // Per-row table with the planted flag so the CSV can be checked by hand.
  Vec planted(S.size(), 0);
  for (size_t i = first_outlier; i < S.size(); ++i) planted[i] = 1;
  std::ostringstream csv;
  csv << std::setprecision(17) << "row,x,y,score,planted_outlier\n";
  for (size_t i = 0; i < S.size(); ++i)
    csv << i << ',' << S[i][0] << ',' << S[i][1] << ',' << score[i] << ','
        << planted[i] << '\n';
  a.write("scores.csv", csv.str());

  // Figures: scatter coloured by planted flag, histogram of all scores.
  Plot scatter;
  scatter.title("Isolation forest: planted outliers marked (100 trees)");
  scatter.xlabel("x");
  scatter.ylabel("y");
  scatter.class_labels({"inlier", "planted outlier"});
  scatter.scatter(S, planted, "planted");
  a.figure("figures/isolation_scatter.svg", scatter);
  Plot h;
  h.title("Distribution of isolation scores");
  h.xlabel("score");
  h.histogram(score, 20);
  a.figure("figures/isolation_histogram.svg", h);
  std::cout << "Saved scores and figures under " << RUN_OUTPUT_DIR << '\n';
}
