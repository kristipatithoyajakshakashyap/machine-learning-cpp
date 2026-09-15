// 03_implementation.cpp
// Lesson: cluster evaluation with the silhouette coefficient.
// First the tiny hand-checkable example, then a synthetic three-blob dataset
// clustered by k-means for k = 2..6 so the silhouette-vs-k curve (the usual
// "pick k" diagnostic) can be written out and plotted.
// sklearn equivalent: sklearn.metrics.silhouette_score / silhouette_samples.

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "../01_kmeans/KMeans.hpp"
#include "helper/math/metrics.hpp"
#include "helper/reporting/artifacts.hpp"

int main() {
  using namespace ml;
  Mat X{{0, 0}, {2, 0}, {8, 0}, {10, 0}};
  Vec labels{0, 0, 1, 1};
  std::cout << silhouette(X, labels);
  std::cout << '\n';

  Artifacts a(RUN_OUTPUT_DIR, ".");
  // Three Gaussian blobs (deterministic seed) so the "true" k is 3.
  std::mt19937 gen(11);
  std::normal_distribution<double> noise(0, .6);
  const Mat centres{{0, 0}, {6, 0}, {3, 5}};
  Mat B;
  for (const auto& c : centres)
    for (int i = 0; i < 50; ++i) B.push_back({c[0] + noise(gen), c[1] + noise(gen)});

  std::ostringstream per_k;
  per_k << std::setprecision(17) << "k,silhouette,inertia\n";
  Vec ks, scores;
  double best_score = -1;
  size_t best_k = 2;
  Vec best_labels;
  std::cout << std::setprecision(4) << "silhouette per k on three blobs\n";
  for (size_t k = 2; k <= 6; ++k) {
    KMeans km(k, 10, 300, 42);
    km.fit(B);
    const double s = silhouette(B, km.labels());
    std::cout << "k = " << k << "  silhouette " << s << "  inertia "
              << km.inertia() << '\n';
    per_k << k << ',' << s << ',' << km.inertia() << '\n';
    ks.push_back(double(k));
    scores.push_back(s);
    if (s > best_score) {
      best_score = s;
      best_k = k;
      best_labels = km.labels();
    }
  }
  std::cout << "best k by silhouette: " << best_k << '\n';
  a.write("silhouette_per_k.csv", per_k.str());

  Plot curve;
  curve.title("Silhouette score versus k (three planted blobs)");
  curve.xlabel("k");
  curve.ylabel("mean silhouette");
  curve.line(ks, scores, "silhouette");
  a.figure("figures/silhouette_per_k.svg", curve);

  // Per-sample silhouettes for the best k.
  Vec per_sample = silhouette_samples(B, best_labels);
  std::ostringstream ps;
  ps << std::setprecision(17) << "row,x,y,cluster,silhouette\n";
  for (size_t i = 0; i < B.size(); ++i)
    ps << i << ',' << B[i][0] << ',' << B[i][1] << ',' << best_labels[i] << ','
       << per_sample[i] << '\n';
  a.write("silhouette_samples.csv", ps.str());

  Plot scatter;
  scatter.title("k-means clusters at best k = " + std::to_string(best_k));
  scatter.xlabel("x");
  scatter.ylabel("y");
  scatter.scatter(B, best_labels, "cluster");
  a.figure("figures/clusters_best_k.svg", scatter);
  std::cout << "Saved silhouette_per_k.csv, silhouette_samples.csv and "
               "figures under "
            << RUN_OUTPUT_DIR << '\n';
}
