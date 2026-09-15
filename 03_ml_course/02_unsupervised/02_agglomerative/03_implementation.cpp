// 03_implementation.cpp
// Lesson: average-linkage hierarchical clustering on iris.
// Unsupervised: the algorithm never sees species labels.  After fitting with
// 3 clusters we best-match cluster ids to the three species (as in k-means)
// and compare accuracies.  Average linkage tends to give tight, globular
// clusters, usually matching k-means quality on iris.
// sklearn equivalent: sklearn.cluster.AgglomerativeClustering(
//   n_clusters=3, linkage='average').
//
// Output depends on the current implementation; saved results are regenerated
// by the lesson.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Agglomerative.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "helper/plot/plot_svg.hpp"

int main() {
  std::cout << std::setprecision(7);

  auto ds = ml::load_iris(std::string(DATA_DIR));
  const size_t k = 3;

  ml::AgglomerativeClustering ac(k, ml::Linkage::Average);
  ac.fit(ds.X);
  std::cout << "iris " << ds.n() << " x " << ds.p() << ", k = " << k
            << ", average linkage\n";

  std::vector<size_t> sizes(k, 0);
  for (size_t i = 0; i < ds.n(); ++i)
    ++sizes[static_cast<size_t>(ac.labels()[i])];
  std::cout << "cluster sizes:";
  for (size_t j = 0; j < k; ++j) std::cout << " " << sizes[j];
  std::cout << "\n";

  double best_acc = 0.0;
  double best_cnt = 0.0;
  std::vector<std::vector<double>> overlap(k, std::vector<double>(k, 0.0));
  for (size_t i = 0; i < ds.n(); ++i)
    overlap[static_cast<size_t>(ac.labels()[i])]
           [static_cast<size_t>(ds.y[i])] += 1.0;
  std::vector<size_t> perm = {0, 1, 2};
  do {
    double ok = 0.0;
    for (size_t c = 0; c < k; ++c) ok += overlap[c][perm[c]];
    if (ok > best_cnt) {
      best_cnt = ok;
      best_acc = ok / static_cast<double>(ds.n());
    }
  } while (std::next_permutation(perm.begin(), perm.end()));
  std::cout << "species best-match accuracy " << best_acc << "\n";
  std::cout << "merges performed: " << ac.merge_history().size() << "\n";
  std::cout << "last merge distance " << ac.merge_history().back().second
            << "\n";
  std::cout << "silhouette score " << ml::silhouette(ds.X, ac.labels()) << "\n";

  ml::Plot fig(600, 380);
  fig.title("average-linkage agglomerative clusters on iris (petals)");
  fig.xlabel(ds.feature_names[2]);
  fig.ylabel(ds.feature_names[3]);
  ml::Mat xy(ds.n(), ml::Vec(2));
  for (size_t i = 0; i < ds.n(); ++i) {
    xy[i][0] = ds.X[i][2];
    xy[i][1] = ds.X[i][3];
  }
  fig.scatter(xy, ac.labels(), "clusters");

  std::string out = std::string(RUN_OUTPUT_DIR) + "/iris_average.svg";
  std::ofstream f(out);
  f << fig.render();
  f.close();
  std::cout << "figure -> " << out << "\n";

  // Save labels for inspection; automated numerical fixtures live in tests/.
  std::ofstream lb(std::string(RUN_OUTPUT_DIR) +
                   "/iris_agglomerative_avg.labels");
  for (double l : ac.labels()) lb << l << "\n";
  lb.close();
  return 0;
}
