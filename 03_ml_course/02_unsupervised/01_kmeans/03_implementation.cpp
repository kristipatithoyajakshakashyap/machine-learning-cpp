// 03_implementation.cpp
// Lesson: k-means implemented on a real dataset (iris).
// Unsupervised means NO labels are given to the algorithm.  On iris we KNOW
// the three species (setosa/versicolor/virginica), so afterwards we can ask
// "did k=3 recover the three species?"  Because cluster ids are arbitrary,
// we match them to species by best-mapping and report the accuracy.
// sklearn equivalent: sklearn.cluster.KMeans(n_clusters=3, random_state=42).
//
// Output depends on the current implementation; saved results are regenerated
// by the lesson.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "KMeans.hpp"
#include "helper/data/datasets.hpp"
#include "helper/math/metrics.hpp"
#include "helper/plot/plot_svg.hpp"

int main() {
  std::cout << std::setprecision(7);

  auto ds = ml::load_iris(std::string(DATA_DIR));
  const size_t k = 3;

  // Best-match the k cluster ids against the true species labels.
  ml::KMeans km(k, 10, 300, 42);
  km.fit(ds.X);
  std::cout << "iris " << ds.n() << " x " << ds.p() << ", k = " << k
            << ", seed 42\n";

  // Find the species id that each cluster maps to (greedy best overlap);
  // then report the best-match accuracy over the 3! id permutations.
  std::vector<size_t> sizes(k, 0);
  for (size_t i = 0; i < ds.n(); ++i)
    ++sizes[static_cast<size_t>(km.labels()[i])];
  std::cout << "cluster sizes:";
  for (size_t j = 0; j < k; ++j) std::cout << " " << sizes[j];
  std::cout << "\n";

  double best_acc = 0.0;
  double best_cnt = 0.0;  // compare in count units, then divide once
  // overlap[c][s] = how many rows in cluster c have true species s.
  std::vector<std::vector<double>> overlap(k, std::vector<double>(k, 0.0));
  for (size_t i = 0; i < ds.n(); ++i)
    overlap[static_cast<size_t>(km.labels()[i])]
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
  std::cout << "inertia " << km.inertia() << "  converged in " << km.n_iter()
            << " iterations\n";
  std::cout << "silhouette score " << ml::silhouette(ds.X, km.labels()) << "\n";

  // scatter of petal width vs petal length coloured by cluster
  ml::Plot fig(600, 380);
  fig.title("k-means clusters on iris (petal features)");
  fig.xlabel(ds.feature_names[2]);
  fig.ylabel(ds.feature_names[3]);
  ml::Mat xy(ds.n(), ml::Vec(2));
  for (size_t i = 0; i < ds.n(); ++i) {
    xy[i][0] = ds.X[i][2];
    xy[i][1] = ds.X[i][3];
  }
  fig.scatter(xy, km.labels(), "clusters");

  std::string out = std::string(RUN_OUTPUT_DIR) + "/iris_kmeans.svg";
  std::ofstream f(out);
  f << fig.render();
  f.close();
  std::cout << "figure -> " << out << "\n";

  // Save labels for inspection; automated numerical fixtures live in tests/.
  std::ofstream lb(std::string(RUN_OUTPUT_DIR) + "/iris_kmeans3.labels");
  for (double l : km.labels()) lb << l << "\n";
  lb.close();
  return 0;
}
