// 03_ml_course/02_unsupervised/04_dbscan/03_implementation.cpp
// Purpose: lesson showing DBSCAN on a tiny hand-checkable 1-D line, then on a
//          2-D synthetic set (two dense blobs plus uniform background noise).
//          Noise points get label -1; an eps / min_samples sweep shows how the
//          two knobs trade cluster count against the noise fraction.
// Inputs:  none from disk; both datasets are generated in code (seed 5).
// Outputs: results/03_implementation_results/{labels.csv,sweep.csv,
//          figures/dbscan_scatter.svg}; also prints labels and the sweep.
// Run target: udb_implementation.
// sklearn equivalent: sklearn.cluster.DBSCAN(eps, min_samples).

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "DBSCAN.hpp"
#include "helper/reporting/artifacts.hpp"

namespace {
// Counts derived from a label vector: number of clusters and noise rows.
struct Summary {
  size_t clusters = 0, noise = 0;
};
// Summarise DBSCAN output. Cluster ids are 0..top, so #clusters = top + 1;
// rows with a negative label are noise. O(n).
Summary summarise(const ml::Vec& labels) {
  Summary s;
  double top = -1;  // stays -1 (0 clusters) if every row is noise
  for (double l : labels) {
    if (l < 0) ++s.noise;
    top = std::max(top, l);
  }
  s.clusters = size_t(top + 1);
  return s;
}
}  // namespace

int main() {
  using namespace ml;
  // Part 1 - four points on a line: {0,2} are 2 apart, {8,10} are 2 apart, and
  // the gap between the pairs is 6. With eps 2.1 and min_samples 2 (self counts)
  // each pair is a cluster, so the expected labels are "0 0 1 1".
  Mat X{{0, 0}, {2, 0}, {8, 0}, {10, 0}};
  DBSCAN model(2.1, 2);
  model.fit(X);
  for (double y : model.labels()) std::cout << y << ' ';
  std::cout << '\n';

  // Part 2 - synthetic 2-D data: 80 points around (0,0), 80 around (6,4)
  // (both with sd 0.4) and 20 uniform background points over [-3,9]^2.
  // Fixed seed 5 keeps the numbers reproducible across runs.
  Artifacts a(RUN_OUTPUT_DIR, ".");
  std::mt19937 gen(5);
  std::normal_distribution<double> tight(0, .4);
  std::uniform_real_distribution<double> bg(-3, 9);
  Mat S;
  for (int i = 0; i < 80; ++i) S.push_back({tight(gen), tight(gen)});
  for (int i = 0; i < 80; ++i) S.push_back({6 + tight(gen), 4 + tight(gen)});
  for (int i = 0; i < 20; ++i) S.push_back({bg(gen), bg(gen)});

  // eps 0.6 is ~1.5 blob standard deviations: inside a blob most points have
  // many neighbours, while isolated background points have almost none.
  const double eps = .6;
  const size_t min_samples = 5;
  DBSCAN dense(eps, min_samples);
  dense.fit(S);
  const Vec& lab = dense.labels();
  const Summary sm = summarise(lab);
  std::cout << "synthetic: eps " << eps << " min_samples " << min_samples
            << " -> " << sm.clusters << " clusters, " << sm.noise
            << " noise points of " << S.size() << '\n';

  // Per-row table with the label plus the core/noise flags (17 digits so the
  // coordinates round-trip exactly).
  std::ostringstream csv;
  csv << std::setprecision(17) << "row,x,y,label,is_core,is_noise\n";
  for (size_t i = 0; i < S.size(); ++i)
    csv << i << ',' << S[i][0] << ',' << S[i][1] << ',' << lab[i] << ','
        << dense.core_samples()[i] << ',' << (lab[i] < 0 ? 1 : 0) << '\n';
  a.write("labels.csv", csv.str());

  // Shift labels by one so noise (-1) becomes class 0 and gets a legend name.
  Vec shifted(lab.size());
  for (size_t i = 0; i < lab.size(); ++i) shifted[i] = lab[i] + 1;
  std::vector<std::string> names{"noise"};
  for (size_t c = 0; c < sm.clusters; ++c)
    names.push_back("cluster " + std::to_string(c));
  Plot scatter;
  scatter.title("DBSCAN (eps 0.6, min_samples 5): noise marked");
  scatter.xlabel("x");
  scatter.ylabel("y");
  scatter.class_labels(names);
  scatter.scatter(S, shifted, "label");
  a.figure("figures/dbscan_scatter.svg", scatter);

  // Part 3 - parameter sweep (4 eps x 3 min_samples = 12 fits). Small eps or
  // large min_samples fragments blobs and grows the noise count; large eps
  // eventually merges both blobs (and the background) into one cluster.
  std::ostringstream sweep;
  sweep << std::setprecision(17) << "eps,min_samples,n_clusters,n_noise\n";
  std::cout << "eps / min_samples sweep (clusters, noise)\n";
  for (double e : {.3, .6, 1.0, 1.5})
    for (size_t m : {3u, 5u, 10u}) {
      DBSCAN d(e, m);
      d.fit(S);
      const Summary r = summarise(d.labels());
      sweep << e << ',' << m << ',' << r.clusters << ',' << r.noise << '\n';
      std::cout << "  eps " << e << " min " << m << " -> " << r.clusters << ", "
                << r.noise << '\n';
    }
  a.write("sweep.csv", sweep.str());
  std::cout << "Saved labels.csv, sweep.csv and figures under "
            << RUN_OUTPUT_DIR << '\n';
}
