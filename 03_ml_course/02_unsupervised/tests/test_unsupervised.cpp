// 03_ml_course/02_unsupervised/tests/test_unsupervised.cpp
// Purpose: numerical and persistence checks for every unsupervised and
//          anomaly-detection model (silhouette, k-means, agglomerative, DBSCAN,
//          GMM, PCA, Isolation Forest, LOF, t-SNE, One-Class SVM, ARI) on tiny
//          hand-checkable fixtures. Any failed check throws, so the process
//          exits non-zero and CTest reports the failure.
// Inputs:  none (all fixtures are built in code; seeds 3 and 11 for blobs).
// Outputs: prints only ("All unsupervised numerical and persistence checks
//          passed" on success).
// Run target: unsupervised_tests (ctest -R unsupervised_numerical).
#include <iostream>
#include <random>
#include <sstream>

#include "02_unsupervised/01_kmeans/KMeans.hpp"
#include "02_unsupervised/02_agglomerative/Agglomerative.hpp"
#include "02_unsupervised/04_dbscan/DBSCAN.hpp"
#include "02_unsupervised/05_gaussian_mixture/GaussianMixture.hpp"
#include "02_unsupervised/06_pca/PCA.hpp"
#include "02_unsupervised/07_tsne/TSNE.hpp"
#include "02_unsupervised/workflow.hpp"
#include "03_anomaly_detection/01_isolation_forest/IsolationForest.hpp"
#include "03_anomaly_detection/02_local_outlier_factor/LocalOutlierFactor.hpp"
#include "03_anomaly_detection/03_one_class_svm/OneClassSVM.hpp"
// Minimal assertion: throw with a message naming the failed check.
void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
// Assert that the callable throws a std::exception (input validation works).
template <class F>
void rejects(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, "expected rejection");
}
int main() {
  using namespace ml;
  // Shared 1-D fixture: two pairs {0,2} and {8,10}, natural 2-cluster split.
  Mat X{{0}, {2}, {8}, {10}};
  Vec y{0, 0, 1, 1};
  // --- silhouette -------------------------------------------------------
  // Hand value: for point 0, a = 2, b = 9 -> 7/9; point 2: a = 2, b = 7 -> 5/7;
  // symmetric for the other pair; mean = 47/63.
  require(std::abs(silhouette(X, y) - 47.0 / 63) < 1e-12, "hand silhouette");
  // A singleton cluster has no intra-cluster distance: its silhouette is 0.
  auto s = silhouette_samples({{0}, {2}, {8}}, {0, 0, 1});
  require(s[2] == 0, "singleton score");
  // Silhouette is undefined for 1 cluster or n clusters: must throw.
  rejects([&] { silhouette(X, {0, 0, 0, 0}); });
  rejects([&] { silhouette(X, {0, 1, 2, 3}); });
  // All points identical: max(a, b) = 0, guard returns 0 instead of NaN.
  require(silhouette({{1}, {1}, {1}}, {0, 0, 1}) == 0, "zero denominator");
  // --- k-means ----------------------------------------------------------
  KMeans km(2);
  km.fit(X);
  // Optimal centroids 1 and 9: inertia = 1 + 1 + 1 + 1 = 4.
  require(std::abs(km.inertia() - 4) < 1e-10, "kmeans objective");
  // ARI = 1 means the partition matches y up to label renaming.
  require(unsup::adjusted_rand(y, km.labels()) == 1, "kmeans partition");
  // Save/load round-trip must reproduce the labels through predict().
  std::stringstream ss;
  km.save(ss);
  KMeans copy;
  copy.load(ss);
  require(copy.predict(X) == km.labels(), "kmeans reload");
  // n_init = 0 is invalid; predicting 2-D rows with a 1-D model is invalid.
  rejects([&] {
    KMeans bad(2, 0);
    bad.fit(X);
  });
  rejects([&] { km.predict({{1, 2}}); });
  // k = 3 on three identical points: every centroid sits on the point.
  KMeans duplicate(3);
  duplicate.fit({{1}, {1}, {1}});
  require(duplicate.inertia() == 0, "duplicate inertia");
  // --- agglomerative ----------------------------------------------------
  for (Linkage l :
       {Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward}) {
    AgglomerativeClustering m(2, l);
    m.fit(X);
    // n = 4 leaves always give n - 1 = 3 merges.
    require(m.linkage_matrix().size() == 3, "complete dendrogram");
    require(unsup::adjusted_rand(y, m.labels()) == 1, "hierarchy partition");
    // Height of the final merge between {0,2} and {8,10}: single = min gap 6,
    // complete = max gap 10, average = mean of {8,10,6,8} = 8, Ward =
    // sqrt(2 * 2 * 2 / (2 + 2)) * |1 - 9| ... = sqrt(128).
    double final = m.linkage_matrix().back()[2];
    double expected = l == Linkage::Single     ? 6
                      : l == Linkage::Complete ? 10
                      : l == Linkage::Average  ? 8
                                               : std::sqrt(128.0);
    require(std::abs(final - expected) < 1e-12, "linkage distance");
    // Cutting at one cluster labels everything 0.
    require(m.cut(1) == Vec(4, 0), "one cluster cut");
    // A reloaded tree cut at 2 must equal the fitted labels.
    std::stringstream state;
    m.save(state);
    AgglomerativeClustering restored;
    restored.load(state);
    require(restored.cut(2) == m.labels(), "hierarchy reload");
  }
  // --- DBSCAN -----------------------------------------------------------
  // eps 2.1, min_samples 2: pairs form clusters, the point at 30 is noise.
  DBSCAN db(2.1, 2);
  db.fit({{0}, {2}, {8}, {10}, {30}});
  require(db.labels() == Vec({0, 0, 1, 1, -1}), "density labels");
  require(db.core_samples().back() == 0, "noise not core");
  // eps too small for any neighbour: every point is noise.
  DBSCAN allnoise(.1, 3);
  allnoise.fit(X);
  require(allnoise.labels() == Vec(4, -1), "all noise");
  rejects([&] {
    DBSCAN bad(0, 2);  // eps must be > 0
    bad.fit(X);
  });
  // --- Gaussian mixture -------------------------------------------------
  GaussianMixture gm(2);
  gm.fit(X);
  // Responsibilities are a probability distribution over components.
  auto R = gm.predict_proba(X);
  for (const auto& row : R)
    require(std::abs(std::accumulate(row.begin(), row.end(), 0.0) - 1) < 1e-12,
            "responsibility normalization");
  require(std::isfinite(gm.bic(X)), "finite BIC");
  std::stringstream gs;
  gm.save(gs);
  GaussianMixture gr;
  gr.load(gs);
  require(gr.predict(X) == gm.predict(X), "mixture reload");
  // Identical points would give zero variance; the reg floor keeps the
  // log-likelihood finite instead of +inf.
  GaussianMixture flat(2);
  flat.fit({{1}, {1}, {1}});
  require(std::isfinite(flat.score({{1}})), "regularized singular mixture");
  // --- PCA --------------------------------------------------------------
  PCA pca(1);
  // Points on the diagonal: one axis explains 100% of the variance and the
  // 1-component reconstruction is exact.
  Mat line{{-1, -1}, {0, 0}, {1, 1}};
  pca.fit(line);
  require(std::abs(pca.explained_variance_ratio()[0] - 1) < 1e-12, "PCA ratio");
  auto back = pca.inverse_transform(pca.transform(line));
  for (size_t i = 0; i < line.size(); ++i)
    require(distance2(back[i], line[i]) < 1e-20, "PCA reconstruction");
  // The training mean projects to the origin (centring is stored).
  Mat train{{10, 10}, {11, 11}, {12, 12}};
  pca.fit(train);
  require(std::abs(pca.transform({{11, 11}})[0][0]) < 1e-12,
          "training mean projection");
  // --- Isolation Forest -------------------------------------------------
  // 100 "ordinary" points in the unit square; (20, 20) is far outside and
  // must isolate in fewer splits, i.e. get a higher anomaly score.
  Mat ordinary;
  for (int i = 0; i < 100; ++i)
    ordinary.push_back({double(i) / 100, double(i % 7) / 7});
  IsolationForest forest(100, 64);
  forest.fit(ordinary);
  auto anomaly = forest.score_samples({{.5, .5}, {20, 20}});
  require(anomaly[1] > anomaly[0], "outlier path shorter");
  // Threshold and every tree survive the save/load round-trip exactly.
  forest.set_threshold(.7);
  std::stringstream fs;
  forest.save(fs);
  IsolationForest fr;
  fr.load(fs);
  require(
      fr.threshold() == .7 && fr.predict(ordinary) == forest.predict(ordinary),
      "forest threshold reload");
  require(fr.score_samples(ordinary) == forest.score_samples(ordinary),
          "forest reload");
  // --- Local Outlier Factor ---------------------------------------------
  // Local Outlier Factor: a 5x4 grid of tightly spaced points plus one far
  // point.  With k = 5 the far point's neighbourhood is far sparser than its
  // neighbours' neighbourhoods.
  Mat cluster;
  for (int i = 0; i < 20; ++i)
    cluster.push_back({double(i % 5) / 10, double(i / 5) / 10});
  const size_t far = cluster.size();
  cluster.push_back({10, 10});
  LocalOutlierFactor lof(5);
  lof.fit(cluster);
  const Vec& factors = lof.fitted_scores();
  // LOF ~ 1 for inliers; the far point must exceed 2 and outrank every inlier.
  require(factors[far] > 2, "far point factor above 2");
  for (size_t i = 0; i < far; ++i)
    require(factors[far] > factors[i], "far point outranks cluster");
  // Novelty scoring of the same coordinates gives the same verdict.
  require(lof.score_samples({{10, 10}})[0] > 2, "novelty far factor");
  // With threshold 2 exactly one of the 21 rows is flagged.
  lof.set_threshold(2);
  auto lof_flags = lof.predict(cluster);
  require(lof_flags[far] == 1 && mean(lof_flags) == 1.0 / cluster.size(),
          "only far point flagged");
  std::stringstream ls;
  lof.save(ls);
  LocalOutlierFactor lr;
  lr.load(ls);
  require(lr.threshold() == 2 && lr.k() == 5, "LOF parameter reload");
  require(lr.score_samples(cluster) == lof.score_samples(cluster),
          "LOF reload reproduces scores");
  // k larger than n - 1 is clamped rather than rejected.
  LocalOutlierFactor clamped(100);
  clamped.fit(X);
  require(clamped.k() == X.size() - 1, "k clamped to n-1");
  // A single row has no neighbours; threshold must be > 0; width mismatch.
  rejects([&] {
    LocalOutlierFactor tiny(3);
    tiny.fit({{1}});
  });
  rejects([&] { lof.set_threshold(0); });
  rejects([&] { lof.score_samples({{1}}); });
  // All-duplicate rows give zero reachability distances; the epsilon guard
  // must keep the factor finite instead of 0/0.
  LocalOutlierFactor duplicates(2);
  duplicates.fit({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
  for (double v : duplicates.score_samples({{1, 1}}))
    require(std::isfinite(v), "epsilon guard on zero distances");
  // --- t-SNE ------------------------------------------------------------
  // t-SNE: three well separated Gaussian clusters must stay separated in the
  // embedding (centroid distances exceed within-cluster spread).
  {
    std::mt19937 gen(3);
    std::normal_distribution<double> noise(0, .1);
    Mat blobs;
    Vec labels;
    const Mat centres{{0, 0, 0}, {10, 0, 0}, {0, 10, 0}};
    for (size_t c = 0; c < centres.size(); ++c)
      for (int i = 0; i < 15; ++i) {
        blobs.push_back({centres[c][0] + noise(gen), centres[c][1] + noise(gen),
                         centres[c][2] + noise(gen)});
        labels.push_back(double(c));
      }
    TSNE tsne(2, 5, 300, 50);  // small n: a modest learning rate
    tsne.fit(blobs);
    const Mat Y = tsne.embedding();
    require(Y.size() == blobs.size() && Y[0].size() == 2, "embedding shape");
    // Embedding centroids per true cluster and the largest within-cluster
    // radius; every centroid pair must be more than twice that radius apart.
    Mat cent(3, Vec(2, 0));
    for (size_t i = 0; i < Y.size(); ++i)
      for (size_t d = 0; d < 2; ++d) cent[size_t(labels[i])][d] += Y[i][d] / 15;
    double spread = 0;
    for (size_t i = 0; i < Y.size(); ++i)
      spread = std::max(spread,
                        std::sqrt(distance2(Y[i], cent[size_t(labels[i])])));
    for (size_t c = 0; c < 3; ++c)
      for (size_t k = c + 1; k < 3; ++k)
        require(std::sqrt(distance2(cent[c], cent[k])) > 2 * spread,
                "t-SNE clusters separated");
    // KL divergence is a non-negative finite number after fitting.
    require(std::isfinite(tsne.kl_divergence()) && tsne.kl_divergence() >= 0,
            "finite KL");
    // Neighbourhoods (k = 5) are preserved almost perfectly on clean blobs.
    require(trustworthiness(blobs, Y, 5) > .9, "trustworthiness high");
    // The snapshot stores coordinates + parameters; refitting from the same
    // seed reproduces the identical embedding.
    std::stringstream ts;
    tsne.save(ts);
    TSNE tr;
    tr.load(ts);
    require(tr.embedding() == Y && tr.perplexity() == 5 && tr.seed() == 42,
            "t-SNE reload exact");
    tr.fit(blobs);
    require(tr.embedding() == Y, "same seed reproduces embedding");
    rejects([&] {
      TSNE big(2, 4);
      big.fit(X);  // perplexity >= n
    });
    rejects([&] { TSNE().embedding(); });  // embedding before fit
  }
  // --- One-Class SVM ----------------------------------------------------
  // One-class SVM: planted outliers score above every inlier; the fraction of
  // training rows flagged at the boundary is close to nu; reload is exact.
  {
    std::mt19937 gen(11);
    std::normal_distribution<double> noise(0, 1);
    Mat blob;
    for (int i = 0; i < 200; ++i) blob.push_back({noise(gen), noise(gen)});
    const size_t first_outlier = blob.size();
    blob.push_back({6, 6});
    blob.push_back({-6, 5});
    blob.push_back({7, -2});
    OneClassSVM svm(.1, .5);  // nu = 0.1, RBF gamma = 0.5
    svm.fit(blob);
    const Vec sc = svm.score_samples(blob);
    double max_inlier = -1e300;
    for (size_t i = 0; i < first_outlier; ++i)
      max_inlier = std::max(max_inlier, sc[i]);
    for (size_t i = first_outlier; i < blob.size(); ++i)
      require(sc[i] > max_inlier, "planted outlier outranks inliers");
    // nu upper-bounds the training outlier fraction; allow a small slack.
    svm.set_threshold(0);
    const double flagged = mean(svm.predict(blob));
    require(flagged <= .1 + .03 && flagged >= .1 - .05, "flag rate near nu");
    // nu also lower-bounds the support-vector fraction.
    require(svm.support_count() >= size_t(.1 * blob.size()),
            "support vectors at least nu n");
    std::stringstream os;
    svm.save(os);
    OneClassSVM osr;
    osr.load(os);
    require(osr.nu() == .1 && osr.gamma() == .5 && osr.rho() == svm.rho(),
            "OCSVM parameter reload");
    require(osr.score_samples(blob) == sc, "OCSVM reload reproduces scores");
    rejects([&] {
      OneClassSVM bad(0);  // nu must lie in (0, 1]
      bad.fit(blob);
    });
    rejects([&] { svm.score_samples({{1}}); });     // width mismatch
    rejects([&] { OneClassSVM().score_samples(blob); });  // before fit
  }
  // --- ARI --------------------------------------------------------------
  // Swapping label names does not change the partition: ARI stays 1.
  require(unsup::adjusted_rand({0, 0, 1, 1}, {1, 1, 0, 0}) == 1,
          "label permutation invariance");
  std::cout << "All unsupervised numerical and persistence checks passed\n";
}
