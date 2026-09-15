// 03_ml_course/02_unsupervised/workflow.hpp
// Purpose: the shared end-to-end pipeline pieces used by every clustering module
//          (k-means, agglomerative, cluster evaluation, DBSCAN, GMM, PCA, t-SNE)
//          and reused by the anomaly-detection modules: run manifest, evaluation
//          report, subsample stability check, model snapshot and reload check.
// Inputs:  a loaded ml::Dataset (Palmer Penguins) plus a standardized matrix and
//          cluster labels; all paths are relative to an Artifacts root
//          (RUN_OUTPUT_DIR of the calling end_to_end target).
// Outputs: run_manifest.json, model/feature_schema.csv, evaluation/*.csv|json,
//          evaluation/figures/*.svg, validation/stability.csv,
//          model/model_state.txt, model/preprocessing_state.txt,
//          inference/reload_verification.json, report.md, execution.log.
// Run target: not an executable; included by every *_end_to_end.cpp and by
//             tests/verify_reload.cpp.
#pragma once
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

#include "helper/math/metrics.hpp"
#include "helper/preprocessing/preprocessor.hpp"
#include "helper/reporting/artifacts.hpp"
namespace ml {
namespace unsup {
// Adjusted Rand Index between two labelings of the same rows (Hubert & Arabie).
// Parameters: a, b - integer-valued labels stored as doubles, same length >= 2.
// Returns: 1 for identical partitions (up to renaming), ~0 for random agreement,
//          negative for worse-than-chance. Label values themselves are irrelevant.
// Throws invalid_argument on length mismatch. Complexity O(n log n) (map inserts).
inline double adjusted_rand(const Vec& a, const Vec& b) {
  if (a.size() != b.size() || a.size() < 2)
    throw std::invalid_argument("ARI needs matched rows");
  // Contingency table cells n_ij plus row and column marginals.
  std::map<std::pair<int, int>, double> cells;
  std::map<int, double> rows, cols;
  for (size_t i = 0; i < a.size(); ++i) {
    ++cells[{int(a[i]), int(b[i])}];
    ++rows[int(a[i])];
    ++cols[int(b[i])];
  }
  // Number of unordered pairs inside a group of size n.
  auto pairs = [](double n) { return n * (n - 1) / 2; };
  double s = 0, r = 0, c = 0;
  for (auto x : cells) s += pairs(x.second);
  for (auto x : rows) r += pairs(x.second);
  for (auto x : cols) c += pairs(x.second);
  // ARI = (index - expected) / (max_index - expected); guard the degenerate
  // case where both partitions are trivial (denominator 0 -> perfect agreement).
  double expected = r * c / pairs(double(a.size())),
         den = .5 * (r + c) - expected;
  return std::abs(den) < 1e-15 ? 1 : (s - expected) / den;
}
// Silhouette that is only defined for 2 <= #clusters < n; otherwise NaN so the
// caller can skip it (single cluster or every row its own cluster).
inline double valid_silhouette(const Mat& X, const Vec& labels) {
  std::set<double> unique(labels.begin(), labels.end());
  return unique.size() > 1 && unique.size() < X.size()
             ? silhouette(X, labels)
             : std::numeric_limits<double>::quiet_NaN();
}
// Stage "manifest": record what was run so a results folder is self-describing.
// Writes run_manifest.json (algorithm, seed 42, dataset name, data fingerprint,
// validation protocol) and model/feature_schema.csv (feature order the saved
// preprocessor and model expect; predict programs rely on this order).
inline void manifest(const Dataset& ds, Artifacts& a,
                     const std::string& algorithm) {
  a.write("run_manifest.json",
          "{\"algorithm\":" + json_quote(algorithm) +
              ",\"seed\":42,\"dataset\":\"Palmer Penguins\",\"fingerprint\":" +
              json_quote(fingerprint(ds)) +
              ",\"validation\":\"internal selection and five 80 percent "
              "subsample stability runs\"}\n");
  std::ostringstream schema;
  schema << "feature\n";
  for (const auto& s : ds.feature_names) schema << csv_quote(s) << '\n';
  a.write("model/feature_schema.csv", schema.str());
}
// Stage "evaluation": turn final labels into tables, figures and a report.
// Parameters: ds - raw dataset (original units, species in ds.y);
//             X - standardized features the model was fitted on;
//             labels - cluster id per row, negative = noise (DBSCAN);
//             a - artifact root; choice - human-readable selected configuration.
// Writes evaluation/assignments.csv, evaluation/metrics.json,
// evaluation/figures/{clusters,silhouette,silhouette_distribution}.svg,
// evaluation/cluster_profiles.csv, report.md and execution.log.
// Noise rows are excluded from silhouette but counted in noise_fraction.
inline void report(const Dataset& ds, const Mat& X, const Vec& labels,
                   Artifacts& a, const std::string& choice) {
  // Drop noise rows (label < 0) before computing silhouettes.
  Mat retained;
  Vec kept;
  size_t noise = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] < 0) {
      ++noise;
      continue;
    }
    retained.push_back(X[i]);
    kept.push_back(labels[i]);
  }
  double sil = valid_silhouette(retained, kept);
  Vec samples;
  if (std::isfinite(sil)) samples = silhouette_samples(retained, kept);
  // Per-row table: cluster, its silhouette (blank for noise), external species.
  std::ostringstream csv;
  csv << "row_id,cluster,silhouette,external_species\n";
  size_t k = 0;  // walks `samples`, which only covers non-noise rows
  for (size_t i = 0; i < labels.size(); ++i) {
    csv << (ds.row_ids.empty() ? i : ds.row_ids[i]) << ',' << labels[i] << ',';
    if (labels[i] >= 0 && std::isfinite(sil)) csv << samples[k++];
    csv << ',' << ds.y[i] << '\n';
  }
  a.write("evaluation/assignments.csv", csv.str());
  // Summary metrics: silhouette (null when undefined), noise share, external ARI.
  std::ostringstream metrics;
  metrics << "{\"silhouette\":";
  if (std::isfinite(sil))
    metrics << sil;
  else
    metrics << "null";
  metrics << ",\"noise_fraction\":" << double(noise) / labels.size()
          << ",\"external_ARI\":" << adjusted_rand(ds.y, labels) << "}\n";
  a.write("evaluation/metrics.json", metrics.str());
  // 2-D scatter of the first two standardized features coloured by cluster.
  Mat xy;
  for (const auto& row : X) xy.push_back({row[0], row[1]});
  Plot scatter;
  scatter.title("Selected clusters, standardized first two features");
  scatter.scatter(xy, labels);
  a.figure("evaluation/figures/clusters.svg", scatter);
  if (!samples.empty()) {
    // Hand-built silhouette plot: one 2px bar per row, grouped and sorted by
    // cluster, zero line at x=380, +/-1 mapped to +/-300px, mean as dashed line.
    std::map<int, Vec> by_cluster;
    for (size_t i = 0; i < samples.size(); ++i)
      by_cluster[int(kept[i])].push_back(samples[i]);
    double height = 100 + samples.size() * 2 + by_cluster.size() * 20;
    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\""
        << height << "\" viewBox=\"0 0 760 " << height
        << "\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><text "
           "x=\"40\" y=\"25\" font-size=\"18\">Silhouette by cluster (noise "
           "excluded)</text><line x1=\"380\" x2=\"380\" y1=\"45\" y2=\""
        << height - 35 << "\" stroke=\"#888\"/>";
    const char* colors[] = {"#2563eb", "#059669", "#d97706",
                            "#7c3aed", "#db2777", "#0891b2"};
    double y = 50;
    for (auto& entry : by_cluster) {
      std::sort(entry.second.begin(), entry.second.end());
      svg << "<text x=\"10\" y=\"" << y + 10 << "\" font-size=\"12\">C"
          << entry.first << "</text>";
      for (double value : entry.second) {
        // Negative silhouettes extend left of the zero line.
        svg << "<rect x=\"" << 380 + 300 * std::min(0.0, value) << "\" y=\""
            << y << "\" width=\"" << 300 * std::abs(value)
            << "\" height=\"2\" fill=\"" << colors[size_t(entry.first) % 6]
            << "\"/>";
        y += 2;
      }
      y += 20;
    }
    svg << "<line x1=\"" << 380 + 300 * sil << "\" x2=\"" << 380 + 300 * sil
        << "\" y1=\"45\" y2=\"" << height - 35
        << "\" stroke=\"#111\" stroke-dasharray=\"5 4\"/><text x=\"80\" y=\""
        << height - 10 << "\">-1</text><text x=\"380\" y=\"" << height - 10
        << "\">0</text><text x=\"680\" y=\"" << height - 10
        << "\">1</text></svg>";
    a.write("evaluation/figures/silhouette.svg", svg.str());
    Plot h;
    h.title("Per-row silhouette distribution (noise excluded)");
    h.histogram(samples, 25);
    a.figure("evaluation/figures/silhouette_distribution.svg", h);
  }
  // Cluster profiles: per-cluster feature means in ORIGINAL units, skipping
  // missing values, so clusters can be interpreted without the standardization.
  std::map<int, Vec> sums, counts;
  for (size_t i = 0; i < ds.n(); ++i) {
    int c = int(labels[i]);
    if (!sums.count(c)) {
      sums[c] = Vec(ds.p(), 0);
      counts[c] = Vec(ds.p(), 0);
    }
    for (size_t j = 0; j < ds.p(); ++j)
      if (std::isfinite(ds.X[i][j])) {
        sums[c][j] += ds.X[i][j];
        ++counts[c][j];
      }
  }
  std::ostringstream profiles;
  profiles << "cluster,feature,mean_original_units\n";
  for (auto& entry : sums)
    for (size_t j = 0; j < ds.p(); ++j)
      profiles << entry.first << ',' << ds.feature_names[j] << ','
               << entry.second[j] / std::max(1.0, counts[entry.first][j])
               << '\n';
  a.write("evaluation/cluster_profiles.csv", profiles.str());
  a.write(
      "report.md",
      "# Cluster experiment\n\nSelected configuration: " + choice +
          ".\n\nSilhouette assesses separation in the standardized feature "
          "space; it does not prove biological species recovery. External "
          "species labels were excluded from fitting and selection. ARI is "
          "descriptive external evaluation after selection. Noise is excluded "
          "from silhouette and its coverage is reported. The first two-feature "
          "projection can hide separation in other dimensions. Examine cluster "
          "profiles in original units and negative silhouettes before "
          "assigning meanings to clusters. These full-dataset exploratory "
          "results are not hold-out prediction estimates.\n");
  a.write("execution.log",
          "Completed EDA, internal selection, fitted-state save, evaluation "
          "and reload verification.\n");
}
// Stage "validation": subsample stability. Five times, take a seeded random 80%
// of the rows (seeds 100..104), re-standardize and refit with `fit`, then score
// agreement with the full-data labels by ARI. Stable clusterings score near 1.
// Parameters: ds - raw dataset; baseline - full-fit labels; fit - callable
//             Mat -> Vec producing labels for a standardized subsample.
// Writes validation/stability.csv. Complexity: 5 x cost of fit.
template <class Fit>
inline void stability(const Dataset& ds, const Vec& baseline, Artifacts& a,
                      Fit fit) {
  std::vector<size_t> ids(ds.n());
  std::iota(ids.begin(), ids.end(), 0);
  std::ostringstream out;
  out << "repeat,rows,adjusted_rand_against_full_fit\n";
  for (unsigned r = 0; r < 5; ++r) {
    std::mt19937 gen(100 + r);
    std::shuffle(ids.begin(), ids.end(), gen);
    std::vector<size_t> subset(ids.begin(), ids.begin() + size_t(.8 * ds.n()));
    auto d = select_rows(ds, subset);
    // The preprocessor is refit on the subsample, as a fresh experiment would.
    Preprocessor p;
    p.fit(d.X);
    auto z = p.transform(d.X);
    Vec reference;
    for (size_t i : subset) reference.push_back(baseline[i]);
    out << r << ',' << subset.size() << ',' << adjusted_rand(reference, fit(z))
        << '\n';
  }
  a.write("validation/stability.csv", out.str());
}
// Stage "snapshot": persist the fitted model and its preprocessor as text
// archives (model/model_state.txt, model/preprocessing_state.txt). Model must
// provide save(std::ostream&) / load(std::istream&) for archive::save_file.
template <class Model>
inline void snapshot(Model& model, const Preprocessor& prep, Artifacts& a) {
  archive::save_file(a.path("model/model_state.txt"), model);
  archive::save_file(a.path("model/preprocessing_state.txt"), prep);
}
// Stage "verify": compare in-memory outputs `a` with outputs `b` from a model
// reloaded from disk. Throws if sizes differ or any |a_i - b_i| > 1e-10, so a
// broken save/load round-trip fails the run instead of silently drifting.
// Writes inference/reload_verification.json with the max absolute error.
inline void verify(const Vec& a, const Vec& b, Artifacts& out) {
  if (a.size() != b.size()) throw std::runtime_error("reload size mismatch");
  double error = 0;
  for (size_t i = 0; i < a.size(); ++i)
    error = std::max(error, std::abs(a[i] - b[i]));
  if (error > 1e-10) throw std::runtime_error("reload changed predictions");
  out.write("inference/reload_verification.json",
            "{\"max_absolute_error\":" + std::to_string(error) + "}\n");
}
}  // namespace unsup
}  // namespace ml
