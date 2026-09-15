// 03_ml_course/02_unsupervised/tests/verify_reload.cpp
// Purpose: fresh-process reload check. Start a NEW process, load the model and
//          preprocessor saved by an end_to_end run, recompute the outputs for
//          the same penguins and compare them with the CSV the run wrote. This
//          proves the snapshot is complete (nothing lives only in memory).
// Inputs:  argv[1] algorithm tag: kmeans | agglomerative | dbscan | gmm | pca |
//          isolation_forest | lof | tsne | ocsvm;
//          argv[2] saved run directory (results/04_end_to_end_results);
//          argv[3] output directory for the verification report;
//          argv[4] data directory holding penguins.csv.
// Outputs: <argv[3]>/fresh_process_verification.json; throws (exit code != 0)
//          on any mismatch above 1e-10.
// Run target: unsupervised_reload (ctest -R <prefix>_fresh_reload; these tests
//             require the matching <prefix>_workflow fixture to run first).
#include <iostream>

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
int main(int argc, char** argv) {
  using namespace ml;
  if (argc != 5)
    throw std::invalid_argument(
        "usage: unsupervised_reload algorithm saved_run_directory "
        "verification_output_directory data_directory");
  std::string type = argv[1];
  Artifacts a(argv[2], ".");       // saved run (read)
  Artifacts report(argv[3], ".");  // verification output (write)
  // Same loader and flags as the end_to_end runs (missing values kept), then
  // the SAVED preprocessor, so rows are imputed/scaled exactly as in training.
  auto ds = load_penguins(argv[4], true);
  Preprocessor p;
  archive::load_file(a.path("model/preprocessing_state.txt"), p);
  Mat X = p.transform(ds.X);
  // Branch A - clustering models: one label per row, compared against
  // evaluation/assignments.csv (row_id, cluster, ...).
  Vec predicted;
  if (type == "kmeans") {
    KMeans m;
    archive::load_file(a.path("model/model_state.txt"), m);
    predicted = m.predict(X);  // nearest centroid
  } else if (type == "agglomerative") {
    AgglomerativeClustering m;
    archive::load_file(a.path("model/model_state.txt"), m);
    predicted = m.labels();  // no predict(): the snapshot stores the labels
  } else if (type == "dbscan") {
    DBSCAN m;
    archive::load_file(a.path("model/model_state.txt"), m);
    predicted = m.labels();  // likewise
  } else if (type == "gmm") {
    GaussianMixture m;
    archive::load_file(a.path("model/model_state.txt"), m);
    predicted = m.predict(X);  // arg-max responsibility
  } else if (type == "pca" || type == "isolation_forest" || type == "lof" ||
             type == "tsne" || type == "ocsvm") {
    // Branch B - models with real-valued outputs (possibly several columns),
    // compared against the CSV named in `file` with a 1e-10 tolerance.
    Mat values;
    std::string file;
    if (type == "pca") {
      PCA m;
      archive::load_file(a.path("model/model_state.txt"), m);
      values = m.transform(X);
      file = "evaluation/projections.csv";  // holds only the test rows
    } else if (type == "isolation_forest") {
      IsolationForest m;
      archive::load_file(a.path("model/model_state.txt"), m);
      for (double v : m.score_samples(X)) values.push_back({v});
      file = "evaluation/predictions.csv";
    } else if (type == "lof") {
      LocalOutlierFactor m;
      archive::load_file(a.path("model/model_state.txt"), m);
      for (double v : m.score_samples(X)) values.push_back({v});
      file = "evaluation/predictions.csv";
    } else if (type == "ocsvm") {
      OneClassSVM m;
      archive::load_file(a.path("model/model_state.txt"), m);
      for (double v : m.score_samples(X)) values.push_back({v});
      file = "evaluation/predictions.csv";
    } else {
      // t-SNE has no transform: refit on the same preprocessed rows with the
      // saved seed and parameters, which must reproduce the embedding.
      TSNE m;
      archive::load_file(a.path("model/model_state.txt"), m);
      m.fit(X);
      values = m.embedding();
      file = "evaluation/embedding.csv";
    }
    // Map row_id -> position, since the CSV may hold a subset (PCA test rows)
    // in any order.
    std::map<size_t, size_t> index;
    for (size_t i = 0; i < ds.n(); ++i)
      index[ds.row_ids.empty() ? i : ds.row_ids[i]] = i;
    std::ifstream in(a.path(file));
    if (!in) throw std::runtime_error("missing saved outputs");
    std::string line;
    std::getline(in, line);  // header
    size_t rows = 0;
    while (std::getline(in, line)) {
      std::stringstream record(line);
      std::string field;
      std::getline(record, field, ',');
      size_t id = std::stoull(field);
      // Compare the first values.size() columns after row_id; extra trailing
      // columns (e.g. external_species) are ignored.
      const auto& actual = values.at(index.at(id));
      for (double v : actual) {
        if (!std::getline(record, field, ','))
          throw std::runtime_error("missing saved value");
        if (std::abs(std::stod(field) - v) > 1e-10)
          throw std::runtime_error("fresh process output mismatch");
      }
      ++rows;
    }
    if (!rows) throw std::runtime_error("no rows verified");
    report.write(
        "fresh_process_verification.json",
        "{\"rows_verified\":" + std::to_string(rows) + ",\"matches\":true}\n");
    std::cout << "Verified " << rows << " rows in fresh process\n";
    return 0;
  } else
    throw std::invalid_argument("unsupported algorithm");
  // Branch A continued: assignments.csv must list every row in order with the
  // same row_id and an exactly equal integer label.
  std::ifstream in(a.path("evaluation/assignments.csv"));
  if (!in) throw std::runtime_error("missing assignments");
  std::string line;
  std::getline(in, line);  // header
  size_t row = 0;
  while (std::getline(in, line)) {
    std::stringstream record(line);
    std::string id, label;
    std::getline(record, id, ',');
    std::getline(record, label, ',');
    if (row >= predicted.size() ||
        std::stoull(id) != (ds.row_ids.empty() ? row : ds.row_ids[row]) ||
        std::stod(label) != predicted[row])
      throw std::runtime_error("fresh-process reload mismatch");
    ++row;
  }
  if (row != predicted.size())
    throw std::runtime_error("missing assignment rows");
  report.write(
      "fresh_process_verification.json",
      "{\"rows_verified\":" + std::to_string(row) + ",\"matches\":true}\n");
  std::cout << "Verified " << row << " rows in a fresh process\n";
}
