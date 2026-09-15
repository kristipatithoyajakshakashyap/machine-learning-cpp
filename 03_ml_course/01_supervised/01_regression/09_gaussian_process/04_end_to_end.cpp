// End-to-end: GP regression on a 300-row subsample of diabetes (fits are
// O(n^3)), length scale selected by training-only CV.  Afterwards the saved
// model is reloaded to write the posterior std for every holdout row.
// Full run: executable; smoke run: executable --quick.
#include "Model.hpp"
#include "helper/pipeline/supervised.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
bool quick_requested(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--quick")
      return true;
  return false;
}
// Reload the artefacts the pipeline just wrote and replay the same holdout
// (train_test_indices 80/20, seed 42) to attach a predictive std to each row.
int write_predictive_std(const ml::Dataset &ds, const std::string &run) {
  try {
    ml::Artifacts a(RUN_OUTPUT_DIR, run);
    course::GaussianProcess gp;
    ml::archive::load_file(a.path("model/model_state.txt"), gp);
    ml::Preprocessor prep;
    ml::archive::load_file(a.path("model/preprocessing_state.txt"), prep);
    ml::TargetTransform target;
    ml::archive::load_file(a.path("model/target_state.txt"), target);
    auto outer = ml::train_test_indices(ds.n(), .8, 42);
    auto te = ml::select_rows(ds, outer.test);
    const ml::Mat Xt = prep.transform(te.X);
    const ml::Vec mean = target.inverse(gp.predict(Xt));
    const ml::Vec sd = gp.predict_std(Xt);
    std::ostringstream csv;
    csv << std::setprecision(10) << "row_id,actual,mean,std,lower95,upper95\n";
    size_t covered = 0;
    for (size_t i = 0; i < te.n(); ++i) {
      const double s = sd[i] * target.scale;
      const size_t id = ds.row_ids.empty() ? outer.test[i] : ds.row_ids.at(outer.test[i]);
      csv << id << ',' << te.y[i] << ',' << mean[i] << ',' << s << ','
          << mean[i] - 1.96 * s << ',' << mean[i] + 1.96 * s << '\n';
      if (std::fabs(te.y[i] - mean[i]) <= 1.96 * s)
        ++covered;
    }
    a.write("evaluation/predictive_std.csv", csv.str());
    std::cout << "Predictive std (latent, no noise): " << covered << '/' << te.n()
              << " holdout rows inside the 95% band; length scale "
              << gp.length_scale() << " -> " << a.path("evaluation/predictive_std.csv")
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "predictive_std: " << e.what() << '\n';
    return 1;
  }
}
} // namespace

int main(int argc, char **argv) {
  const bool inference = ml::inference_requested(argc, argv);
  ml::Dataset ds = inference ? ml::Dataset{}
                             : ml::subsample(ml::load_diabetes(DATA_DIR), 300, 42);
  int status = ml::run_supervised(
      ds, RUN_OUTPUT_DIR, "end_to_end", {0.3, 1, 3, 10},
      [](double p) { return course::GaussianProcess(p, 1.0, 0.1); }, false,
      argc, argv);
  if (status != 0 || inference)
    return status;
  return write_predictive_std(ds, quick_requested(argc, argv) ? "quick" : "full");
}
