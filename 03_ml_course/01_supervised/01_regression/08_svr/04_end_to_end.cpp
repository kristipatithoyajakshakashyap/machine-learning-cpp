#include "Model.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc, char **argv) {
  return ml::run_supervised(
      (ml::inference_requested(argc, argv) ? ml::Dataset{}
                                           : ml::load_diabetes(DATA_DIR)),
      RUN_OUTPUT_DIR, "end_to_end", {1.0, 10.0, 100.0},
      [](double p) { return course::SVR(p, 0.1, true, 0.1); }, false, argc,
      argv);
}
