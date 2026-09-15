#include "Model.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc, char **argv) {
  return ml::run_supervised(
      (ml::inference_requested(argc, argv) ? ml::Dataset{}
                                           : ml::load_breast_cancer(DATA_DIR)),
      RUN_OUTPUT_DIR, "end_to_end", {0.001, 0.01, 0.1},
      [](double p) { return course::QDA(p); }, true, argc, argv);
}
