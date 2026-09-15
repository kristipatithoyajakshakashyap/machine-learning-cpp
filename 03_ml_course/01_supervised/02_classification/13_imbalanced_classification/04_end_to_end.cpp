// Imbalanced binary classification. The dataset is breast_cancer with the
// malignant class deliberately subsampled to 30 rows (about 8% positives).
// Full run: executable; smoke run: executable --quick.
#include "Model.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc, char **argv) {
  return ml::run_supervised(
      ml::inference_requested(argc, argv)
          ? ml::Dataset{}
          : ml::make_imbalanced(ml::load_breast_cancer(DATA_DIR), 1.0, 30, 42),
      RUN_OUTPUT_DIR, "end_to_end", {1, 2, 4, 8},
      [](double w) { return course::WeightedLogistic(w); }, true, argc, argv);
}
