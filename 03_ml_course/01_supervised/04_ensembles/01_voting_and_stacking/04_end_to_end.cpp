// End-to-end: hard voting (0), soft voting (1) and stacking (2) of logistic
// regression, 5-NN and a 50-tree forest on wine. The pipeline's
// validation/candidate_scores.csv compares the three modes fold by fold.
// Full run: executable; smoke run: executable --quick.
#include "Model.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc, char **argv) {
  return ml::run_supervised(
      (ml::inference_requested(argc, argv) ? ml::Dataset{}
                                           : ml::load_wine(DATA_DIR)),
      RUN_OUTPUT_DIR, "end_to_end", {0, 1, 2},
      [](double p) { return course::VotingStacking(p); }, true, argc, argv);
}
