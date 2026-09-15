// Final project: adaboost classifier. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "AdaBoost.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_breast_cancer(DATA_DIR),RUN_OUTPUT_DIR,"adaboost classifier",{20,50,100},
    [](double p) { return ml::AdaBoost(static_cast<size_t>(p),0.5); },true,argc,argv);
}
