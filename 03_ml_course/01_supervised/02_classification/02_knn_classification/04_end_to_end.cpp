// Final project: knn classification. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "KNN.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_breast_cancer(DATA_DIR),RUN_OUTPUT_DIR,"knn classification",{3,7,15},
    [](double p) { return ml::KNNClassifier(static_cast<size_t>(p)); },true,argc,argv);
}
