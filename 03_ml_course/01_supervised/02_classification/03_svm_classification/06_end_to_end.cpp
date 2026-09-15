// Final project: svm classification. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "SVM.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_breast_cancer(DATA_DIR),RUN_OUTPUT_DIR,"svm classification",{0.1,1,10},
    [](double p) { return ml::LinearSVM(p,0.01,800); },true,argc,argv);
}
