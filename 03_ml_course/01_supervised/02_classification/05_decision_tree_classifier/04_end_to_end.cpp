// Final project: decision tree classifier. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "DecisionTree.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_breast_cancer(DATA_DIR),RUN_OUTPUT_DIR,"decision tree classifier",{2,4,8},
    [](double p) { return ml::DecisionTree(ml::TreeTask::Classification,ml::SplitCriterion::Gini,static_cast<size_t>(p),3); },true,argc,argv);
}
