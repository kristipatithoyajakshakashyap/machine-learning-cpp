// Final project: decision tree regressor. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "DecisionTree.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_california(DATA_DIR,false,true),RUN_OUTPUT_DIR,"decision tree regressor",{3,6,10},
    [](double p) { return ml::DecisionTree(ml::TreeTask::Regression,ml::SplitCriterion::Gini,static_cast<size_t>(p),5); },false,argc,argv);
}
