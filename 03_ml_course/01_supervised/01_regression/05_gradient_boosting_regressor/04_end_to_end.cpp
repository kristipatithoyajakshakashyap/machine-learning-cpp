// Final project: gradient boosting regressor. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "GradientBoostedTrees.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_california(DATA_DIR,false,true),RUN_OUTPUT_DIR,"gradient boosting regressor",{0.03,0.1,0.2},
    [](double p) { ml::GradientBoostedTrees m(50,p,3,5); m.set_task(ml::TreeTask::Regression); return m; },false,argc,argv);
}
