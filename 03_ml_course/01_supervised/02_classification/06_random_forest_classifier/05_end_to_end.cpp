// Final project: random forest classifier. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "RandomForest.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_breast_cancer(DATA_DIR),RUN_OUTPUT_DIR,"random forest classifier",{3,6,10},
    [](double p) { ml::RandomForest m(40,0,static_cast<size_t>(p),2); m.set_task(ml::TreeTask::Classification); return m; },true,argc,argv);
}
