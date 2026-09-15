// Final project: mlp regressor. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "MLP.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_california(DATA_DIR,false,true),RUN_OUTPUT_DIR,"mlp regressor",{0.001,0.005,0.01},
    [](double p) { return ml::MLP({16,8},1,p,150); },false,argc,argv);
}
