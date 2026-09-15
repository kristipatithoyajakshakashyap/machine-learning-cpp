// Final project: knn regression. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "KNN.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_california(DATA_DIR,false,true),RUN_OUTPUT_DIR,"knn regression",{3,7,15},
    [](double p) { return ml::KNNRegressor(static_cast<size_t>(p)); },false,argc,argv);
}
