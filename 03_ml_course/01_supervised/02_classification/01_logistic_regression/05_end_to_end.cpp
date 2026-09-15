// Final project: logistic regression. Read theory.md and math_intuition.md first.
// Training-only CV, independent holdout, raw-data EDA, saved/reloadable pipeline.
// Full run: executable; smoke run: executable --quick.
#include "LogisticRegression.hpp"
#include "helper/pipeline/supervised.hpp"
int main(int argc,char** argv) {
  return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_breast_cancer(DATA_DIR),RUN_OUTPUT_DIR,"logistic regression",{0.1,1,10,100},
    [](double p) { return ml::LogisticRegression(2,p,0.03,600); },true,argc,argv);
}
