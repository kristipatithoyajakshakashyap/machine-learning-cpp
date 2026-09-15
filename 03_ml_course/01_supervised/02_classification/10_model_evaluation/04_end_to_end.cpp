// 03_ml_course/01_supervised/02_classification/10_model_evaluation/04_end_to_end.cpp
// Lesson 04: compare four regression model families on California housing using
// the shared supervised pipeline. The "hyper-parameter" searched by CV is the
// family index itself (0 ridge, 1 kNN, 2 random forest, 3 MLP); only the winner
// is refit and scored on the hold-out split.
//
// Reads:   DATA_DIR/california housing CSV via ml::load_california (skipped when
//          --predict is given; then the saved model under --model is loaded).
// Writes:  results/04_end_to_end_results/ (RUN_OUTPUT_DIR): data/, validation/,
//          evaluation/ (metrics.json, predictions.csv, figures/), model/ state
//          files, inference/reload_verification.json and run_manifest.json.
//          With --predict input.csv --model <dir>: predictions.csv only.
// Run:     cmake --build --preset course --target eval_end_to_end
//          then build/03_ml_course/01_supervised/02_classification/
//          10_model_evaluation/eval_end_to_end [--quick]
// Model-family selection uses training CV; only the winning family sees holdout.
#include "01_supervised/01_regression/01_linear_regression/LinearRegression.hpp"
#include "01_supervised/01_regression/02_knn_regression/KNN.hpp"
#include "01_supervised/01_regression/04_random_forest_regressor/RandomForest.hpp"
#include "01_supervised/02_classification/09_mlp_classifier/MLP.hpp"
#include "helper/pipeline/supervised.hpp"
// Adapter giving four different models one fit/predict/save/load interface so
// the generic pipeline can treat the family choice as a numeric parameter.
//   kind   which member is active: 0 ridge (alpha 1), 1 kNN (k 7),
//          2 random forest (40 trees, depth 8, min leaf 3), 3 MLP (16-8, lr 0.005,
//          150 epochs). The archive stores `kind` first so load() knows which
//          member to read; an out-of-range kind is rejected.
class ComparisonModel {
 public:
  int kind=0; ml::LinearRegression linear; ml::KNNRegressor knn{7}; ml::RandomForest forest{40,0,8,3}; ml::MLP neural{{16,8},1,0.005,150};
  // Select the family; the forest must be told it is doing regression.
  explicit ComparisonModel(int k=0):kind(k) {forest.set_task(ml::TreeTask::Regression);}
  // Train only the active member; the others stay default-constructed.
  void fit(const ml::Mat& X,const ml::Vec& y){if(kind==0)linear.fit_ridge(X,y,1);else if(kind==1)knn.fit(X,y);else if(kind==2)forest.fit(X,y);else neural.fit(X,y);}
  // Predict with the active member.
  ml::Vec predict(const ml::Mat& X)const{if(kind==0)return linear.predict(X);if(kind==1)return knn.predict(X);if(kind==2)return forest.predict(X);return neural.predict(X);}
  // Write `kind`, then the active member (its own tag and state).
  void save(std::ostream& out)const{ml::archive::write(out,kind);if(kind==0)linear.save(out);else if(kind==1)knn.save(out);else if(kind==2)forest.save(out);else neural.save(out);}
  // Read `kind`, validate it, then load the matching member.
  void load(std::istream& in){ml::archive::read(in,kind);if(kind<0||kind>3)throw std::runtime_error("Invalid comparison model");if(kind==0)linear.load(in);else if(kind==1)knn.load(in);else if(kind==2)forest.load(in);else neural.load(in);}
};
// The lambda turns a candidate value into a model; `false` = regression task.
int main(int argc,char**argv){return ml::run_supervised(ml::inference_requested(argc,argv)?ml::Dataset{}:ml::load_california(DATA_DIR,false,true),RUN_OUTPUT_DIR,"Model comparison: 0 ridge, 1 kNN, 2 forest, 3 MLP",{0,1,2,3},[](double p){return ComparisonModel(static_cast<int>(p));},false,argc,argv);}
