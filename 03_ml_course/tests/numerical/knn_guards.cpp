// 03_ml_course/tests/numerical/knn_guards.cpp
//
// Purpose : input-validation and persistence guards for KNNRegressor and
//           KNNClassifier. Every "rejects" line proves that invalid input
//           throws instead of silently producing a wrong model; the fixture
//           lines pin down tie-breaking and averaging semantics.
// Inputs  : none (tiny in-memory matrices; no data files, no defines used).
// Outputs : prints only. Exit 0 on success, 1 after the first failed check.
// Target  : ml_knn_guards (CTest: ml_knn_guards)
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "01_supervised/01_regression/02_knn_regression/KNN.hpp"

int main(){
  size_t checks=0;
  // check: count the assertion and throw on failure (non-zero exit for CTest).
  auto check=[&](bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);};
  // rejects: true when the action throws any std::exception.
  auto rejects=[](const auto& action){try{action();}catch(const std::exception&){return true;}return false;};
  const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
  try{
    // Both models share the fit() validation rules, so run the same guards twice.
    for(bool classification:{false,true}){
      auto fit=[&](const ml::Mat& X,const ml::Vec& y,size_t k=1){
        if(classification){ml::KNNClassifier model(k);model.fit(X,y);}
        else{ml::KNNRegressor model(k);model.fit(X,y);}
      };
      check(rejects([&]{fit({},{});}),"empty training data");           // no rows at all
      check(rejects([&]{fit({{}},{0});}),"zero features");               // a row with no columns
      // rows differ in width
      check(rejects([&]{fit({{0},{1,2}},{0,1});}),"ragged training features");
      check(rejects([&]{fit({{0},{1}},{0});}),"target row mismatch");    // |y| != |X|
      check(rejects([&]{fit({{0},{1}},{0,1},0);}),"zero k");             // k must be >= 1
      check(rejects([&]{fit({{0},{1}},{0,1},3);}),"k exceeds training rows"); // k > n rows
      check(rejects([&]{fit({{nan},{1}},{0,1});}),"NaN training feature");
      check(rejects([&]{fit({{0},{inf}},{0,1});}),"infinite training feature");
      check(rejects([&]{fit({{0},{1}},{nan,1});}),"NaN target");
      check(rejects([&]{fit({{0},{1}},{0,inf});}),"infinite target");
    }
    // Classifier labels must be contiguous class codes 0..K-1: negative,
    // fractional, gapped (0,2), single-class (1,1) and huge values all fail.
    for(const ml::Vec& y:std::vector<ml::Vec>{{-1,0},{0,.5},{0,2},{1,1},{0,1e100}})
      check(rejects([&]{ml::KNNClassifier model(1);model.fit({{0},{1}},y);}),"invalid class encoding");
    // Regression averages the k=2 neighbours: (-3.5 + 1.5) / 2 = -1, so
    // negative fractional targets are not truncated into class codes.
    ml::KNNRegressor reg(2);reg.fit({{0},{2}},{-3.5,1.5});
    check(reg.predict({{1}})[0]==-1,"signed fractional regression target retained");
    // Classifier with k=2 over classes {1,0}: query 0.1 sees both neighbours,
    // one vote each, so the tie must resolve to the smallest class code (0).
    ml::KNNClassifier classifier(2);classifier.fit({{0},{2}},{1,0});
    check(classifier.predict({{.1}})[0]==0,"tie breaks by smallest class code");
    // Vote fractions: one neighbour per class -> exactly [0.5, 0.5].
    check(classifier.predict_proba({{1}})==ml::Mat{{.5,.5}},"probability vote fixture");
    // Query-time guards: empty row, wrong width, NaN, inf and ragged batches
    // must all be rejected by predict and predict_proba alike.
    for(const ml::Mat& query:std::vector<ml::Mat>{{{}},{{1,2}},{{nan}},{{inf}},{{0},{1,2}}}){
      check(rejects([&]{reg.predict(query);}),"regression prediction input guard");
      check(rejects([&]{classifier.predict(query);}),"classification prediction input guard");
      check(rejects([&]{classifier.predict_proba(query);}),"probability prediction input guard");
    }
    // An empty batch is valid input and yields an empty result, not an error.
    check(reg.predict({}).empty()&&classifier.predict({}).empty()&&classifier.predict_proba({}).empty(),"empty prediction batches safe");
    // Archive round trip: save -> load into a fresh object -> identical output.
    std::stringstream saved_reg;reg.save(saved_reg);ml::KNNRegressor restored_reg;restored_reg.load(saved_reg);
    check(restored_reg.predict({{1}})==reg.predict({{1}}),"regression archive roundtrip");
    std::stringstream saved_classifier;classifier.save(saved_classifier);ml::KNNClassifier restored_classifier;restored_classifier.load(saved_classifier);
    check(restored_classifier.predict_proba({{1}})==classifier.predict_proba({{1}}),"classifier archive roundtrip");
    // Hand-built archives with the right header but inconsistent payloads.
    // The archive layout mirrors KNN::save: name, k, [classes,] X, y.
    auto bad_reg=[&](size_t k,const ml::Mat& X,const ml::Vec& y){std::stringstream in;ml::archive::write(in,std::string("KNNRegressor"),k,X,y);reg.load(in);};
    auto bad_class=[&](size_t k,size_t classes,const ml::Mat& X,const ml::Vec& y){std::stringstream in;ml::archive::write(in,std::string("KNNClassifier"),k,classes,X,y);classifier.load(in);};
    check(rejects([&]{bad_reg(0,{{0}},{1});}),"archive k validation");                 // k == 0
    check(rejects([&]{bad_reg(1,{{0},{1,2}},{1,2});}),"archive rectangular validation"); // ragged X
    // |y| != |X|
    check(rejects([&]{bad_reg(1,{{0},{1}},{1});}),"archive target shape validation");
    // negative label
    check(rejects([&]{bad_class(1,2,{{0},{1}},{-1,0});}),"archive label validation");
    // 3 declared, 2 seen
    check(rejects([&]{bad_class(1,3,{{0},{1}},{0,1});}),"archive class-count consistency");
    // label 2 with 1 class
    check(rejects([&]{bad_class(1,1,{{0},{1}},{0,2});}),"archive class contiguity");
    check(rejects([&]{std::stringstream in("\"KNNClassifier\"\n1\n2\n");classifier.load(in);}),"truncated archive");
    check(rejects([&]{std::stringstream in;reg.save(in);classifier.load(in);}),"wrong archive model type");
    // A rejected load must leave the previously fitted model untouched.
    check(reg.k()==2&&reg.predict({{1}})[0]==-1,"failed regression load preserves fitted state");
    check(classifier.k()==2&&classifier.predict_proba({{1}})==ml::Mat{{.5,.5}},"failed classifier load preserves fitted state");
    std::cout<<checks<<" KNN guard and persistence checks passed\n";return 0;
  }catch(const std::exception& error){std::cerr<<"Failure after "<<checks<<" checks: "<<error.what()<<'\n';return 1;}
}
