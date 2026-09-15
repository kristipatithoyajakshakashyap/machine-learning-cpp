// 11_final_pipeline/predict.cpp
// Fresh-process inference: reload a saved capstone model and verify that its
// predictions match the ones written by the training run (s11_05_eval).
//
// Reads:   argv[1] model_state.txt (written by Model::save),
//          argv[2] input CSV with the raw Titanic columns,
//          argv[3] expected predictions CSV (PassengerId,actual,prediction).
// Writes:  RUN_OUTPUT_DIR/predictions.csv and RUN_OUTPUT_DIR/reload_verification.json
//          (results/predict_results/).
// Run:     cmake --build --preset course --target s11_predict
//          then build/02_data_science/11_final_pipeline/s11_predict
//            <model_state.txt> <test_features.csv> <predictions.csv>
//          Exit code 2 on bad usage; throws (non-zero) if predictions differ.
//
// dsts functions used: read_csv, DataFrame::numeric, DataFrame::rows.

#include "pipeline.hpp"
// Entry point: load model, predict on the input CSV, compare against the expected
// predictions row by row, then write predictions.csv and a JSON verification stamp.
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"Usage: s11_predict model_state.txt input.csv expected_predictions.csv\n";return 2;}
  const auto model=capstone::Model::load(argv[1]);
  const auto table=dsts::read_csv(argv[2]);
  const auto pred=model.predict(table);
  const auto expected=dsts::read_csv(argv[3]);
  if(expected.rows()!=pred.size())throw std::runtime_error("reload row count mismatch");
  auto out=capstone::output(std::filesystem::path(RUN_OUTPUT_DIR)/"predictions.csv");out<<"PassengerId,prediction\n";
  // Row-by-row: both the prediction and the PassengerId alignment must match.
  for(size_t r=0;r<pred.size();++r){
    if(*expected.numeric("prediction")[r]!=pred[r]||expected.numeric("PassengerId")[r]!=table.numeric("PassengerId")[r])
      throw std::runtime_error("fresh-process reload differs");
    out<<*table.numeric("PassengerId")[r]<<','<<pred[r]<<'\n';
  }
  auto verified=capstone::output(std::filesystem::path(RUN_OUTPUT_DIR)/"reload_verification.json");
  verified<<"{\"fresh_process\":true,\"identical_predictions\":true,\"rows\":"<<pred.size()<<"}\n";
  return 0;
}
